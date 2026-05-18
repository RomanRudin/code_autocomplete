"""
modules/Evaluation.py — Unified evaluation for all four model variants.

Supports:
  • Custom TokenModel (baseline / BPE+RoPE)        — `evaluate_token_model`
  • HuggingFace CausalLM    (CodeGen)              — `evaluate_token_model`
  • Custom LineModel  (baseline / BPE+RoPE)        — `evaluate_line_model`
  • HuggingFace Seq2Seq     (CodeT5)               — `evaluate_line_model`

Usage from a notebook:

    from modules.Evaluation import evaluate_token_model, evaluate_line_model

    res_token = evaluate_token_model(
        model           = tok_model,
        tokenizer       = tokenizer,            # any of the 4 tokenizer types
        eval_texts      = val_texts,
        device          = device,
        ctx             = 128,
        n_samples       = 2000,
        is_hf           = False,                # True for HF CausalLM (CodeGen)
        out_dir         = "eval/token_baseline",
        title           = "Token Baseline",
    )

    res_line = evaluate_line_model(
        model           = line_model,
        tokenizer       = tokenizer,
        eval_texts      = val_texts,
        device          = device,
        n_samples       = 500,
        is_hf_seq2seq   = True,                 # True for T5
        out_dir         = "eval/line_codet5",
        title           = "Line CodeT5",
    )

Both functions return a metrics dict and save:
  - eval_results.json   (raw numbers)
  - eval_dashboard.png  (visual dashboard)
"""

from __future__ import annotations
import os, json, time, math, ast, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS — Python token categorisation (for per-type accuracy)
# ════════════════════════════════════════════════════════════════════════════

PYTHON_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
}
PUNCT_CHARS = set("()[]{}.,;:=+-*/!&|~^@#\"'`<>%\\")


def classify_token(tok_str: str) -> str:
    """Best-effort token-type classifier."""
    s = tok_str.strip()
    # leading-whitespace tokens (BPE often adds a "Ġ" or " " prefix)
    if not s:
        return "indent"
    if s.startswith("Ġ"): # GPT-2 / RoBERTa BPE space marker
        s = s[1:]
    if not s:
        return "indent"
    if s in PYTHON_KEYWORDS:
        return "keyword"
    if s.replace(".", "", 1).replace("_", "").isdigit():
        return "number"
    if all(c in PUNCT_CHARS for c in s):
        return "punctuation"
    if s.replace("_", "").isalnum():
        return "identifier"
    return "other"


# ════════════════════════════════════════════════════════════════════════════
#  N-GRAM METRICS (no external deps — no NLTK required)
# ════════════════════════════════════════════════════════════════════════════

def _ngrams(seq: List, n: int) -> Dict[tuple, int]:
    counts: Dict[tuple, int] = {}
    for i in range(len(seq) - n + 1):
        g = tuple(seq[i: i + n])
        counts[g] = counts.get(g, 0) + 1
    return counts


def bleu4(ref: List, hyp: List) -> float:
    """Corpus-style BLEU-4 with brevity penalty. Single sample → score in [0,1]."""
    if not hyp or not ref:
        return 0.0
    bp = min(1.0, math.exp(1 - len(ref) / max(1, len(hyp))))
    log_sum = 0.0
    for n in range(1, 5):
        r_ng = _ngrams(ref, n)
        h_ng = _ngrams(hyp, n)
        if not h_ng:
            return 0.0
        clip = sum(min(c, r_ng.get(g, 0)) for g, c in h_ng.items())
        tot  = sum(h_ng.values())
        if clip == 0:
            return 0.0
        log_sum += math.log(clip / tot)
    return bp * math.exp(log_sum / 4)


def chrf(ref: str, hyp: str, n: int = 6) -> float:
    """Character-level n-gram F-score (chrF), averaged over n=1..6."""
    if not ref or not hyp:
        return 0.0
    fs = []
    for i in range(1, n + 1):
        r = _ngrams(list(ref), i)
        h = _ngrams(list(hyp), i)
        if not h or not r:
            fs.append(0.0); continue
        prec = sum(min(c, r.get(g, 0)) for g, c in h.items()) / max(1, sum(h.values()))
        rec  = sum(min(c, h.get(g, 0)) for g, c in r.items()) / max(1, sum(r.values()))
        f    = 2 * prec * rec / max(1e-9, prec + rec)
        fs.append(f)
    return float(np.mean(fs))


def edit_similarity(ref: List, hyp: List) -> float:
    """1 - normalised Levenshtein distance.  1.0 = identical, 0.0 = totally different."""
    if not ref and not hyp: return 1.0
    if not ref or not hyp:  return 0.0
    a, b = ref, hyp
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def keyword_match(ref_str: str, hyp_str: str) -> float:
    """Fraction of Python keywords from ref that also appear in hyp."""
    ref_kw = {w for w in ref_str.split() if w in PYTHON_KEYWORDS}
    if not ref_kw:
        return 1.0   # vacuous — no keywords to match
    hyp_kw = {w for w in hyp_str.split() if w in PYTHON_KEYWORDS}
    return len(ref_kw & hyp_kw) / len(ref_kw)


def identifier_match(ref_str: str, hyp_str: str) -> float:
    """Fraction of identifiers from ref also in hyp (excluding keywords)."""
    def ids(s):
        return {w for w in s.split()
                if w.replace("_", "").isalnum()
                and not w[0].isdigit()
                and w not in PYTHON_KEYWORDS}
    r, h = ids(ref_str), ids(hyp_str)
    if not r:
        return 1.0
    return len(r & h) / len(r)


def is_valid_python(prefix: str, completion: str) -> bool:
    """True if `prefix + completion` parses as Python."""
    try:
        ast.parse(prefix + completion)
        return True
    except SyntaxError:
        return False
    except (ValueError, TypeError):
        return False


def indent_match(ref: str, hyp: str) -> float:
    """Compare leading-whitespace count of first non-empty line."""
    def lead(s):
        for line in s.splitlines():
            if line.strip(): return len(line) - len(line.lstrip())
        return 0
    return 1.0 if lead(ref) == lead(hyp) else 0.0


# ════════════════════════════════════════════════════════════════════════════
#  TOKENIZER ABSTRACTION — handles all four tokenizer types
# ════════════════════════════════════════════════════════════════════════════

class TokenizerWrapper:
    """
    Thin uniform wrapper so eval code doesn't care which tokenizer it has.
    Detects type at construction.
    """
    def __init__(self, tok):
        self.tok = tok
        # detect tokenizer family
        if hasattr(tok, "encode") and hasattr(tok, "id2token") and hasattr(tok, "vocab"):
            self.kind = "custom"          # CodeTokenizer / BPECodeTokenizer
        elif hasattr(tok, "encode") and hasattr(tok, "vocab_size") and hasattr(tok, "decode"):
            self.kind = "hf"              # HF AutoTokenizer
        else:
            raise ValueError(f"Unknown tokenizer type: {type(tok)}")

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        if self.kind == "custom":
            return self.tok.encode(text)
        # HF
        return self.tok.encode(text, add_special_tokens=add_special)

    def decode(self, ids: List[int]) -> str:
        if self.kind == "custom":
            return self.tok.decode(ids)
        return self.tok.decode(ids, skip_special_tokens=True)

    def id_to_str(self, idx: int) -> str:
        if self.kind == "custom":
            return self.tok.id2token.get(idx, "")
        return self.tok.convert_ids_to_tokens(idx) or ""

    @property
    def vocab_size(self) -> int:
        if self.kind == "custom":
            return self.tok.vocab
        return self.tok.vocab_size

    @property
    def pad_id(self) -> int:
        if self.kind == "custom":
            return self.tok.pad_id
        pid = self.tok.pad_token_id
        return pid if pid is not None else self.tok.eos_token_id

    @property
    def eos_id(self) -> int:
        if self.kind == "custom":
            return self.tok.eos_id
        return self.tok.eos_token_id

    @property
    def bos_id(self) -> int:
        if self.kind == "custom":
            return self.tok.bos_id
        bid = getattr(self.tok, "bos_token_id", None)
        return bid if bid is not None else self.eos_id


# ════════════════════════════════════════════════════════════════════════════
#  TOKEN MODEL EVALUATION
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_token_model(
    model,
    tokenizer,
    eval_texts: List[str],
    device: torch.device,
    ctx: int = 128,
    n_samples: int = 2000,
    is_hf: bool = False,
    out_dir: str = "eval_results",
    title: str = "Token Model",
) -> Dict[str, Any]:
    """
    Evaluate a next-token model.  Works with custom TokenModel and HF CausalLM.

    Returns dict with keys:
        perplexity, top1_acc, top5_acc, mrr, ece, entropy,
        type_accuracy{keyword,identifier,punctuation,indent,number,other},
        latency_p50, latency_p90, latency_p99, latency_mean,
        calibration_curve{bin_centers, accuracy, confidence, count}
    """
    os.makedirs(out_dir, exist_ok=True)
    tw = TokenizerWrapper(tokenizer)
    model.eval()

    # ── build a single token stream from eval texts ──────────────────────────
    print(f"[Eval Token] tokenising {len(eval_texts)} texts …")
    all_ids: List[int] = []
    for t in eval_texts:
        all_ids.extend(tw.encode(t))
    if len(all_ids) < ctx + 2:
        raise ValueError(f"Eval data too short: {len(all_ids)} tokens, need > {ctx+2}")

    # ── pick random starting indices ─────────────────────────────────────────
    max_start = len(all_ids) - ctx - 1
    n_samples = min(n_samples, max_start)
    indices   = random.sample(range(max_start), n_samples)
    print(f"[Eval Token] {n_samples} samples, ctx={ctx}")

    # ── accumulators ─────────────────────────────────────────────────────────
    total_nll  = 0.0
    total_tok  = 0
    top1 = top5 = 0
    mrr_sum    = 0.0
    entropies  = []
    latencies  = []

    # calibration: 10 confidence bins
    n_bins   = 10
    bin_acc  = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_cnt  = np.zeros(n_bins, dtype=np.int64)

    # per-type accuracy
    type_correct = defaultdict(int)
    type_total   = defaultdict(int)

    PAD = tw.pad_id

    for idx in indices:
        chunk = torch.tensor(
            all_ids[idx: idx + ctx + 1], dtype=torch.long, device=device
        ).unsqueeze(0)
        x = chunk[:, :-1]
        y = chunk[:,  1:]

        t0 = time.perf_counter()
        if is_hf:
            logits = model(input_ids=x).logits        # (1, ctx, V)
        else:
            logits = model(x)                         # (1, ctx, V)
        latencies.append((time.perf_counter() - t0) * 1000)

        # full-sequence NLL for perplexity
        log_probs = F.log_softmax(logits, dim=-1)
        nll       = F.nll_loss(log_probs.view(-1, log_probs.size(-1)),
                               y.view(-1), reduction="sum",
                               ignore_index=PAD).item()
        n_valid   = (y != PAD).sum().item()
        total_nll += nll
        total_tok += n_valid

        # last-position metrics (one prediction per sample for top-k / MRR)
        last_logits = logits[0, -1]
        last_probs  = F.softmax(last_logits, dim=-1)
        true_id     = y[0, -1].item()
        if true_id == PAD:
            continue

        # top-1 / top-5 / MRR
        sorted_ids = torch.argsort(last_logits, descending=True).tolist()
        rank       = sorted_ids.index(true_id) + 1
        top1      += int(rank == 1)
        top5      += int(rank <= 5)
        mrr_sum   += 1.0 / rank

        # entropy (decisiveness)
        ent = -(last_probs * (last_probs.clamp(min=1e-12).log())).sum().item()
        entropies.append(ent)

        # calibration: bucket by max-prob
        conf, pred = last_probs.max(dim=-1)
        conf_v = conf.item()
        correct = int(pred.item() == true_id)
        bin_idx = min(int(conf_v * n_bins), n_bins - 1)
        bin_acc[bin_idx]  += correct
        bin_conf[bin_idx] += conf_v
        bin_cnt[bin_idx]  += 1

        # per-type accuracy
        ttype = classify_token(tw.id_to_str(true_id))
        type_total[ttype]   += 1
        type_correct[ttype] += correct

    # ── aggregate ────────────────────────────────────────────────────────────
    perplexity = math.exp(min(total_nll / max(1, total_tok), 20))

    type_accuracy = {
        k: type_correct[k] / type_total[k] if type_total[k] else 0.0
        for k in ["keyword", "identifier", "punctuation", "indent", "number", "other"]
    }

    # ECE
    ece = 0.0
    for i in range(n_bins):
        if bin_cnt[i] > 0:
            acc_i  = bin_acc[i]  / bin_cnt[i]
            conf_i = bin_conf[i] / bin_cnt[i]
            ece   += (bin_cnt[i] / max(1, n_samples)) * abs(acc_i - conf_i)

    cal_curve = {
        "bin_centers": [(i + 0.5) / n_bins for i in range(n_bins)],
        "accuracy":    [bin_acc[i] / bin_cnt[i] if bin_cnt[i] else 0.0 for i in range(n_bins)],
        "confidence":  [bin_conf[i] / bin_cnt[i] if bin_cnt[i] else 0.0 for i in range(n_bins)],
        "count":       bin_cnt.tolist(),
    }

    res = {
        "model_type":    "hf_causal" if is_hf else "custom_token",
        "n_samples":     n_samples,
        "ctx":           ctx,
        "perplexity":    perplexity,
        "top1_acc":      top1 / n_samples,
        "top5_acc":      top5 / n_samples,
        "mrr":           mrr_sum / n_samples,
        "ece":           ece,
        "entropy":       float(np.mean(entropies)) if entropies else 0.0,
        "type_accuracy": type_accuracy,
        "type_counts":   {k: type_total[k] for k in type_accuracy},
        "latency_mean":  float(np.mean(latencies)),
        "latency_p50":   float(np.percentile(latencies, 50)),
        "latency_p90":   float(np.percentile(latencies, 90)),
        "latency_p99":   float(np.percentile(latencies, 99)),
        "calibration_curve": cal_curve,
    }

    json_path = os.path.join(out_dir, "eval_token_results.json")
    with open(json_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[Eval Token] results → {json_path}")

    plot_path = os.path.join(out_dir, "eval_token_dashboard.png")
    _plot_token_dashboard(res, title, plot_path)
    return res


# ════════════════════════════════════════════════════════════════════════════
#  LINE MODEL EVALUATION
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_line_model(
    model,
    tokenizer,
    eval_texts: List[str],
    device: torch.device,
    n_samples: int = 500,
    max_new: int = 64,
    is_hf_seq2seq: bool = False,
    out_dir: str = "eval_results",
    title: str = "Line Model",
    min_line_chars: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate a line completion model.  Works with custom LineModel and HF Seq2Seq (T5).
    """
    os.makedirs(out_dir, exist_ok=True)
    tw = TokenizerWrapper(tokenizer)
    model.eval()

    # ── build (prefix_str, suffix_str) sample pairs ──────────────────────────
    print(f"[Eval Line] building samples from {len(eval_texts)} texts …")
    samples: List[Tuple[str, str]] = []
    for text in eval_texts:
        for line in text.splitlines():
            line = line.rstrip()
            if len(line.strip()) < min_line_chars:
                continue
            words = line.split()
            if len(words) < 4:
                continue
            cut = max(2, int(len(words) * 0.5))
            samples.append((" ".join(words[:cut]), " ".join(words[cut:])))
    if not samples:
        raise ValueError("No suitable lines found in eval_texts")
    random.shuffle(samples)
    samples = samples[:n_samples]
    n_samples = len(samples)
    print(f"[Eval Line] {n_samples} samples")

    # ── accumulators ─────────────────────────────────────────────────────────
    exact = prefix20 = 0
    bleu_l, chrf_l, edit_l = [], [], []
    kw_l, id_l, syn_l, ind_l = [], [], [], []
    latencies = []

    for prefix, suffix in samples:
        ref_ids = tw.encode(suffix, add_special=False)
        if not ref_ids:
            continue

        prefix_ids = tw.encode(prefix, add_special=(not is_hf_seq2seq))

        t0 = time.perf_counter()
        if is_hf_seq2seq:
            inp = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            out = model.generate(
                inp,
                max_new_tokens=max_new,
                num_beams=1,
                do_sample=False,
                pad_token_id=tw.pad_id,
            )
            # T5 starts with decoder_start_token; trim it
            hyp_ids = out[0].tolist()
            if hyp_ids and hyp_ids[0] in (tw.pad_id, tw.bos_id):
                hyp_ids = hyp_ids[1:]
        else:
            # custom model — has .generate(prefix_ids, ...)
            hyp_ids = model.generate(
                prefix_ids,
                max_new=max_new,
                temperature=1.0,        # near-greedy via top_k=1
                top_k=1,
                tokenizer=tokenizer,
            )
        latencies.append((time.perf_counter() - t0) * 1000)

        # strip eos from prediction
        hyp_ids = [i for i in hyp_ids if i != tw.eos_id]

        hyp_str = tw.decode(hyp_ids)
        ref_str = suffix

        exact    += int(hyp_ids == ref_ids)
        prefix20 += int(hyp_ids[:20] == ref_ids[:20])
        bleu_l.append(bleu4(ref_ids, hyp_ids))
        chrf_l.append(chrf(ref_str, hyp_str))
        edit_l.append(edit_similarity(ref_ids, hyp_ids))
        kw_l.append(keyword_match(ref_str, hyp_str))
        id_l.append(identifier_match(ref_str, hyp_str))
        syn_l.append(float(is_valid_python(prefix + " ", " " + hyp_str)))
        ind_l.append(indent_match(ref_str, hyp_str))

    n = max(1, len(bleu_l))
    res = {
        "model_type":      "hf_seq2seq" if is_hf_seq2seq else "custom_line",
        "n_samples":       n,
        "exact_match":     exact / n,
        "prefix20_match":  prefix20 / n,
        "bleu4":           float(np.mean(bleu_l)),
        "chrf":            float(np.mean(chrf_l)),
        "edit_similarity": float(np.mean(edit_l)),
        "keyword_match":   float(np.mean(kw_l)),
        "identifier_match":float(np.mean(id_l)),
        "syntax_valid":    float(np.mean(syn_l)),
        "indent_match":    float(np.mean(ind_l)),
        "latency_mean":    float(np.mean(latencies)),
        "latency_p50":     float(np.percentile(latencies, 50)),
        "latency_p90":     float(np.percentile(latencies, 90)),
        "latency_p99":     float(np.percentile(latencies, 99)),
    }
    res["codebleu"] = (
        0.25 * res["bleu4"] +
        0.25 * res["chrf"]  +
        0.25 * res["keyword_match"] +
        0.25 * res["identifier_match"]
    )

    json_path = os.path.join(out_dir, "eval_line_results.json")
    with open(json_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[Eval Line] results → {json_path}")

    plot_path = os.path.join(out_dir, "eval_line_dashboard.png")
    _plot_line_dashboard(res, title, plot_path)
    return res


# ════════════════════════════════════════════════════════════════════════════
#  COMPARE — render side-by-side dashboard for multiple models
# ════════════════════════════════════════════════════════════════════════════

def compare_models(
    results: Dict[str, Dict[str, Any]],
    out_path: str = "eval_results/comparison.png",
    kind: str = "token",       # or "line"
):
    """
    `results` = {"baseline": dict_from_evaluate_*, "codegen": dict_from_evaluate_*, ...}
    Renders grouped-bar dashboard.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    _STYLE.apply()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), facecolor=_STYLE.DARK)
    names = list(results.keys())
    if kind == "token":
        primary = ["top1_acc", "top5_acc", "mrr", "ece"]
        latency = ["latency_p50", "latency_p90", "latency_p99"]
    else:
        primary = ["exact_match", "bleu4", "chrf", "edit_similarity",
                   "syntax_valid", "indent_match"]
        latency = ["latency_p50", "latency_p90", "latency_p99"]

    _grouped_bars(axes[0], primary, results, names, "Quality metrics")
    _grouped_bars(axes[1], latency, results, names, "Latency (ms)")

    fig.suptitle(f"Model comparison — {kind}", fontsize=14, color=_STYLE.BLUE)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Compare] → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
#  PLOTTING — shared style + dashboards
# ════════════════════════════════════════════════════════════════════════════

class _STYLE:
    DARK   = "#0d1117"; MID = "#161b22"; GRID = "#21262d"
    BLUE   = "#58a6ff"; GREEN = "#3fb950"; ORG = "#ffa657"
    RED    = "#f78166"; PURPLE = "#d2a8ff"; TXT = "#c9d1d9"
    @classmethod
    def apply(cls):
        plt.rcParams.update({
            "axes.facecolor": cls.MID, "axes.edgecolor": cls.GRID,
            "axes.labelcolor": cls.TXT, "xtick.color": cls.TXT,
            "ytick.color": cls.TXT, "text.color": cls.TXT,
            "grid.color": cls.GRID,
        })


def _bar(ax, labels, values, colors, title, ylim=(0, 1), fmt="{:.3f}"):
    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + (ylim[1] - ylim[0]) * 0.01,
                fmt.format(v), ha="center", fontsize=8, color=_STYLE.TXT)
    ax.set_title(title, fontsize=10, color=_STYLE.BLUE, pad=6)
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", lw=0.5, zorder=0)
    ax.tick_params(axis="x", labelsize=8)


def _grouped_bars(ax, metrics, results, names, title):
    n_groups = len(metrics)
    n_models = len(names)
    width    = 0.8 / max(1, n_models)
    palette  = [_STYLE.BLUE, _STYLE.GREEN, _STYLE.ORG, _STYLE.PURPLE, _STYLE.RED]
    x        = np.arange(n_groups)
    for i, name in enumerate(names):
        vals = [results[name].get(m, 0.0) for m in metrics]
        ax.bar(x + i * width, vals, width,
               label=name, color=palette[i % len(palette)], zorder=3)
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=8)
    ax.set_title(title, fontsize=10, color=_STYLE.BLUE, pad=6)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", lw=0.5, zorder=0)


def _plot_token_dashboard(r: Dict, title: str, out_path: str):
    _STYLE.apply()
    fig = plt.figure(figsize=(18, 11), facecolor=_STYLE.DARK)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.30)

    # 0 — Core accuracy
    ax = fig.add_subplot(gs[0, 0])
    _bar(ax,
         ["Top-1", "Top-5", "MRR"],
         [r["top1_acc"], r["top5_acc"], r["mrr"]],
         [_STYLE.BLUE, _STYLE.GREEN, _STYLE.ORG],
         "Core accuracy")

    # 1 — Per-type accuracy
    ax = fig.add_subplot(gs[0, 1])
    types = list(r["type_accuracy"].keys())
    vals  = [r["type_accuracy"][t] for t in types]
    cols  = [_STYLE.BLUE, _STYLE.ORG, _STYLE.GREEN, _STYLE.PURPLE, _STYLE.RED, _STYLE.TXT][:len(types)]
    _bar(ax, types, vals, cols, "Accuracy by token type")
    ax.tick_params(axis="x", labelrotation=20)

    # 2 — Calibration: ECE + entropy
    ax = fig.add_subplot(gs[0, 2])
    _bar(ax,
         ["ECE ↓", "Entropy"],
         [r["ece"], r["entropy"]],
         [_STYLE.RED, _STYLE.PURPLE],
         "Confidence calibration",
         ylim=(0, max(1.0, r["entropy"] * 1.2)),
         fmt="{:.3f}")

    # 3 — Reliability diagram
    ax = fig.add_subplot(gs[1, 0])
    cc = r["calibration_curve"]
    ax.plot([0, 1], [0, 1], "--", color=_STYLE.GRID, lw=1, label="perfect")
    ax.bar(cc["bin_centers"], cc["accuracy"], width=0.085,
           color=_STYLE.BLUE, alpha=0.7, label="accuracy", zorder=3)
    ax.plot(cc["bin_centers"], cc["confidence"], "o-",
            color=_STYLE.ORG, lw=1.5, label="confidence")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram", fontsize=10, color=_STYLE.BLUE, pad=6)
    ax.legend(fontsize=8); ax.grid(True, lw=0.5, zorder=0)

    # 4 — Confidence distribution (sample counts per bin)
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(cc["bin_centers"], cc["count"], width=0.085, color=_STYLE.GREEN, zorder=3)
    ax.set_xlabel("Confidence"); ax.set_ylabel("# samples")
    ax.set_title("Confidence distribution", fontsize=10, color=_STYLE.BLUE, pad=6)
    ax.grid(True, lw=0.5, zorder=0)

    # 5 — Latency percentiles
    ax = fig.add_subplot(gs[1, 2])
    _bar(ax,
         ["mean", "p50", "p90", "p99"],
         [r["latency_mean"], r["latency_p50"], r["latency_p90"], r["latency_p99"]],
         [_STYLE.BLUE, _STYLE.GREEN, _STYLE.ORG, _STYLE.RED],
         "Inference latency (ms)",
         ylim=(0, max(r["latency_p99"], 1) * 1.15),
         fmt="{:.1f}")

    # 6 — Perplexity
    ax = fig.add_subplot(gs[2, 0])
    _bar(ax, ["Perplexity"], [r["perplexity"]],
         [_STYLE.ORG], "Perplexity ↓",
         ylim=(0, max(r["perplexity"] * 1.25, 10)),
         fmt="{:.2f}")

    # 7 — Token-type sample distribution
    ax = fig.add_subplot(gs[2, 1])
    types  = list(r["type_counts"].keys())
    counts = [r["type_counts"][t] for t in types]
    cols   = [_STYLE.BLUE, _STYLE.ORG, _STYLE.GREEN, _STYLE.PURPLE, _STYLE.RED, _STYLE.TXT][:len(types)]
    _bar(ax, types, counts, cols, "Eval-set token-type counts",
         ylim=(0, max(counts) * 1.15 if counts else 1), fmt="{:d}")
    ax.tick_params(axis="x", labelrotation=20)

    # 8 — Quality radar (overall snapshot)
    ax = fig.add_subplot(gs[2, 2], polar=True)
    ax.set_facecolor(_STYLE.MID)
    radar_metrics = ["Top-1", "Top-5", "MRR", "1−ECE", "Punct\nAcc", "Indent\nAcc"]
    radar_values  = [
        r["top1_acc"], r["top5_acc"], r["mrr"], 1.0 - r["ece"],
        r["type_accuracy"].get("punctuation", 0.0),
        r["type_accuracy"].get("indent", 0.0),
    ]
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    rv = radar_values + [radar_values[0]]; ra = angles + [angles[0]]
    ax.plot(ra, rv, "o-", color=_STYLE.BLUE, lw=2)
    ax.fill(ra, rv, color=_STYLE.BLUE, alpha=0.2)
    ax.set_xticks(angles); ax.set_xticklabels(radar_metrics, fontsize=8, color=_STYLE.TXT)
    ax.set_ylim(0, 1); ax.set_title("Quality radar", fontsize=10, color=_STYLE.BLUE, pad=15)
    ax.spines["polar"].set_color(_STYLE.GRID)

    fig.suptitle(f"{title} — Token Evaluation  ({r['n_samples']} samples)",
                 fontsize=14, color=_STYLE.BLUE, y=1.00)
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] → {out_path}")


def _plot_line_dashboard(r: Dict, title: str, out_path: str):
    _STYLE.apply()
    fig = plt.figure(figsize=(18, 11), facecolor=_STYLE.DARK)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.30)

    # 0 — String-level metrics
    ax = fig.add_subplot(gs[0, 0])
    _bar(ax,
         ["Exact", "Prefix-20"],
         [r["exact_match"], r["prefix20_match"]],
         [_STYLE.BLUE, _STYLE.GREEN], "Exact / prefix match")

    # 1 — Overlap-style metrics
    ax = fig.add_subplot(gs[0, 1])
    _bar(ax,
         ["BLEU-4", "chrF", "Edit\nsim"],
         [r["bleu4"], r["chrf"], r["edit_similarity"]],
         [_STYLE.BLUE, _STYLE.ORG, _STYLE.GREEN], "Sequence overlap")

    # 2 — Code-specific metrics
    ax = fig.add_subplot(gs[0, 2])
    _bar(ax,
         ["Keyword", "Ident", "Syntax", "Indent"],
         [r["keyword_match"], r["identifier_match"],
          r["syntax_valid"], r["indent_match"]],
         [_STYLE.BLUE, _STYLE.GREEN, _STYLE.ORG, _STYLE.RED],
         "Code-specific metrics")

    # 3 — CodeBLEU composite (bar with components)
    ax = fig.add_subplot(gs[1, 0])
    _bar(ax,
         ["BLEU", "chrF", "Keyword", "Ident", "CodeBLEU"],
         [r["bleu4"], r["chrf"], r["keyword_match"],
          r["identifier_match"], r["codebleu"]],
         [_STYLE.BLUE, _STYLE.ORG, _STYLE.GREEN, _STYLE.PURPLE, _STYLE.RED],
         "CodeBLEU breakdown")
    ax.tick_params(axis="x", labelrotation=20)

    # 4 — Latency
    ax = fig.add_subplot(gs[1, 1])
    _bar(ax,
         ["mean", "p50", "p90", "p99"],
         [r["latency_mean"], r["latency_p50"], r["latency_p90"], r["latency_p99"]],
         [_STYLE.BLUE, _STYLE.GREEN, _STYLE.ORG, _STYLE.RED],
         "Inference latency (ms)",
         ylim=(0, max(r["latency_p99"], 1) * 1.15),
         fmt="{:.1f}")

    # 5 — Sample summary text panel
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    txt = (
        f"Eval samples : {r['n_samples']:,}\n"
        f"Model type   : {r['model_type']}\n\n"
        f"Best metric  : {max(r.items(), key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) and kv[0] not in ('latency_mean','latency_p50','latency_p90','latency_p99','n_samples') else -1)[0]}\n"
        f"CodeBLEU     : {r['codebleu']:.3f}\n"
    )
    ax.text(0.05, 0.5, txt, color=_STYLE.TXT, fontsize=11,
            family="monospace", va="center")
    ax.set_title("Run summary", fontsize=10, color=_STYLE.BLUE, pad=6)

    # 6 — Quality radar
    ax = fig.add_subplot(gs[2, 0], polar=True)
    ax.set_facecolor(_STYLE.MID)
    radar_metrics = ["BLEU-4", "chrF", "Edit\nsim",
                     "Syntax", "Indent", "Keyword"]
    radar_values  = [r["bleu4"], r["chrf"], r["edit_similarity"],
                     r["syntax_valid"], r["indent_match"], r["keyword_match"]]
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    rv = radar_values + [radar_values[0]]; ra = angles + [angles[0]]
    ax.plot(ra, rv, "o-", color=_STYLE.BLUE, lw=2)
    ax.fill(ra, rv, color=_STYLE.BLUE, alpha=0.2)
    ax.set_xticks(angles); ax.set_xticklabels(radar_metrics, fontsize=8, color=_STYLE.TXT)
    ax.set_ylim(0, 1); ax.set_title("Quality radar", fontsize=10, color=_STYLE.BLUE, pad=15)
    ax.spines["polar"].set_color(_STYLE.GRID)

    # 7 — Empty / future expansion
    ax = fig.add_subplot(gs[2, 1]); ax.axis("off")
    ax.text(0.5, 0.5,
            "ℹ Why these metrics?\n\n"
            "• Exact / Prefix-20 — strict correctness\n"
            "• BLEU / chrF / Edit — partial credit\n"
            "• Keyword / Ident — code semantics\n"
            "• Syntax — runnable Python check\n"
            "• Indent — Python-specific structure",
            color=_STYLE.TXT, fontsize=9, family="monospace",
            ha="center", va="center",
            bbox=dict(facecolor=_STYLE.MID, edgecolor=_STYLE.GRID, boxstyle="round"))

    # 8 — Composite score gauge
    ax = fig.add_subplot(gs[2, 2])
    _bar(ax, ["CodeBLEU"], [r["codebleu"]],
         [_STYLE.PURPLE], "Composite score",
         ylim=(0, 1.05), fmt="{:.3f}")

    fig.suptitle(f"{title} — Line Evaluation  ({r['n_samples']} samples)",
                 fontsize=14, color=_STYLE.BLUE, y=1.00)
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] → {out_path}")
