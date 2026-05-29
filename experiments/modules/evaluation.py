"""
modules/Evaluation.py  Unified evaluation for all four model variants.

Supports:
  • Custom TokenModel (baseline / BPE+RoPE)         `evaluate_token_model`
  • HuggingFace CausalLM    (CodeGen)               `evaluate_token_model`
  • Custom LineModel  (baseline / BPE+RoPE)         `evaluate_line_model`
  • HuggingFace Seq2Seq     (CodeT5)                `evaluate_line_model`

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

# 
#  CONSTANTS  Python token categorisation (for per-type accuracy)
# 

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


# 
#  N-GRAM METRICS (no external deps  no NLTK required)
# 

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
        return 1.0   # vacuous  no keywords to match
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


# 
#  TOKENIZER ABSTRACTION  handles all four tokenizer types
# 

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



# 
#  AUTOCOMPLETE-FOCUSED METRICS
#  These reward "produces valid, idiomatic Python" rather than "reproduces the
#  exact reference continuation". For a basic autocomplete tool this is what
#  actually matters  there are many valid completions, not one.
# 

def brackets_balanced(prefix: str, completion: str) -> float:
    """
    1.0 if prefix+completion has balanced (), [], {}  else 0.0.
    A completion that opens brackets should close them.
    """
    s = prefix + completion
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    in_str = None
    escape = False
    for c in s:
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if in_str:
            if c == in_str:
                in_str = None
            continue
        if c in ('"', "'"):
            in_str = c
            continue
        if c in '([{':
            stack.append(c)
        elif c in pairs:
            if not stack or stack.pop() != pairs[c]:
                return 0.0
    return 1.0 if not stack else 0.0


def starts_like_reference(ref: str, hyp: str, n_tokens: int = 1) -> float:
    """
    1.0 if the first n whitespace-delimited tokens of hyp match ref.
    For autocomplete, often only the first token the user accepts matters.
    """
    r = ref.split()
    h = hyp.split()
    if not r:
        return 1.0 if not h else 0.0
    if len(h) < n_tokens or len(r) < n_tokens:
        n_tokens = min(len(r), len(h), n_tokens)
        if n_tokens == 0:
            return 0.0
    return 1.0 if r[:n_tokens] == h[:n_tokens] else 0.0


def nonempty_completion(hyp: str) -> float:
    """1.0 if the model produced a non-trivial completion (not blank/whitespace)."""
    return 1.0 if hyp.strip() else 0.0


# ── Basic Python construction test set ──────────────────────────────────────
# Each entry: (prefix, [acceptable regex patterns for the completion]).
# A completion is "acceptable" if it matches ANY pattern OR (prefix+completion)
# is valid Python. This measures "handles basic constructions" without
# demanding an exact string match.
import re as _re

BASIC_CONSTRUCTIONS = [
    # for loops
    ("for i in ",            [r"range\s*\(", r"enumerate\s*\(", r"\w+\s*:", r"\w+\s*\)"]),
    ("for item in ",         [r"\w+", r"\w+\s*:"]),
    ("for key, value in ",   [r"\w+\.items\s*\(\)", r"\w+"]),
    # conditionals
    ("if x ",                [r"==", r"!=", r"<", r">", r"is\b", r"in\b", r">=", r"<="]),
    ("if not ",              [r"\w+", r"\w+\s*:"]),
    ("elif ",                [r"\w+", r".*:"]),
    # function / class defs
    ("def __init__(self",    [r"\)", r",\s*\w+", r"\):"]),
    ("def main(",            [r"\)", r"\):", r"\w+"]),
    ("class Foo(",           [r"\w+", r"\):", r"\)"]),
    ("return ",              [r"\w+", r"\[", r"\{", r"\(", r"None", r"True", r"False", r"self\."]),
    # imports
    ("import ",              [r"\w+"]),
    ("from os import ",      [r"\w+"]),
    ("import numpy as ",     [r"np\b", r"\w+"]),
    # data structures
    ("x = [",                [r"\]", r"\w+", r"\d+"]),
    ("d = {",                [r"\}", r"['\"]\w+['\"]\s*:", r"\w+\s*:"]),
    ("result = [x for x in ", [r"\w+"]),
    # assignment / calls
    ("self.value = ",        [r"\w+", r"\d+", r"None", r"\["]),
    ("print(",               [r"\w+", r"['\"]", r"f['\"]", r"\)"]),
    ("with open(",           [r"['\"].*['\"]", r"\w+"]),
    ("while ",               [r"\w+", r"True", r".*:"]),
]


def construction_acceptable(prefix: str, completion: str, patterns: List[str]) -> float:
    """
    1.0 if completion matches any acceptable pattern (at its start, after
    stripping leading whitespace) OR prefix+completion parses as valid Python.
    """
    comp = completion.lstrip()
    for pat in patterns:
        if _re.match(pat, comp):
            return 1.0
    # fall back to syntax check
    if is_valid_python(prefix, completion):
        return 1.0
    return 0.0


@torch.no_grad()
def evaluate_basic_constructions(
    model,
    tokenizer,
    device: torch.device,
    is_hf: bool = False,
    is_hf_seq2seq: bool = False,
    max_new: int = 12,
    construction_set: Optional[List] = None,
) -> Dict[str, Any]:
    """
    Run the model on a fixed set of basic Python construction prompts and
    measure how often it produces an *acceptable* (not identical) completion.

    Works for token models (is_hf True/False) and seq2seq line models
    (is_hf_seq2seq=True). Returns per-construction results + overall score.
    """
    tw = TokenizerWrapper(tokenizer)
    model.eval()
    cset = construction_set or BASIC_CONSTRUCTIONS

    per_item = []
    n_accept = 0
    n_valid  = 0
    n_bracket = 0

    for prefix, patterns in cset:
        # ── generate completion ──────────────────────────────────────────
        if is_hf_seq2seq:
            inp = tw.encode(prefix, add_special=True)
            inp_t = torch.tensor([inp], dtype=torch.long, device=device)
            out = model.generate(inp_t, max_new_tokens=max_new,
                                 num_beams=1, do_sample=False,
                                 pad_token_id=tw.pad_id)
            hyp_ids = out[0].tolist()
            if hyp_ids and hyp_ids[0] in (tw.pad_id, tw.bos_id):
                hyp_ids = hyp_ids[1:]
            hyp = tw.decode([i for i in hyp_ids if i != tw.eos_id])
        elif is_hf:
            inp = tw.encode(prefix, add_special=True)
            inp_t = torch.tensor([inp], dtype=torch.long, device=device)
            out = model.generate(inp_t, max_new_tokens=max_new,
                                 do_sample=False,
                                 pad_token_id=tw.pad_id)
            new_ids = out[0, inp_t.shape[1]:].tolist()
            hyp = tw.decode([i for i in new_ids if i != tw.eos_id])
        else:
            # custom model with .generate(prefix_ids, ...)
            prefix_ids = tw.encode(prefix, add_special=True)
            if hasattr(model, "generate"):
                # token model: generate returns continuation ids
                try:
                    hyp_ids = model.generate(prefix_ids, max_new=max_new,
                                             temperature=0.2, top_k=10,
                                             tokenizer=tokenizer)
                except TypeError:
                    # line model signature
                    hyp_ids = model.generate(prefix_ids, max_new=max_new,
                                             temperature=0.2, top_k=10,
                                             tokenizer=tokenizer)
                hyp = tw.decode([i for i in hyp_ids if i != tw.eos_id])
            else:
                hyp = ""

        # ── score ────────────────────────────────────────────────────────
        acc = construction_acceptable(prefix, hyp, patterns)
        val = float(is_valid_python(prefix, hyp))
        brk = brackets_balanced(prefix, hyp)
        n_accept  += acc
        n_valid   += val
        n_bracket += brk
        per_item.append({
            "prefix":     prefix,
            "completion": hyp[:60],
            "acceptable": acc,
            "valid":      val,
            "brackets":   brk,
        })

    n = len(cset)
    return {
        "n_constructions":     n,
        "acceptable_rate":     n_accept / n,
        "syntax_valid_rate":   n_valid / n,
        "bracket_balance_rate": n_bracket / n,
        "per_item":            per_item,
    }


# 
#  TOKEN MODEL EVALUATION  (next-token; metrics fit autocomplete dropdown UX)
# 

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
    run_constructions: bool = True,
    no_load: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a next-token model.  Works with custom TokenModel and HF CausalLM.
    Leads with autocomplete-relevant metrics (top-1/top-5 = dropdown success),
    plus per-token-type accuracy and a basic-construction probe.
    """
    os.makedirs(out_dir, exist_ok=True)
    tw = TokenizerWrapper(tokenizer)
    model.eval()

    print(f"[Eval Token] tokenising {len(eval_texts)} texts ...")
    all_ids: List[int] = []
    for t in eval_texts:
        all_ids.extend(tw.encode(t))
    if len(all_ids) < ctx + 2:
        raise ValueError(f"Eval data too short: {len(all_ids)} tokens, need > {ctx+2}")

    max_start = len(all_ids) - ctx - 1
    n_samples = min(n_samples, max_start)
    indices   = random.sample(range(max_start), n_samples)
    print(f"[Eval Token] {n_samples} samples, ctx={ctx}")

    # exclude special tokens from accuracy  they're trivially predictable
    SPECIAL_IDS = {tw.pad_id, tw.unk_id, tw.bos_id, tw.eos_id} if hasattr(tw, "unk_id") else {tw.pad_id, tw.bos_id, tw.eos_id}
    try:
        SPECIAL_IDS = {tw.pad_id, tw.bos_id, tw.eos_id}
        # add unk if the tokenizer exposes it
        if tw.kind == "custom" and hasattr(tw.tok, "unk_id"):
            SPECIAL_IDS.add(tw.tok.unk_id)
        elif tw.kind == "hf" and getattr(tw.tok, "unk_token_id", None) is not None:
            SPECIAL_IDS.add(tw.tok.unk_token_id)
    except Exception:
        SPECIAL_IDS = {tw.pad_id}

    total_nll  = 0.0
    total_tok  = 0
    top1 = top5 = 0
    n_scored   = 0
    mrr_sum    = 0.0
    entropies  = []
    latencies  = []

    n_bins   = 10
    bin_acc  = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_cnt  = np.zeros(n_bins, dtype=np.int64)

    type_correct = defaultdict(int)
    type_total   = defaultdict(int)

    PAD = tw.pad_id

    for idx in indices:
        chunk = torch.tensor(all_ids[idx: idx + ctx + 1], dtype=torch.long, device=device).unsqueeze(0)
        x = chunk[:, :-1]
        y = chunk[:,  1:]

        t0 = time.perf_counter()
        if is_hf:
            logits = model(input_ids=x).logits
        else:
            logits = model(x)
        latencies.append((time.perf_counter() - t0) * 1000)

        log_probs = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(log_probs.view(-1, log_probs.size(-1)), y.view(-1),
                         reduction="sum", ignore_index=PAD).item()
        total_nll += nll
        total_tok += (y != PAD).sum().item()

        last_logits = logits[0, -1]
        last_probs  = F.softmax(last_logits, dim=-1)
        true_id     = y[0, -1].item()
        if true_id in SPECIAL_IDS:
            continue
        n_scored += 1

        sorted_ids = torch.argsort(last_logits, descending=True).tolist()
        rank       = sorted_ids.index(true_id) + 1
        top1      += int(rank == 1)
        top5      += int(rank <= 5)
        mrr_sum   += 1.0 / rank

        ent = -(last_probs * (last_probs.clamp(min=1e-12).log())).sum().item()
        entropies.append(ent)

        conf, pred = last_probs.max(dim=-1)
        conf_v  = conf.item()
        correct = int(pred.item() == true_id)
        bin_idx = min(int(conf_v * n_bins), n_bins - 1)
        bin_acc[bin_idx]  += correct
        bin_conf[bin_idx] += conf_v
        bin_cnt[bin_idx]  += 1

        ttype = classify_token(tw.id_to_str(true_id))
        type_total[ttype]   += 1
        type_correct[ttype] += correct

    n_scored = max(1, n_scored)
    perplexity = math.exp(min(total_nll / max(1, total_tok), 20))
    type_accuracy = {
        k: type_correct[k] / type_total[k] if type_total[k] else 0.0
        for k in ["keyword", "identifier", "punctuation", "indent", "number", "other"]
    }
    ece = 0.0
    for i in range(n_bins):
        if bin_cnt[i] > 0:
            acc_i  = bin_acc[i]  / bin_cnt[i]
            conf_i = bin_conf[i] / bin_cnt[i]
            ece   += (bin_cnt[i] / max(1, n_scored)) * abs(acc_i - conf_i)
    cal_curve = {
        "bin_centers": [(i + 0.5) / n_bins for i in range(n_bins)],
        "accuracy":    [bin_acc[i] / bin_cnt[i] if bin_cnt[i] else 0.0 for i in range(n_bins)],
        "confidence":  [bin_conf[i] / bin_cnt[i] if bin_cnt[i] else 0.0 for i in range(n_bins)],
        "count":       bin_cnt.tolist(),
    }

    res = {
        "model_type":    "hf_causal" if is_hf else "custom_token",
        "n_samples":     n_samples,
        "n_scored":      n_scored,
        "ctx":           ctx,
        "perplexity":    perplexity,
        "top1_acc":      top1 / n_scored,
        "top5_acc":      top5 / n_scored,
        "mrr":           mrr_sum / n_scored,
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

    # ── basic-construction probe ─────────────────────────────────────────
    if run_constructions:
        print(f"[Eval Token] running {len(BASIC_CONSTRUCTIONS)} basic-construction prompts ...")
        try:
            cres = evaluate_basic_constructions(model, tokenizer, device, is_hf=is_hf)
            res["construction_acceptable"] = cres["acceptable_rate"]
            res["construction_syntax"]     = cres["syntax_valid_rate"]
            res["construction_brackets"]   = cres["bracket_balance_rate"]
            res["construction_detail"]     = cres["per_item"]
        except Exception as e:
            print(f"[Eval Token] construction probe failed: {e}")
            res["construction_acceptable"] = 0.0
            res["construction_syntax"]     = 0.0
            res["construction_brackets"]   = 0.0

    json_path = os.path.join(out_dir, f"{"NO-LOAD " if no_load else ""}eval_token_results.json")
    with open(json_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[Eval Token] results -> {json_path}")

    _plot_token_dashboard(res, title, os.path.join(out_dir, f"{"NO-LOAD " if no_load else ""}eval_token_dashboard.png"))
    return res


# 
#  LINE MODEL EVALUATION  (autocomplete-focused: char-split, acceptability)
# 

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
    min_line_chars: int = 12,
    run_constructions: bool = True,
    no_load: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a line-completion model with autocomplete-relevant metrics.

    KEY FIX vs the old version: splits each line at a CHARACTER position
    (preserving exact spacing/indentation), so the reference target is the
    real remaining text  not a whitespace-normalised reconstruction.

    Leads with: syntax validity, bracket balance, first-token match, and a
    basic-construction probe. Exact/BLEU kept for reference but de-emphasised.
    """
    os.makedirs(out_dir, exist_ok=True)
    tw = TokenizerWrapper(tokenizer)
    model.eval()

    print(f"[Eval Line] building samples from {len(eval_texts)} texts ...")
    samples: List[Tuple[str, str]] = []
    for text in eval_texts:
        for line in text.splitlines():
            if len(line.strip()) < min_line_chars:
                continue
            # split at a CHARACTER position in the middle  keep exact text
            lo = max(1, int(len(line) * 0.3))
            hi = max(lo + 1, int(len(line) * 0.7))
            cut = (lo + hi) // 2
            prefix = line[:cut]
            suffix = line[cut:]
            if len(suffix.strip()) < 3:
                continue
            samples.append((prefix, suffix))
    if not samples:
        raise ValueError("No suitable lines found in eval_texts")
    random.shuffle(samples)
    samples = samples[:n_samples]
    n_samples = len(samples)
    print(f"[Eval Line] {n_samples} samples")

    exact = prefix_tok = first_tok = 0
    bleu_l, chrf_l, edit_l = [], [], []
    kw_l, id_l, syn_l, brk_l, nonempty_l = [], [], [], [], []
    latencies = []

    for prefix, suffix in samples:
        ref_str = suffix
        ref_ids = tw.encode(ref_str, add_special=False)
        if not ref_ids:
            continue

        prefix_ids = tw.encode(prefix, add_special=(not is_hf_seq2seq))

        t0 = time.perf_counter()
        if is_hf_seq2seq:
            inp = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            out = model.generate(inp, max_new_tokens=max_new, num_beams=1,
                                 do_sample=False, pad_token_id=tw.pad_id)
            hyp_ids = out[0].tolist()
            if hyp_ids and hyp_ids[0] in (tw.pad_id, tw.bos_id):
                hyp_ids = hyp_ids[1:]
        else:
            hyp_ids = model.generate(prefix_ids, max_new=max_new,
                                     temperature=1.0, top_k=1, tokenizer=tokenizer)
        latencies.append((time.perf_counter() - t0) * 1000)

        hyp_ids = [i for i in hyp_ids if i != tw.eos_id]
        hyp_str = tw.decode(hyp_ids)

        # ── metrics ──────────────────────────────────────────────────────
        exact      += int(hyp_str.strip() == ref_str.strip())
        first_tok  += starts_like_reference(ref_str, hyp_str, n_tokens=1)
        prefix_tok += starts_like_reference(ref_str, hyp_str, n_tokens=3)
        bleu_l.append(bleu4(ref_ids, hyp_ids))
        chrf_l.append(chrf(ref_str, hyp_str))
        edit_l.append(edit_similarity(list(ref_str), list(hyp_str)))   # char-level
        kw_l.append(keyword_match(ref_str, hyp_str))
        id_l.append(identifier_match(ref_str, hyp_str))
        syn_l.append(float(is_valid_python(prefix, hyp_str)))
        brk_l.append(brackets_balanced(prefix, hyp_str))
        nonempty_l.append(nonempty_completion(hyp_str))

    n = max(1, len(bleu_l))
    res = {
        "model_type":       "hf_seq2seq" if is_hf_seq2seq else "custom_line",
        "n_samples":        n,
        # ── autocomplete-relevant (lead with these) ──
        "syntax_valid":     float(np.mean(syn_l)),
        "bracket_balance":  float(np.mean(brk_l)),
        "first_token_match": first_tok / n,
        "prefix3_match":    prefix_tok / n,
        "nonempty_rate":    float(np.mean(nonempty_l)),
        "keyword_match":    float(np.mean(kw_l)),
        "identifier_match": float(np.mean(id_l)),
        # ── exact-reproduction (de-emphasised) ──
        "exact_match":      exact / n,
        "bleu4":            float(np.mean(bleu_l)),
        "chrf":             float(np.mean(chrf_l)),
        "edit_similarity":  float(np.mean(edit_l)),
        # ── latency ──
        "latency_mean":     float(np.mean(latencies)),
        "latency_p50":      float(np.percentile(latencies, 50)),
        "latency_p90":      float(np.percentile(latencies, 90)),
        "latency_p99":      float(np.percentile(latencies, 99)),
    }

    if run_constructions:
        print(f"[Eval Line] running {len(BASIC_CONSTRUCTIONS)} basic-construction prompts ...")
        try:
            cres = evaluate_basic_constructions(model, tokenizer, device,
                                                is_hf_seq2seq=is_hf_seq2seq)
            res["construction_acceptable"] = cres["acceptable_rate"]
            res["construction_syntax"]     = cres["syntax_valid_rate"]
            res["construction_brackets"]   = cres["bracket_balance_rate"]
            res["construction_detail"]     = cres["per_item"]
        except Exception as e:
            print(f"[Eval Line] construction probe failed: {e}")
            res["construction_acceptable"] = 0.0
            res["construction_syntax"]     = 0.0
            res["construction_brackets"]   = 0.0

    json_path = os.path.join(out_dir, f"{"NO-LOAD " if no_load else ""}eval_line_results.json")
    with open(json_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[Eval Line] results -> {json_path}")

    _plot_line_dashboard(res, title, os.path.join(out_dir, f"{"NO-LOAD " if no_load else ""}eval_line_dashboard.png"))
    return res


# 
#  COMPARE
# 

def compare_models(results: Dict[str, Dict[str, Any]],
                   out_path: str = "eval_results/comparison.png",
                   kind: str = "token"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    _STYLE.apply()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), facecolor=_STYLE.DARK)
    names = list(results.keys())

    if kind == "token":
        # lead with autocomplete-relevant quality metrics
        primary = ["top1_acc", "top5_acc", "construction_acceptable",
                   "construction_syntax", "construction_brackets"]
    else:
        primary = ["syntax_valid", "bracket_balance", "first_token_match",
                   "construction_acceptable", "construction_syntax"]
    latency = ["latency_p50", "latency_p90", "latency_p99"]

    _grouped_bars(axes[0], primary, results, names, "Autocomplete quality")
    _grouped_bars(axes[1], latency, results, names, "Latency (ms)")

    fig.suptitle(f"Model comparison  {kind}", fontsize=14, color=_STYLE.BLUE)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Compare] -> {out_path}")


# 
#  PLOTTING
# 

class _STYLE:
    DARK   = "#0d1117"; MID = "#161b22"; GRID = "#21262d"
    BLUE   = "#58a6ff"; GREEN = "#3fb950"; ORG = "#ffa657"
    RED    = "#f78166"; PURPLE = "#d2a8ff"; TXT = "#c9d1d9"
    @classmethod
    def apply(cls):
        plt.rcParams.update({
            "axes.facecolor": cls.MID, "axes.edgecolor": cls.GRID,
            "axes.labelcolor": cls.TXT, "xtick.color": cls.TXT,
            "ytick.color": cls.TXT, "text.color": cls.TXT, "grid.color": cls.GRID,
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
    palette  = [_STYLE.BLUE, _STYLE.GREEN, _STYLE.ORG, _STYLE.PURPLE, _STYLE.RED,
                "#79c0ff", "#56d364", "#e3b341"]
    x = np.arange(n_groups)
    for i, name in enumerate(names):
        vals = [results[name].get(m, 0.0) for m in metrics]
        ax.bar(x + i * width, vals, width, label=name,
               color=palette[i % len(palette)], zorder=3)
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=8)
    ax.set_title(title, fontsize=10, color=_STYLE.BLUE, pad=6)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", lw=0.5, zorder=0)


def _plot_token_dashboard(r: Dict, title: str, out_path: str):
    _STYLE.apply()
    fig = plt.figure(figsize=(18, 11), facecolor=_STYLE.DARK)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.30)

    # 0  Autocomplete dropdown success (THE headline metric)
    ax = fig.add_subplot(gs[0, 0])
    _bar(ax, ["Top-1", "Top-5", "MRR"],
         [r["top1_acc"], r["top5_acc"], r["mrr"]],
         [_STYLE.BLUE, _STYLE.GREEN, _STYLE.ORG],
         "Dropdown success (higher=better)")

    # 1  Basic construction handling (your stated goal)
    ax = fig.add_subplot(gs[0, 1])
    _bar(ax, ["Acceptable", "Syntax\nvalid", "Brackets\nbalanced"],
         [r.get("construction_acceptable", 0), r.get("construction_syntax", 0),
          r.get("construction_brackets", 0)],
         [_STYLE.GREEN, _STYLE.BLUE, _STYLE.ORG],
         "Basic constructions")

    # 2  Per-type accuracy
    ax = fig.add_subplot(gs[0, 2])
    types = list(r["type_accuracy"].keys())
    vals  = [r["type_accuracy"][t] for t in types]
    cols  = [_STYLE.BLUE, _STYLE.ORG, _STYLE.GREEN, _STYLE.PURPLE, _STYLE.RED, _STYLE.TXT][:len(types)]
    _bar(ax, types, vals, cols, "Accuracy by token type")
    ax.tick_params(axis="x", labelrotation=20)

    # 3  Reliability diagram
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

    # 4  Calibration / entropy
    ax = fig.add_subplot(gs[1, 1])
    _bar(ax, ["ECE down", "Entropy"], [r["ece"], r["entropy"]],
         [_STYLE.RED, _STYLE.PURPLE], "Confidence calibration",
         ylim=(0, max(1.0, r["entropy"] * 1.2)))

    # 5  Latency
    ax = fig.add_subplot(gs[1, 2])
    _bar(ax, ["mean", "p50", "p90", "p99"],
         [r["latency_mean"], r["latency_p50"], r["latency_p90"], r["latency_p99"]],
         [_STYLE.BLUE, _STYLE.GREEN, _STYLE.ORG, _STYLE.RED],
         "Inference latency (ms)",
         ylim=(0, max(r["latency_p99"], 1) * 1.15), fmt="{:.1f}")

    # 6  Perplexity
    ax = fig.add_subplot(gs[2, 0])
    _bar(ax, ["Perplexity"], [r["perplexity"]], [_STYLE.ORG],
         "Perplexity (lower=better)",
         ylim=(0, max(r["perplexity"] * 1.25, 10)), fmt="{:.1f}")

    # 7  token-type sample counts
    ax = fig.add_subplot(gs[2, 1])
    types  = list(r["type_counts"].keys())
    counts = [r["type_counts"][t] for t in types]
    cols   = [_STYLE.BLUE, _STYLE.ORG, _STYLE.GREEN, _STYLE.PURPLE, _STYLE.RED, _STYLE.TXT][:len(types)]
    _bar(ax, types, counts, cols, "Eval-set token-type counts",
         ylim=(0, max(counts) * 1.15 if counts else 1), fmt="{:d}")
    ax.tick_params(axis="x", labelrotation=20)

    # 8  quality radar
    ax = fig.add_subplot(gs[2, 2], polar=True)
    ax.set_facecolor(_STYLE.MID)
    radar = ["Top-1", "Top-5", "Construct\naccept", "Syntax", "1-ECE"]
    vals  = [r["top1_acc"], r["top5_acc"], r.get("construction_acceptable", 0),
             r.get("construction_syntax", 0), 1.0 - r["ece"]]
    ang = np.linspace(0, 2*np.pi, len(radar), endpoint=False).tolist()
    rv = vals + [vals[0]]; ra = ang + [ang[0]]
    ax.plot(ra, rv, "o-", color=_STYLE.BLUE, lw=2)
    ax.fill(ra, rv, color=_STYLE.BLUE, alpha=0.2)
    ax.set_xticks(ang); ax.set_xticklabels(radar, fontsize=8, color=_STYLE.TXT)
    ax.set_ylim(0, 1); ax.set_title("Quality radar", fontsize=10, color=_STYLE.BLUE, pad=15)
    ax.spines["polar"].set_color(_STYLE.GRID)

    fig.suptitle(f"{title}  Token Evaluation  ({r['n_samples']} samples, {r.get('n_scored', '?')} scored)",
                 fontsize=14, color=_STYLE.BLUE, y=1.00)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] -> {out_path}")


def _plot_line_dashboard(r: Dict, title: str, out_path: str):
    _STYLE.apply()
    fig = plt.figure(figsize=(18, 11), facecolor=_STYLE.DARK)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.30)

    # 0  Autocomplete quality (THE headline panel)
    ax = fig.add_subplot(gs[0, 0])
    _bar(ax, ["Syntax\nvalid", "Brackets\nbalanced", "Non-empty"],
         [r["syntax_valid"], r["bracket_balance"], r["nonempty_rate"]],
         [_STYLE.BLUE, _STYLE.GREEN, _STYLE.ORG],
         "Produces valid Python")

    # 1  Basic construction handling
    ax = fig.add_subplot(gs[0, 1])
    _bar(ax, ["Acceptable", "Syntax\nvalid", "Brackets"],
         [r.get("construction_acceptable", 0), r.get("construction_syntax", 0),
          r.get("construction_brackets", 0)],
         [_STYLE.GREEN, _STYLE.BLUE, _STYLE.ORG],
         "Basic constructions")

    # 2  First-token / prefix match (autocomplete acceptance)
    ax = fig.add_subplot(gs[0, 2])
    _bar(ax, ["First\ntoken", "First-3\ntokens", "Keyword", "Ident"],
         [r["first_token_match"], r["prefix3_match"],
          r["keyword_match"], r["identifier_match"]],
         [_STYLE.BLUE, _STYLE.GREEN, _STYLE.ORG, _STYLE.PURPLE],
         "Matches what user expects")

    # 3  Exact-reproduction metrics (de-emphasised, kept for reference)
    ax = fig.add_subplot(gs[1, 0])
    _bar(ax, ["Exact", "BLEU-4", "chrF", "Edit\nsim"],
         [r["exact_match"], r["bleu4"], r["chrf"], r["edit_similarity"]],
         [_STYLE.RED, _STYLE.RED, _STYLE.ORG, _STYLE.ORG],
         "Exact-reproduction (NOT the goal)")

    # 4  Latency
    ax = fig.add_subplot(gs[1, 1])
    _bar(ax, ["mean", "p50", "p90", "p99"],
         [r["latency_mean"], r["latency_p50"], r["latency_p90"], r["latency_p99"]],
         [_STYLE.BLUE, _STYLE.GREEN, _STYLE.ORG, _STYLE.RED],
         "Inference latency (ms)",
         ylim=(0, max(r["latency_p99"], 1) * 1.15), fmt="{:.1f}")

    # 5  Summary text
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    txt = (
        f"Eval samples : {r['n_samples']:,}\n"
        f"Model type   : {r['model_type']}\n\n"
        f"Syntax valid : {r['syntax_valid']:.1%}\n"
        f"Brackets ok  : {r['bracket_balance']:.1%}\n"
        f"Construct ok : {r.get('construction_acceptable', 0):.1%}\n"
        f"First-tok hit: {r['first_token_match']:.1%}\n"
    )
    ax.text(0.05, 0.5, txt, color=_STYLE.TXT, fontsize=11, family="monospace", va="center")
    ax.set_title("Run summary", fontsize=10, color=_STYLE.BLUE, pad=6)

    # 6  quality radar
    ax = fig.add_subplot(gs[2, 0], polar=True)
    ax.set_facecolor(_STYLE.MID)
    radar = ["Syntax", "Brackets", "First\ntok", "Construct\naccept", "Keyword"]
    vals  = [r["syntax_valid"], r["bracket_balance"], r["first_token_match"],
             r.get("construction_acceptable", 0), r["keyword_match"]]
    ang = np.linspace(0, 2*np.pi, len(radar), endpoint=False).tolist()
    rv = vals + [vals[0]]; ra = ang + [ang[0]]
    ax.plot(ra, rv, "o-", color=_STYLE.BLUE, lw=2)
    ax.fill(ra, rv, color=_STYLE.BLUE, alpha=0.2)
    ax.set_xticks(ang); ax.set_xticklabels(radar, fontsize=8, color=_STYLE.TXT)
    ax.set_ylim(0, 1); ax.set_title("Quality radar", fontsize=10, color=_STYLE.BLUE, pad=15)
    ax.spines["polar"].set_color(_STYLE.GRID)

    # 7  why-these-metrics info
    ax = fig.add_subplot(gs[2, 1]); ax.axis("off")
    ax.text(0.5, 0.5,
            "Metrics for BASIC AUTOCOMPLETE\n\n"
            "LEAD (top row):\n"
            "  Syntax valid  - parseable Python?\n"
            "  Brackets      - closes what it opens?\n"
            "  Construct ok  - handles for/if/def?\n"
            "  First-token   - right first suggestion?\n\n"
            "DE-EMPHASISED (row 2 left):\n"
            "  Exact/BLEU    - reproduces EXACT ref\n"
            "  (many valid completions exist, so\n"
            "   low scores here are expected & OK)",
            color=_STYLE.TXT, fontsize=8, family="monospace",
            ha="center", va="center",
            bbox=dict(facecolor=_STYLE.MID, edgecolor=_STYLE.GRID, boxstyle="round"))

    # 8  composite autocomplete score
    ax = fig.add_subplot(gs[2, 2])
    composite = (
        0.35 * r["syntax_valid"] +
        0.25 * r["bracket_balance"] +
        0.20 * r.get("construction_acceptable", 0) +
        0.20 * r["first_token_match"]
    )
    _bar(ax, ["Autocomplete\nscore"], [composite], [_STYLE.PURPLE],
         "Composite (weighted)", ylim=(0, 1.05))

    fig.suptitle(f"{title}  Line Evaluation  ({r['n_samples']} samples)",
                 fontsize=14, color=_STYLE.BLUE, y=1.00)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] -> {out_path}")
