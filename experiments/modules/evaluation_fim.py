"""
modules/evaluation_fim.py  Evaluation of fill-in-the-middle LINE models.

Unlike prefix-only line completion, the hole is framed by code on both sides,
which makes two things measurable that were not before:
  • exact match / edit similarity become informative — the suffix pins down the answer;
  • syntax validity is checked on the WHOLE FILE (prefix + completion + suffix),
    only files that parse on their own are used, so the reference always scores 1.0.
    (Parsing a single cut line, as in evaluate_line_model, fails for any `for ...:`.)

Usage from a notebook:

    from modules.evaluation_fim import evaluate_fim_model

    res = evaluate_fim_model(
        model=model, tokenizer=hf_tok, eval_texts=test_texts, device=device,
        kind="t5",                        # "custom" | "t5" | "causal" (+ family="santacoder")
        n_samples=500,
        out_dir="results/line_model_FIM_T5_small", title="FIM Line - CodeT5-small",
    )
    # ablation: the same model without the code below the cursor
    res_nosuf = evaluate_fim_model(..., use_suffix=False, tag="NO-SUFFIX ")
"""

import ast, json, os, random, time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from modules.fim import fim_generate
from modules.datasets.fim_dataset import MODES, sample_fim_span
from modules.evaluation import (_STYLE, _bar, _grouped_bars, chrf, edit_similarity,
                                identifier_match, starts_like_reference)


def _parses(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError):
        return False


def build_fim_eval_set(eval_texts: List[str], n_samples: int = 500, seed: int = 42,
                       mode_probs=(0.5, 0.25, 0.25), min_line_chars: int = 12,
                       max_middle_chars: int = 120) -> List[Dict[str, str]]:
    """Fixed, reproducible set of holes cut from files that parse as Python."""
    rng = random.Random(seed)
    files = [t.splitlines() for t in eval_texts if _parses(t)]
    print(f"[Eval FIM] {len(files)}/{len(eval_texts)} files parse as Python")
    cands = [(f, i) for f, lines in enumerate(files)
             for i, ln in enumerate(lines) if len(ln.strip()) >= min_line_chars]
    rng.shuffle(cands)
    samples = []
    for f, i in cands:
        s = sample_fim_span(files[f], i, rng, mode_probs=mode_probs, p_no_suffix=0.0,
                            max_middle_chars=max_middle_chars)
        if s is None:
            continue
        prefix, middle, suffix, mode = s
        samples.append({"prefix": prefix, "middle": middle, "suffix": suffix, "mode": mode})
        if len(samples) >= n_samples:
            break
    if not samples:
        raise ValueError("No suitable lines found in eval_texts")
    return samples


def evaluate_fim_model(
    model,
    tokenizer,
    eval_texts: List[str],
    device: torch.device,
    kind: str = "t5",
    family: Optional[str] = None,
    n_samples: int = 500,
    max_new: int = 48,
    use_suffix: bool = True,
    out_dir: str = "eval_results",
    title: str = "FIM Line Model",
    tag: str = "",
    seed: int = 42,
    **gen_kw,
) -> Dict[str, Any]:
    """
    Greedy FIM decoding on a fixed hole set. `use_suffix=False` hides the code after
    the cursor (ablation: how much does the suffix help?). Files are written as
    `{tag}eval_fim_results.json` / `{tag}eval_fim_dashboard.png` (tag e.g. "NO-LOAD ").
    """
    os.makedirs(out_dir, exist_ok=True)
    samples = build_fim_eval_set(eval_texts, n_samples, seed)
    print(f"[Eval FIM] {len(samples)} samples  kind={kind}  use_suffix={use_suffix}")

    rows = []
    latencies = []
    for s in samples:
        prefix, ref, suffix = s["prefix"], s["middle"], s["suffix"]
        t0 = time.perf_counter()
        hyp, raw = fim_generate(model, tokenizer, prefix, suffix if use_suffix else "",
                                device, kind=kind, family=family, max_new=max_new,
                                return_raw=True, **gen_kw)
        latencies.append((time.perf_counter() - t0) * 1000)
        raw_line = raw.split("\n", 1)[0]
        rows.append({
            "mode":       s["mode"],
            "exact":      float(hyp.strip() == ref.strip()),
            "edit_sim":   edit_similarity(list(ref), list(hyp)),
            "chrf":       chrf(ref, hyp),
            "first_tok":  starts_like_reference(ref, hyp, n_tokens=1),
            "prefix3":    starts_like_reference(ref, hyp, n_tokens=3),
            "ident":      identifier_match(ref, hyp),
            "syntax":     float(_parses(prefix + hyp + suffix)),
            "nonempty":   float(bool(hyp.strip())),
            "trimmed":    float(hyp.rstrip() != raw_line.rstrip()),
        })

    def mean(key, rs=rows):
        return float(np.mean([r[key] for r in rs])) if rs else 0.0

    res: Dict[str, Any] = {
        "model_type":        kind if kind != "causal" else f"causal/{family}",
        "use_suffix":        use_suffix,
        "n_samples":         len(rows),
        # ── hole filled correctly ──
        "exact_match":       mean("exact"),
        "edit_similarity":   mean("edit_sim"),
        "chrf":              mean("chrf"),
        "first_token_match": mean("first_tok"),
        "prefix3_match":     mean("prefix3"),
        "identifier_match":  mean("ident"),
        # ── file stays valid ──
        "syntax_valid":      mean("syntax"),
        "nonempty_rate":     mean("nonempty"),
        "suffix_overlap_trimmed": mean("trimmed"),
        # ── latency ──
        "latency_mean":      float(np.mean(latencies)),
        "latency_p50":       float(np.percentile(latencies, 50)),
        "latency_p90":       float(np.percentile(latencies, 90)),
        "latency_p99":       float(np.percentile(latencies, 99)),
        "per_mode": {
            m: {"n": sum(r["mode"] == m for r in rows),
                "exact_match": mean("exact", [r for r in rows if r["mode"] == m]),
                "edit_similarity": mean("edit_sim", [r for r in rows if r["mode"] == m]),
                "syntax_valid": mean("syntax", [r for r in rows if r["mode"] == m])}
            for m in MODES
        },
    }

    json_path = os.path.join(out_dir, f"{tag}eval_fim_results.json")
    with open(json_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[Eval FIM] results -> {json_path}")
    _plot_fim_dashboard(res, f"{tag}{title}", os.path.join(out_dir, f"{tag}eval_fim_dashboard.png"))
    return res


def _plot_fim_dashboard(r: Dict, title: str, out_path: str):
    _STYLE.apply()
    fig = plt.figure(figsize=(18, 8), facecolor=_STYLE.DARK)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    _bar(ax, ["Exact", "Edit\nsim", "chrF", "First\ntoken"],
         [r["exact_match"], r["edit_similarity"], r["chrf"], r["first_token_match"]],
         [_STYLE.GREEN, _STYLE.BLUE, _STYLE.ORG, _STYLE.PURPLE],
         "Hole filled like the reference")

    ax = fig.add_subplot(gs[0, 1])
    _bar(ax, ["Syntax valid\n(whole file)", "Non-empty", "Suffix overlap\ntrimmed"],
         [r["syntax_valid"], r["nonempty_rate"], r["suffix_overlap_trimmed"]],
         [_STYLE.BLUE, _STYLE.GREEN, _STYLE.RED],
         "File stays valid Python")

    ax = fig.add_subplot(gs[0, 2])
    modes = [m for m in MODES if r["per_mode"][m]["n"]]
    _grouped_bars(ax, ["exact_match", "edit_similarity", "syntax_valid"],
                  {m: r["per_mode"][m] for m in modes},
                  modes, "Per hole type")
    ax.set_ylim(0, 1)

    ax = fig.add_subplot(gs[1, 0])
    _bar(ax, ["mean", "p50", "p90", "p99"],
         [r["latency_mean"], r["latency_p50"], r["latency_p90"], r["latency_p99"]],
         [_STYLE.BLUE, _STYLE.GREEN, _STYLE.ORG, _STYLE.RED],
         "Inference latency (ms)",
         ylim=(0, max(r["latency_p99"], 1) * 1.15), fmt="{:.1f}")

    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    txt = (
        f"Eval samples : {r['n_samples']:,}\n"
        f"Model type   : {r['model_type']}\n"
        f"Uses suffix  : {r['use_suffix']}\n\n"
        f"Exact match  : {r['exact_match']:.1%}\n"
        f"Edit sim     : {r['edit_similarity']:.3f}\n"
        f"Syntax valid : {r['syntax_valid']:.1%}\n"
        f"Latency p50  : {r['latency_p50']:.0f} ms\n"
    )
    ax.text(0.05, 0.5, txt, color=_STYLE.TXT, fontsize=11, family="monospace", va="center")
    ax.set_title("Run summary", fontsize=10, color=_STYLE.BLUE, pad=6)

    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.text(0.5, 0.5,
            "Hole types\n\n"
            "  eol   - cursor in a line, fill to its end\n"
            "  inner - span inside a line, rest of the\n"
            "          line (\"):\" etc.) is already there\n"
            "  line  - a whole line between two lines\n\n"
            "Syntax is checked on prefix+hole+suffix,\n"
            "i.e. on the whole (parseable) file",
            color=_STYLE.TXT, fontsize=8, family="monospace", ha="center", va="center",
            bbox=dict(facecolor=_STYLE.MID, edgecolor=_STYLE.GRID, boxstyle="round"))

    fig.suptitle(f"{title}  FIM Evaluation  ({r['n_samples']} samples)",
                 fontsize=14, color=_STYLE.BLUE, y=1.00)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] -> {out_path}")
