from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")           # headless — saves PNG files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


@dataclass
class MetricLog:
    train_loss:  List[float] = field(default_factory=list)
    val_loss:    List[float] = field(default_factory=list)
    train_ppl:   List[float] = field(default_factory=list)
    val_ppl:     List[float] = field(default_factory=list)
    lr:          List[float] = field(default_factory=list)
    token_acc:   List[float] = field(default_factory=list)   # top-1 accuracy
    grad_norm:   List[float] = field(default_factory=list)

    def append(self, **kw):
        for k, v in kw.items():
            getattr(self, k).append(v)


def plot_metrics(log: MetricLog, title: str, save_path: str):
    """
    Rich 2×3 dashboard saved to PNG.
    """
    epochs = list(range(1, len(log.train_loss) + 1))
    fig = plt.figure(figsize=(18, 10), facecolor="#0d1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ACCENT = "#58a6ff"
    WARN   = "#f78166"
    OK     = "#3fb950"
    GRID   = "#21262d"
    TXT    = "#c9d1d9"

    plt.rcParams.update({
        "axes.facecolor":   "#161b22",
        "axes.edgecolor":   GRID,
        "axes.labelcolor":  TXT,
        "xtick.color":      TXT,
        "ytick.color":      TXT,
        "text.color":       TXT,
        "grid.color":       GRID,
        "grid.linewidth":   0.6,
    })

    def _ax(pos, ylabel, title_s):
        ax = fig.add_subplot(pos)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title_s, fontsize=10, color=ACCENT, pad=6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)
        return ax

    # 1 — Loss
    ax = _ax(gs[0, 0], "Loss", "Train / Val Loss")
    ax.plot(epochs, log.train_loss, color=ACCENT, lw=1.8, label="train")
    if log.val_loss:
        ax.plot(epochs, log.val_loss, color=WARN, lw=1.8, linestyle="--", label="val")
    ax.legend(fontsize=8)

    # 2 — Perplexity
    ax = _ax(gs[0, 1], "Perplexity", "Train / Val Perplexity")
    ax.plot(epochs, log.train_ppl, color=ACCENT, lw=1.8, label="train")
    if log.val_ppl:
        ax.plot(epochs, log.val_ppl, color=WARN, lw=1.8, linestyle="--", label="val")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    # 3 — Token top-1 accuracy
    ax = _ax(gs[0, 2], "Accuracy", "Top-1 Token Accuracy")
    ax.plot(epochs, log.token_acc, color=OK, lw=1.8)
    ax.set_ylim(0, 1)

    # 4 — LR schedule
    ax = _ax(gs[1, 0], "LR", "Learning Rate")
    ax.plot(epochs, log.lr, color="#d2a8ff", lw=1.5)
    ax.set_yscale("log")

    # 5 — Gradient norm
    ax = _ax(gs[1, 1], "Grad Norm", "Gradient Norm")
    ax.plot(epochs, log.grad_norm, color="#ffa657", lw=1.5)

    # 6 — Train vs Val gap (over-fit indicator)
    ax = _ax(gs[1, 2], "Δ Loss (train-val)", "Generalisation Gap")
    if log.val_loss:
        gap = [v - t for t, v in zip(log.train_loss, log.val_loss)]
        ax.fill_between(epochs, 0, gap,
                        where=[g > 0 for g in gap], color=WARN, alpha=0.35, label="overfit")
        ax.fill_between(epochs, 0, gap,
                        where=[g <= 0 for g in gap], color=OK, alpha=0.35, label="underfit")
        ax.plot(epochs, gap, color=TXT, lw=1.0)
        ax.axhline(0, color=GRID, lw=1)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=14, color=ACCENT, y=1.01)
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] saved → {save_path}")