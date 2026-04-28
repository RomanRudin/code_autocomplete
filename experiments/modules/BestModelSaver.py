import os
from pathlib import Path
import torch
from torch import nn
from typing import List, Tuple, Optional

class BestModelSaver:
    """Keeps the best N checkpoints by val loss."""
    def __init__(self, ckpt_dir: str, model_name: str, keep: int = 3):
        self.dir   = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name  = model_name
        self.keep  = keep
        self.saved: List[Tuple[float, str]] = []  # (val_loss, path)

    def save(self, model: nn.Module, val_loss: float, epoch: int, extra: dict = None):
        path = str(self.dir / f"{self.name}_ep{epoch:03d}_loss{val_loss:.4f}.pt")
        payload = {
            "model_state": model.state_dict(),
            "val_loss":    val_loss,
            "epoch":       epoch,
            "cfg":         model.cfg,
            # "cfg":         getattr(model, "cfg", None),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        self.saved.append((val_loss, path))
        self.saved.sort(key=lambda x: x[0])
        while len(self.saved) > self.keep:
            _, old = self.saved.pop()
            try:   os.remove(old)
            except FileNotFoundError: pass
            print(f"[Saver] removed old ckpt: {old}")
        print(f"[Saver] saved ckpt: {path}  (val_loss={val_loss:.4f})")

    def best_path(self) -> Optional[str]:
        return self.saved[0][1] if self.saved else None