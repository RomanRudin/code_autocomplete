import os, glob, torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import List

def load_files(data_dir: str, max_files: int = 0) -> List[str]:
    """Load .py / .txt files from a directory tree."""
    patterns = ["**/*.py", "**/*.txt"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pat), recursive=True))
    if max_files:
        files = files[:max_files]
    texts = []
    for fp in files:
        try:
            texts.append(Path(fp).read_text(errors="replace"))
        except Exception:
            pass
    print(f"[Data] loaded {len(texts)} files from {data_dir}")
    return texts



class TokenDataset(Dataset):
    """
    Sliding-window dataset for next-token prediction.
    Target at each position is the next token id.
    """
    def __init__(self, ids: List[int], ctx: int = 128):
        self.ctx = ctx
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.ctx - 1)

    def __getitem__(self, i):
        x = self.data[i: i + self.ctx]
        y = self.data[i + 1: i + self.ctx + 1]
        return x, y
    

