import glob, os
from pathlib import Path
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