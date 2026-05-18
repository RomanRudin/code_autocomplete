import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Tuple


class T5LineDataset(Dataset):
    """
    Each sample: tokenise the prefix with AutoTokenizer → input_ids
                 tokenise the suffix                        → labels
    Padding and label-masking are handled in the collator below.
    """
    def __init__(self, texts: List[str], hf_tokenizer: AutoTokenizer,
                 max_prefix: int = 96, max_suffix: int = 64):
        self.tok = hf_tokenizer
        self.max_prefix = max_prefix
        self.max_suffix = max_suffix
        self.samples: List[Tuple[str, str]] = []

        for text in texts:
            for line in text.splitlines():
                line = line.rstrip()
                if len(line.strip()) < 10:
                    continue
                # split at 30–70 % of the line (your "root cause 2" fix)
                words = line.split()
                if len(words) < 3:
                    continue
                cut = random.randint(
                    max(1, int(len(words) * 0.3)),
                    max(2, int(len(words) * 0.7)),
                )
                prefix = " ".join(words[:cut])
                suffix = " ".join(words[cut:])
                self.samples.append((prefix, suffix))

        print(f"[T5LineDataset] {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        prefix, suffix = self.samples[i]
        enc = self.tok(
            prefix,
            max_length=self.max_prefix,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        dec = self.tok(
            suffix,
            max_length=self.max_suffix,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            dec["input_ids"].squeeze(0),
        )



def collate_t5(batch, pad_id: int):
    """Pad input_ids, attention_mask, and labels in a single pass."""
    src_ids, src_masks, lbl_ids = zip(*batch)

    max_src = max(t.size(0) for t in src_ids)
    max_lbl = max(t.size(0) for t in lbl_ids)

    B = len(batch)
    SRC  = torch.full((B, max_src), pad_id, dtype=torch.long)
    MASK = torch.zeros((B, max_src), dtype=torch.long)
    LBL  = torch.full((B, max_lbl), -100,   dtype=torch.long)   # -100 = ignored by T5 loss

    for i, (s, m, l) in enumerate(zip(src_ids, src_masks, lbl_ids)):
        SRC[i,  :s.size(0)] = s
        MASK[i, :m.size(0)] = m
        LBL[i,  :l.size(0)] = l

    return SRC, MASK, LBL