from typing import List, Tuple
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random
import torch

class T5LineDataset(Dataset):
    """
    Multi-line context for line completion.

    For each line that's long enough to be a meaningful target, build a prefix
    consisting of:
        previous N lines      (full prior context)
        current line so far   (partial — cut at 30-70 % of words)
    The suffix is the remainder of the current line.

    This is the key change from the single-line version: the encoder now sees
    surrounding code (imports, function signature, variable definitions),
    which lets it reference identifiers and patterns from above instead of
    completing the current line in a vacuum.
    """
    def __init__(self, texts: List[str], hf_tokenizer: AutoTokenizer,
                 max_prefix: int = 384,
                 max_suffix: int = 64,
                 context_lines: int = 3,
                 min_line_chars: int = 10):
        self.tok            = hf_tokenizer
        self.max_prefix     = max_prefix
        self.max_suffix     = max_suffix
        self.context_lines  = context_lines
        self.samples: List[Tuple[str, str]] = []

        for text in texts:
            lines = text.splitlines()
            for i, raw_line in enumerate(lines):
                cur_line = raw_line.rstrip()
                if len(cur_line.strip()) < min_line_chars:
                    continue
                words = cur_line.split()
                if len(words) < 3:
                    continue

                cut = random.randint( # 30–70 % cut inside the cur line
                    max(1, int(len(words) * 0.3)),
                    max(2, int(len(words) * 0.7)),
                )
                current_prefix = " ".join(words[:cut])
                suffix         = " ".join(words[cut:])

                # gather up to context_lines previous non-empty lines
                ctx_start = max(0, i - self.context_lines)
                ctx_lines = [
                    ln.rstrip() for ln in lines[ctx_start:i]
                    if ln.strip()       # skip blank lines but keep indentation on others
                ]
                ctx_block = "\n".join(ctx_lines)

                # final prefix = previous lines + newline + current partial line
                if ctx_block:
                    prefix = ctx_block + "\n" + current_prefix
                else:
                    prefix = current_prefix

                self.samples.append((prefix, suffix))

        print(f"[T5LineDataset] {len(self.samples)} samples "
              f"(context_lines={self.context_lines}, max_prefix={self.max_prefix})")

    def __len__(self):
        return len(self.samples)

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