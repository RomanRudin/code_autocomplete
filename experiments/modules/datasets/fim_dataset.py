"""
Fill-in-the-middle datasets for LINE-level infilling.

One sample = (prefix, middle, suffix), prefix + middle + suffix == the original file:
    prefix = the file above the cursor + the current line up to the cursor
    middle = what the model has to write (never crosses a line break)
    suffix = the rest of the current line + the file below
Context is cut by the token budget in `modules/fim.py` (left side of the prefix,
right side of the suffix), exactly as at inference time.

Hole types, mixed with `mode_probs`:
    "eol"    cursor inside the line, middle = everything up to the end of the line
             (the old L task, but the code BELOW the cursor is visible)
    "inner"  middle is a span inside the line; the suffix starts with the rest of the
             line, e.g. brackets/colon the editor has already inserted
    "line"   the whole line (after indentation) is missing between two existing lines
With probability `p_no_suffix` an "eol" sample loses its suffix, so the model also
stays a plain line completer (cursor at the end of the file).

Spans are re-sampled on every access (free augmentation across epochs);
`deterministic=True` fixes them per index (validation).
"""

import random
import string
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from modules.fim import (FIM_HOLE, build_custom_input, build_custom_target,
                         build_t5_input, build_t5_target)

MODES = ("eol", "inner", "line")
_ID_CHARS = set(string.ascii_letters + string.digits + "_")
_TRAILING_CLOSERS = set(")]}:;,'\" ")


def _pick_cut(line: str, lo: int, hi: int, rng, p_mid_word: float) -> int:
    """A cursor position in [lo, hi]: mostly between tokens, sometimes inside an identifier."""
    if rng.random() < p_mid_word:
        return rng.randint(lo, hi)
    cands = [p for p in range(lo, hi + 1)
             if p >= len(line) or not (line[p - 1] in _ID_CHARS and line[p] in _ID_CHARS)]
    return rng.choice(cands) if cands else rng.randint(lo, hi)


def sample_fim_span(lines: Sequence[str], i: int, rng=random,
                    mode_probs: Sequence[float] = (0.5, 0.25, 0.25),
                    p_no_suffix: float = 0.15, p_mid_word: float = 0.3,
                    max_middle_chars: int = 120) -> Optional[Tuple[str, str, str, str]]:
    """Cut a hole in line `i` of a file. Returns (prefix, middle, suffix, mode) or None."""
    line = lines[i]
    indent = len(line) - len(line.lstrip())
    end_of_line = len(line)
    if end_of_line - indent < 2:
        return None

    mode = rng.choices(MODES, weights=mode_probs)[0]
    if mode == "line":
        start, end = indent, end_of_line
    elif mode == "eol":
        start = _pick_cut(line, indent + 1, end_of_line - 1, rng, p_mid_word)
        end = end_of_line
    else:  # inner
        if end_of_line - indent < 3:
            return None
        start = _pick_cut(line, indent + 1, end_of_line - 2, rng, p_mid_word)
        closers = end_of_line
        while closers > start + 1 and line[closers - 1] in _TRAILING_CLOSERS:
            closers -= 1
        if closers < end_of_line and rng.random() < 0.5:
            end = closers                   # rest of line = auto-inserted "):" etc.
        else:
            end = _pick_cut(line, start + 1, end_of_line - 1, rng, 0.0)

    start = max(start, end - max_middle_chars)
    middle = line[start:end]
    if not middle.strip():
        return None

    above = "".join(l + "\n" for l in lines[:i])
    below = "".join("\n" + l for l in lines[i + 1:])
    prefix = above + line[:start]
    suffix = line[end:] + below
    if mode == "eol" and rng.random() < p_no_suffix:
        suffix = ""
    return prefix, middle, suffix, mode


class _FIMBase(Dataset):
    def __init__(self, texts: List[str], min_line_chars: int = 10,
                 line_sample_rate: float = 1.0, deterministic: bool = False,
                 seed: int = 42, **span_kw):
        self.files = [t.splitlines() for t in texts]
        self.index = [(f, i) for f, lines in enumerate(self.files)
                      for i, ln in enumerate(lines) if len(ln.strip()) >= min_line_chars]
        if line_sample_rate < 1.0:
            r = random.Random(seed)
            self.index = [x for x in self.index if r.random() < line_sample_rate]
        self.deterministic = deterministic
        self.seed = seed
        self.span_kw = span_kw
        print(f"[{type(self).__name__}] {len(self.index)} lines from {len(self.files)} files")

    def __len__(self) -> int:
        return len(self.index)

    def triple(self, idx: int) -> Tuple[str, str, str]:
        f, i = self.index[idx]
        rng = random.Random(self.seed * 1_000_003 + idx) if self.deterministic else random
        for _ in range(4):
            s = sample_fim_span(self.files[f], i, rng, **self.span_kw)
            if s is not None:
                return s[:3]
        # fall back to "whole line" (always valid for lines of min_line_chars+)
        s = sample_fim_span(self.files[f], i, rng, **{**self.span_kw, "mode_probs": (0, 0, 1)})
        return s[:3]


class FIMLineDataset(_FIMBase):
    """
    For the custom encoder-decoder (L_rope_model.LineModel + BPECodeTokenizer).
    Sample = (src_ids, tgt_ids), compatible with `collate_line` and the L-bpe-rope
    training loop:  src = <BOS> prefix <FIM_HOLE> suffix <EOS>,  tgt = <BOS> middle <EOS>
    (decoder input tgt[:-1] starts with <BOS>, exactly as in LineModel.generate).
    """
    def __init__(self, texts: List[str], tokenizer, max_prefix: int = 256,
                 max_suffix: int = 128, max_middle: int = 48, **kw):
        if tokenizer.tk.token_to_id(FIM_HOLE) is None:
            raise ValueError("call modules.fim.add_fim_token(tokenizer) before building the dataset")
        self.tok = tokenizer
        self.max_prefix, self.max_suffix, self.max_middle = max_prefix, max_suffix, max_middle
        super().__init__(texts, **kw)

    def __getitem__(self, idx):
        prefix, middle, suffix = self.triple(idx)
        src = build_custom_input(self.tok, prefix, suffix, self.max_prefix, self.max_suffix)
        tgt = build_custom_target(self.tok, middle, self.max_middle)
        return src, tgt


class T5FIMDataset(_FIMBase):
    """
    For CodeT5 / CodeT5+ in their native span-denoising format.
    Sample = (input_ids, attention_mask, labels) tensors, compatible with `collate_t5`:
        input  = <s> prefix <extra_id_0> suffix </s>
        labels = <extra_id_0> middle <extra_id_1> </s>
    """
    def __init__(self, texts: List[str], hf_tokenizer, max_prefix: int = 320,
                 max_suffix: int = 160, max_middle: int = 64, **kw):
        self.tok = hf_tokenizer
        self.max_prefix, self.max_suffix, self.max_middle = max_prefix, max_suffix, max_middle
        super().__init__(texts, **kw)

    def __getitem__(self, idx):
        prefix, middle, suffix = self.triple(idx)
        src = torch.tensor(build_t5_input(self.tok, prefix, suffix, self.max_prefix, self.max_suffix))
        lbl = torch.tensor(build_t5_target(self.tok, middle, self.max_middle))
        return src, torch.ones_like(src), lbl
