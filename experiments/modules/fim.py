"""
modules/fim.py  Fill-in-the-middle (FIM) formats and inference.

Shared by the FIM datasets, `evaluation_fim.py` and `hand_test_fim_repl`, so the
model sees exactly the same input layout in training, evaluation and hand testing.

Supported model kinds:
  • "custom"  L_rope_model.LineModel + BPECodeTokenizer with an extra <FIM_HOLE> token
              encoder: <BOS> prefix <FIM_HOLE> suffix <EOS>    decoder: <BOS> middle <EOS>
  • "t5"      CodeT5 / CodeT5+ — their span-denoising pretraining already is FIM
              encoder: <s> prefix <extra_id_0> suffix </s>     decoder: <extra_id_0> middle <extra_id_1> </s>
  • "causal"  FIM-pretrained decoder-only LMs (SantaCoder, StarCoder2, Qwen2.5-Coder, DeepSeek-Coder)
              PSM order: <fim_prefix> prefix <fim_suffix> suffix <fim_middle> -> middle <eos>
"""

from typing import Dict, List, Optional, Tuple
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

FIM_HOLE = "<FIM_HOLE>"

# (prefix, suffix, middle) sentinel tokens of FIM-pretrained causal LMs
CAUSAL_FIM_TOKENS: Dict[str, Tuple[str, str, str]] = {
    "santacoder": ("<fim-prefix>", "<fim-suffix>", "<fim-middle>"),        # bigcode/gpt_bigcode-santacoder
    "starcoder":  ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),        # bigcode/starcoder2-*
    "qwen":       ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"),  # Qwen/Qwen2.5-Coder-*
    "deepseek":   ("<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>"),     # deepseek-ai/deepseek-coder-*-base
}

# rough upper bound of characters per token, used to avoid tokenising a whole file
# when only the last/first few hundred tokens are kept
_CHARS_PER_TOKEN = 8

DEFAULT_BUDGETS = {                 # (max_prefix, max_suffix) in tokens
    "custom": (256, 128),
    "t5":     (320, 160),           # CodeT5 was pretrained with 512-token inputs
    "causal": (768, 256),
}


# ── custom BPE encoder-decoder ──────────────────────────────────────────────

def add_fim_token(tokenizer) -> int:
    """Add <FIM_HOLE> to a BPECodeTokenizer (no-op if already present), return its id."""
    if tokenizer.tk.token_to_id(FIM_HOLE) is None:
        tokenizer.tk.add_special_tokens([FIM_HOLE])
    return tokenizer.tk.token_to_id(FIM_HOLE)


def _bpe_ids(tokenizer, text: str) -> List[int]:
    return tokenizer.tk.encode(text).ids if text else []


def build_custom_input(tokenizer, prefix: str, suffix: str,
                       max_prefix: int = 256, max_suffix: int = 128) -> List[int]:
    hole = tokenizer.tk.token_to_id(FIM_HOLE)
    if hole is None:
        raise ValueError("tokenizer has no <FIM_HOLE> token — call add_fim_token(tokenizer) first")
    pre = _bpe_ids(tokenizer, prefix[-max_prefix * _CHARS_PER_TOKEN:])[-max_prefix:]
    suf = _bpe_ids(tokenizer, suffix[:max_suffix * _CHARS_PER_TOKEN])[:max_suffix]
    return [tokenizer.bos_id] + pre + [hole] + suf + [tokenizer.eos_id]


def build_custom_target(tokenizer, middle: str, max_middle: int = 48) -> List[int]:
    mid = _bpe_ids(tokenizer, middle)
    if len(mid) > max_middle:
        # no <EOS> after a truncated middle, otherwise the model learns to stop too early
        return [tokenizer.bos_id] + mid[:max_middle]
    return [tokenizer.bos_id] + mid + [tokenizer.eos_id]


# ── CodeT5 (sentinel format) ────────────────────────────────────────────────

def t5_sentinels(hf_tok) -> Tuple[int, int]:
    return (hf_tok.convert_tokens_to_ids("<extra_id_0>"),
            hf_tok.convert_tokens_to_ids("<extra_id_1>"))


def _hf_ids(hf_tok, text: str) -> List[int]:
    return hf_tok.encode(text, add_special_tokens=False) if text else []


def build_t5_input(hf_tok, prefix: str, suffix: str,
                   max_prefix: int = 320, max_suffix: int = 160) -> List[int]:
    s0, _ = t5_sentinels(hf_tok)
    pre = _hf_ids(hf_tok, prefix[-max_prefix * _CHARS_PER_TOKEN:])[-max_prefix:]
    suf = _hf_ids(hf_tok, suffix[:max_suffix * _CHARS_PER_TOKEN])[:max_suffix]
    return hf_tok.build_inputs_with_special_tokens(pre + [s0] + suf)


def build_t5_target(hf_tok, middle: str, max_middle: int = 64) -> List[int]:
    s0, s1 = t5_sentinels(hf_tok)
    mid = _hf_ids(hf_tok, middle)
    if len(mid) > max_middle:
        return [s0] + mid[:max_middle]
    return [s0] + mid + [s1, hf_tok.eos_token_id]


def extract_t5_middle(hf_tok, out_ids: List[int]) -> str:
    """Take the text between <extra_id_0> and the next sentinel / </s>."""
    s0, s1 = t5_sentinels(hf_tok)
    ids = list(out_ids)
    if s0 in ids[:4]:                       # after decoder_start (<pad>) and maybe <s>
        ids = ids[ids.index(s0) + 1:]
    stop = (set(hf_tok.all_special_ids) | {s0, s1}) - {hf_tok.unk_token_id}
    mid = []
    for i in ids:
        if i in stop:
            if mid:
                break
            continue                        # leading <pad>/<s>
        mid.append(i)
    return hf_tok.decode(mid, skip_special_tokens=True, clean_up_tokenization_spaces=False)


# ── causal FIM LMs (PSM) ────────────────────────────────────────────────────

def build_causal_input(hf_tok, prefix: str, suffix: str, family: str,
                       max_prefix: int = 768, max_suffix: int = 256) -> List[int]:
    fp, fs, fm = (hf_tok.convert_tokens_to_ids(t) for t in CAUSAL_FIM_TOKENS[family])
    pre = _hf_ids(hf_tok, prefix[-max_prefix * _CHARS_PER_TOKEN:])[-max_prefix:]
    suf = _hf_ids(hf_tok, suffix[:max_suffix * _CHARS_PER_TOKEN])[:max_suffix]
    ids = [fp] + pre + [fs] + suf + [fm]
    if getattr(hf_tok, "add_bos_token", False) and hf_tok.bos_token_id is not None:
        ids = [hf_tok.bos_token_id] + ids     # DeepSeek-Coder expects <bos>
    return ids


# ── post-processing ─────────────────────────────────────────────────────────

_OPEN, _CLOSE = "([{", ")]}"


def _bracket_imbalance(s: str) -> int:
    return abs(sum(s.count(c) for c in _OPEN) - sum(s.count(c) for c in _CLOSE))


def trim_suffix_overlap(middle: str, prefix: str, suffix: str) -> str:
    """
    Classic FIM failure: the model re-writes what already stands after the cursor
    (e.g. the editor inserted "):" and the model generates "x):" → "f(x)):").
    Cut the longest tail of `middle` equal to the head of the current line's rest,
    but only when that does not make the line's bracket balance worse.
    """
    rest = suffix.split("\n", 1)[0]
    if not rest.strip() or not middle:
        return middle
    line_pre = prefix.rsplit("\n", 1)[-1]
    before = _bracket_imbalance(line_pre + middle + rest)
    for k in range(min(len(middle), len(rest)), 0, -1):
        if not rest[:k].strip() or not middle.endswith(rest[:k]):
            continue
        cand = middle[:-k]
        after = _bracket_imbalance(line_pre + cand + rest)
        if after < before or (k >= 3 and after <= before):
            return cand
    return middle


def postprocess_middle(raw: str, prefix: str, suffix: str,
                       single_line: bool = True, trim_overlap: bool = True) -> str:
    text = raw.split("\n", 1)[0] if single_line else raw
    if trim_overlap:
        text = trim_suffix_overlap(text, prefix, suffix)
    if not suffix.split("\n", 1)[0]:        # hole reaches end of line → no trailing spaces
        text = text.rstrip()
    return text


# ── generation ──────────────────────────────────────────────────────────────

_NEWLINE_IDS: Dict[int, torch.Tensor] = {}


def _newline_ids(hf_tok) -> torch.Tensor:
    """Ids of all vocabulary tokens that contain a line break (cached per tokenizer)."""
    key = id(hf_tok)
    if key not in _NEWLINE_IDS:
        ids = [i for t, i in hf_tok.get_vocab().items()
               if "\n" in hf_tok.convert_tokens_to_string([t])]
        _NEWLINE_IDS[key] = torch.tensor(sorted(ids), dtype=torch.long)
    return _NEWLINE_IDS[key]


class _StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids: torch.Tensor, start_len: int):
        self.stop_ids = stop_ids
        self.start_len = start_len

    def __call__(self, input_ids, scores, **kwargs):
        done = torch.isin(input_ids[:, -1], self.stop_ids.to(input_ids.device))
        return done & (input_ids.shape[1] > self.start_len)


@torch.no_grad()
def fim_generate(model, tokenizer, prefix: str, suffix: str, device: torch.device,
                 kind: str = "t5", family: Optional[str] = None,
                 max_new: int = 48, temperature: float = 0.0, top_k: int = 0,
                 max_prefix: Optional[int] = None, max_suffix: Optional[int] = None,
                 single_line: bool = True, trim_overlap: bool = True,
                 return_raw: bool = False):
    """
    Fill the hole between `prefix` and `suffix`. temperature=0 → greedy decoding.
    Returns the post-processed middle (and the raw model output if return_raw=True).
    """
    d_pre, d_suf = DEFAULT_BUDGETS[kind]
    max_prefix = max_prefix or d_pre
    max_suffix = max_suffix or d_suf
    sample = temperature > 0
    model.eval()

    if kind == "custom":
        ids = build_custom_input(tokenizer, prefix, suffix, max_prefix, max_suffix)
        out = model.generate(ids, max_new=max_new,
                             temperature=temperature if sample else 1.0,
                             top_k=top_k if sample else 1, tokenizer=tokenizer)
        raw = tokenizer.decode(out)

    elif kind == "t5":
        ids = build_t5_input(tokenizer, prefix, suffix, max_prefix, max_suffix)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        _, s1 = t5_sentinels(tokenizer)
        gen = dict(max_new_tokens=max_new, num_beams=1, do_sample=sample,
                   eos_token_id=[s1, tokenizer.eos_token_id],
                   pad_token_id=tokenizer.pad_token_id)
        if single_line:
            gen["stopping_criteria"] = StoppingCriteriaList([_StopOnTokens(_newline_ids(tokenizer), 3)])
        if sample:
            gen.update(temperature=temperature, top_k=top_k or None)
        out = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp), **gen)
        raw = extract_t5_middle(tokenizer, out[0].tolist())

    elif kind == "causal":
        if family not in CAUSAL_FIM_TOKENS:
            raise ValueError(f"family must be one of {list(CAUSAL_FIM_TOKENS)}")
        ids = build_causal_input(tokenizer, prefix, suffix, family, max_prefix, max_suffix)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        gen = dict(max_new_tokens=max_new, do_sample=sample,
                   eos_token_id=tokenizer.eos_token_id,
                   pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        if single_line:
            gen["stopping_criteria"] = StoppingCriteriaList([_StopOnTokens(_newline_ids(tokenizer), len(ids))])
        if sample:
            gen.update(temperature=temperature, top_k=top_k or None)
        out = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp), **gen)
        raw = tokenizer.decode(out[0, len(ids):], skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
    else:
        raise ValueError(f"unknown kind: {kind}")

    middle = postprocess_middle(raw, prefix, suffix, single_line, trim_overlap)
    return (middle, raw) if return_raw else middle
