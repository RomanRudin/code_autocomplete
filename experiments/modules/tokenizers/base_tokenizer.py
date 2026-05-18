# character-level BPE-lite

from typing import List, Dict
from collections import defaultdict
import json

SPECIAL = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

class CodeTokenizer:
    """
    Simple sub-word tokenizer tailored for Python source code.
    Splits on whitespace/punctuation, keeps indentation tokens,
    and falls back to characters for unknowns.
    """
    PUNCT = set(",;&|~^@#")

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.token2id: Dict[str, int] = dict(SPECIAL)
        self.id2token: Dict[int, str] = {v: k for k, v in SPECIAL.items()}
        self.built = False

    # ── build ──────────────────────────────────────────────
    def build(self, texts: List[str], min_freq: int = 3):
        freq: Dict[str, int] = defaultdict(int)
        for t in texts:
            for tok in self._raw_split(t):
                freq[tok] += 1
        sorted_tokens = sorted(freq.items(), key=lambda x: -x[1])
        for tok, cnt in sorted_tokens:
            if cnt < min_freq:
                break
            if tok not in self.token2id and len(self.token2id) < self.vocab_size:
                idx = len(self.token2id)
                self.token2id[tok] = idx
                self.id2token[idx] = tok
        # fill remaining slots with single chars
        for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,_ \t\n":  #for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ \t\n":
            if c not in self.token2id and len(self.token2id) < self.vocab_size:
                idx = len(self.token2id)
                self.token2id[c] = idx
                self.id2token[idx] = c
        self.built = True
        print(f"[Tokenizer] vocab_size={len(self.token2id)}")

    def _raw_split(self, text: str) -> List[str]:
        tokens = []
        for line in text.splitlines(keepends=True):
            # capture leading whitespace as indent token
            stripped = line.lstrip(" \t")
            indent = line[: len(line) - len(stripped)]
            for ch in indent:
                tokens.append(ch)
            # split remainder on punctuation / spaces
            buf = ""
            for ch in stripped:
                if ch in self.PUNCT or ch in " \t\n\r":
                    if ch == "\n" or ch.strip():
                        if buf:
                            tokens.append(buf)
                        tokens.append(ch)
                        continue
                    if buf:
                        buf += ch
                        tokens.append(buf)
                        buf = ""
                        tokens.append("\n")
                else:
                    buf += ch
            if buf:
                tokens.append(buf)
        return tokens

    def encode(self, text: str) -> List[int]:
        ids = [SPECIAL["<BOS>"]]
        for tok in self._raw_split(text):
            if tok in self.token2id:
                ids.append(self.token2id[tok])
            else:
                # char fallback
                for ch in tok:
                    ids.append(self.token2id.get(ch, SPECIAL["<UNK>"]))
        ids.append(SPECIAL["<EOS>"])
        return ids

    def decode(self, ids: List[int]) -> str:
        parts = []
        for i in ids:
            tok = self.id2token.get(i, "")
            if tok in SPECIAL:
                continue
            parts.append(tok)
        return "".join(parts)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"token2id": self.token2id}, f)

    @classmethod
    def load(cls, path: str) -> "CodeTokenizer":
        with open(path) as f:
            d = json.load(f)
        obj = cls()
        obj.token2id = {k: int(v) for k, v in d["token2id"].items()}
        obj.id2token = {v: k for k, v in obj.token2id.items()}
        obj.built = True
        return obj

    @property
    def pad_id(self):  return SPECIAL["<PAD>"]
    @property
    def eos_id(self):  return SPECIAL["<EOS>"]
    @property
    def bos_id(self):  return SPECIAL["<BOS>"]
    @property
    def vocab(self):   return len(self.token2id)