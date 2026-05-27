# character-level tokenizer with proper punctuation handling

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
    # All Python punctuation that should be treated as standalone tokens
    PUNCT = set("()[]{}.,;:=+-*/\\%<>!&|~^@#\"'")

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.token2id: Dict[str, int] = dict(SPECIAL)
        self.id2token: Dict[int, str] = {v: k for k, v in SPECIAL.items()}
        self.built = False

    # ── build ──────────────────────────────────────────────
    def build(self, texts: List[str], min_freq: int = 3):
        # 1. Count frequencies of all whole tokens from _raw_split
        freq: Dict[str, int] = defaultdict(int)
        for t in texts:
            for tok in self._raw_split(t):
                freq[tok] += 1
        sorted_tokens = sorted(freq.items(), key=lambda x: -x[1])

        # 2. Add high-frequency whole tokens to vocab
        for tok, cnt in sorted_tokens:
            if cnt < min_freq:
                break
            if tok not in self.token2id and len(self.token2id) < self.vocab_size:
                idx = len(self.token2id)
                self.token2id[tok] = idx
                self.id2token[idx] = tok

        # 3. Guarantee fallback coverage — every individual char that appears
        #    in training data gets an id (so char fallback never emits <UNK>)
        seen_chars = set()
        for t in texts:
            seen_chars.update(t)

        # Add all seen chars first (priority over the static alphabet)
        for c in seen_chars:
            if c not in self.token2id and len(self.token2id) < self.vocab_size:
                idx = len(self.token2id)
                self.token2id[c] = idx
                self.id2token[idx] = c

        # Then top up with a fixed alphabet in case any of these chars
        # weren't seen in training (still useful for inference on new files)
        FALLBACK_CHARS = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            "_ \t\n\r"
            "()[]{}.,;:=+-*/\\%<>!&|~^@#\"'"
        )
        for c in FALLBACK_CHARS:
            if c not in self.token2id and len(self.token2id) < self.vocab_size:
                idx = len(self.token2id)
                self.token2id[c] = idx
                self.id2token[idx] = c

        self.built = True
        print(f"[Tokenizer] vocab_size={len(self.token2id)}, "
              f"unique seen chars={len(seen_chars)}")

    def _raw_split(self, text: str) -> List[str]:
        """
        Token stream:
          • indent characters (each \t or space at line start = one token)
          • identifiers / numbers (runs of [A-Za-z0-9_])
          • each punctuation char = one token
          • each \n = one token
        """
        tokens = []
        for line in text.splitlines(keepends=True):
            # leading whitespace = indent tokens (one per char)
            stripped = line.lstrip(" \t")
            indent = line[: len(line) - len(stripped)]
            for ch in indent:
                tokens.append(ch)

            # split the rest
            buf = ""
            for ch in stripped:
                if ch in self.PUNCT:
                    if buf:
                        tokens.append(buf)
                        buf = ""
                    tokens.append(ch)
                elif ch == " " or ch == "\t":
                    if buf:
                        tokens.append(buf)
                        buf = ""
                    tokens.append(ch)
                elif ch == "\n" or ch == "\r":
                    if buf:
                        tokens.append(buf)
                        buf = ""
                    tokens.append("\n")
                else:
                    # identifier / number / unicode char — keep building
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
                # char fallback — should rarely hit <UNK> after the build fix
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
        with open(path, "w", encoding="utf-8") as f:
            print(self.token2id)
            json.dump({"token2id": self.token2id}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CodeTokenizer":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        obj = cls()
        obj.token2id = {k: int(v) for k, v in d["token2id"].items()}
        obj.id2token = {v: k for k, v in obj.token2id.items()}
        obj.built = True
        return obj

    @property
    def pad_id(self):  return SPECIAL["<PAD>"]
    @property
    def unk_id(self):  return SPECIAL["<UNK>"]
    @property
    def eos_id(self):  return SPECIAL["<EOS>"]
    @property
    def bos_id(self):  return SPECIAL["<BOS>"]
    @property
    def vocab(self):   return len(self.token2id)