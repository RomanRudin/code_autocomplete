# Byte-level BPE

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers import Tokenizer
import os, tempfile
from pathlib import Path

SPECIAL = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

class _IdToTokenView:
    def __init__(self, tk): self.tk = tk
    def get(self, i, default=""):
        t = self.tk.id_to_token(i); return t if t is not None else default
    def __getitem__(self, i):
        t = self.tk.id_to_token(i)
        if t is None: raise KeyError(i)
        return t
    def __contains__(self, i): return self.tk.id_to_token(i) is not None


class BPECodeTokenizer:
    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def __init__(self, vocab_size=16000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq   = min_freq
        self.tk         = None

    def build(self, texts):
        bpe = ByteLevelBPETokenizer()
        bpe.train_from_iterator(
            iter(texts),
            vocab_size=self.vocab_size,
            min_frequency=self.min_freq,
            special_tokens=self.SPECIAL_TOKENS,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp = f.name
        bpe.save(tmp)
        self.tk = Tokenizer.from_file(tmp)
        os.unlink(tmp)
        print(f"[BPE Tokenizer] vocab_size={self.tk.get_vocab_size()}")

    def encode(self, text):
        return [self.bos_id] + self.tk.encode(text).ids + [self.eos_id]

    def decode(self, ids):
        clean = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.tk.decode(clean)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tk.save(str(path))

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.tk = Tokenizer.from_file(str(path))
        obj.vocab_size = obj.tk.get_vocab_size()
        return obj

    @property
    def vocab(self):  return self.tk.get_vocab_size()
    @property
    def pad_id(self): return self.tk.token_to_id("<PAD>") or 0
    @property
    def unk_id(self): return self.tk.token_to_id("<UNK>") or 1
    @property
    def bos_id(self): return self.tk.token_to_id("<BOS>") or 2
    @property
    def eos_id(self): return self.tk.token_to_id("<EOS>") or 3
    @property
    def id2token(self): return _IdToTokenView(self.tk)
    @property
    def built(self):  return self.tk is not None