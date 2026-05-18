import torch, math
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional
from modules.tokenizers.base_tokenizer import CodeTokenizer, SPECIAL
import torch.nn.functional as F

@dataclass
class ModelCfg:
    vocab: int = 8000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    max_len: int = 256
    dropout: float = 0.1


class PositionalEncoding(nn.Module):
    def __init__(self, d: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TokenModel(nn.Module):
    """
    Decoder-only Transformer for causal next-token prediction.
    """
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.emb   = nn.Embedding(cfg.vocab, cfg.d_model, padding_idx=0)
        self.pos   = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        layer      = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
            batch_first=True, norm_first=True
        )
        self.enc   = nn.TransformerEncoder(layer, cfg.n_layers)
        self.head  = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.emb.weight = self.head.weight  # weight tying

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.pos(self.emb(x))
        h = self.enc(h, mask=mask, is_causal=True)
        return self.head(h)

    @torch.no_grad()
    def generate(self, prefix_ids: List[int], max_new: int,
                 temperature: float = 0.8, top_k: int = 50,
                 stop_at_word_end: bool = True,
                 tokenizer: Optional[CodeTokenizer] = None) -> List[int]:
        self.eval()
        dev   = next(self.parameters()).device
        ids   = list(prefix_ids)
        generated = []
        PUNCT_CHARS = set("()[]{}.,;:=+-*/\\%<>!&|~^@# \t\n\"'`")
        for _ in range(max_new):
            x = torch.tensor([ids[-self.cfg.max_len:]], dtype=torch.long, device=dev)
            logits = self(x)[0, -1] / temperature
            if top_k:
                topk_v, _ = torch.topk(logits, top_k)
                logits[logits < topk_v[-1]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            if nxt == SPECIAL["<EOS>"]:
                break
            ids.append(nxt)
            generated.append(nxt)
            if stop_at_word_end and tokenizer:
                tok = tokenizer.id2token.get(nxt, "")
                if any(c in PUNCT_CHARS for c in tok):
                    break
        return generated