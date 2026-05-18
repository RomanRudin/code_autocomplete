from torch import nn
from torch.nn import functional as F
from typing import List, Optional
import torch
import math
from modules.tokenizers.base_tokenizer import CodeTokenizer, SPECIAL
from dataclasses import dataclass

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



class LineModel(nn.Module):
    """
    Encoder-Decoder Transformer for seq2seq line completion.
    Encoder: reads prefix.  Decoder: generates rest of line.
    """
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.enc_emb  = nn.Embedding(cfg.vocab, cfg.d_model, padding_idx=0)
        self.dec_emb  = nn.Embedding(cfg.vocab, cfg.d_model, padding_idx=0)
        self.enc_pos  = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        self.dec_pos  = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        self.transformer = nn.Transformer(
            cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.n_layers,
            cfg.d_ff, cfg.dropout, batch_first=True, norm_first=True
        )
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.dec_emb.weight = self.head.weight

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        T = tgt.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=src.device)
        enc_out = self.transformer.encoder(
            self.enc_pos(self.enc_emb(src)),
            src_key_padding_mask=src_key_padding_mask
        )
        dec_out = self.transformer.decoder(
            self.dec_pos(self.dec_emb(tgt)),
            enc_out,
            tgt_mask=causal,
            tgt_is_causal=True,
            memory_key_padding_mask=src_key_padding_mask
        )
        return self.head(dec_out)

    @torch.no_grad()
    def generate(self, prefix_ids: List[int], max_new: int = 64,
                 temperature: float = 0.7, top_k: int = 40,
                 tokenizer: Optional[CodeTokenizer] = None) -> List[int]:
        self.eval()
        dev = next(self.parameters()).device
        src = torch.tensor([prefix_ids], dtype=torch.long, device=dev)
        dec_ids = [SPECIAL["<BOS>"]]
        out_ids = []
        for _ in range(max_new):
            tgt = torch.tensor([dec_ids], dtype=torch.long, device=dev)
            logits = self(src, tgt)[0, -1] / temperature
            if top_k:
                topk_v, _ = torch.topk(logits, top_k)
                logits[logits < topk_v[-1]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            if nxt == SPECIAL["<EOS>"]:
                break
            dec_ids.append(nxt)
            out_ids.append(nxt)
        return out_ids