import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from modules.tokenizers.base_tokenizer import SPECIAL

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=4096, base=10000):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self._cached_len = 0
        self._build_cache(max_seq_len, torch.device("cpu"))
    def _build_cache(self, seq_len, device):
        t     = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self._cached_len = seq_len
    def forward(self, seq_len, device):
        if seq_len > self._cached_len or self.cos_cached.device != device:
            self._build_cache(max(seq_len, self._cached_len * 2), device)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0); sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


@dataclass
class ModelCfg:
    vocab:     int   = 16000
    d_model:   int   = 256
    n_heads:   int   = 8
    n_layers:  int   = 4
    d_ff:      int   = 1024
    max_len:   int   = 256
    dropout:   float = 0.1
    rope_base: int   = 10000


class CausalSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model, n_heads, dropout, rope):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = dropout
        self.resid_drop = nn.Dropout(dropout)
        self.rope = rope
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        cos, sin = self.rope(T, x.device)
        cos = cos.to(q.dtype); sin = sin.to(q.dtype)
        q, k = apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class TransformerBlockRoPE(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, rope):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttentionRoPE(d_model, n_heads, dropout, rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn (self.norm2(x))
        return x


class TokenModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg  = cfg
        self.emb  = nn.Embedding(cfg.vocab, cfg.d_model, padding_idx=0)
        head_dim  = cfg.d_model // cfg.n_heads
        self.rope = RotaryEmbedding(head_dim, max_seq_len=max(cfg.max_len, 1024), base=cfg.rope_base)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlockRoPE(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, self.rope)
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.emb.weight = self.head.weight
        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, x):
        h = self.drop(self.emb(x))
        for block in self.blocks: h = block(h)
        return self.head(self.norm(h))

    @torch.no_grad()
    def generate(self, prefix_ids, max_new, temperature=0.8, top_k=50,
                 stop_at_word_end=True, tokenizer=None):
        self.eval()
        dev = next(self.parameters()).device
        ids = list(prefix_ids); generated = []
        PUNCT = set("()[]{}.,;:=+-*/\\%<>!&|~^@# \t\n\"'`")
        eos_id = tokenizer.eos_id if tokenizer is not None else SPECIAL["<EOS>"]
        for _ in range(max_new):
            x = torch.tensor([ids[-self.cfg.max_len:]], dtype=torch.long, device=dev)
            logits = self(x)[0, -1] / max(temperature, 1e-6)
            if top_k:
                topk_v, _ = torch.topk(logits, top_k)
                logits[logits < topk_v[-1]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            if nxt == eos_id: break
            ids.append(nxt); generated.append(nxt)
            if stop_at_word_end and tokenizer:
                tok = tokenizer.id2token.get(nxt, "")
                if any(c in PUNCT for c in tok): break
        return generated