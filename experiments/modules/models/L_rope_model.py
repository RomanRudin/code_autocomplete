'''
`BiSelfAttentionRoPE` — bidirectional self-attention with RoPE (encoder).
`CausalSelfAttentionRoPE` — causal self-attention with RoPE + optional padding mask (decoder).
`CrossAttention` — standard MHA, no RoPE (Q-decoder vs K/V-encoder).
`EncoderBlockRoPE`, `DecoderBlockRoPE` — pre-norm sub-blocks with residual connections.
'''

import torch
from torch import nn
from dataclasses import dataclass
import torch.nn.functional as F
from modules.tokenizers.BPE_tokenizer import SPECIAL

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


class BiSelfAttentionRoPE(nn.Module):
    """Bidirectional self-attention with RoPE — used in the encoder."""
    def __init__(self, d_model, n_heads, dropout, rope):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = dropout
        self.resid_drop = nn.Dropout(dropout)
        self.rope = rope

    def forward(self, x, key_padding_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                       # (B, H, T, hd) each
        cos, sin = self.rope(T, x.device); cos = cos.to(q.dtype); sin = sin.to(q.dtype)
        q, k = apply_rope(q, k, cos, sin)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :]         # (B, 1, 1, T) bool, True = pad
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class CausalSelfAttentionRoPE(nn.Module):
    """Causal self-attention with RoPE — used in the decoder."""
    def __init__(self, d_model, n_heads, dropout, rope):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = dropout
        self.resid_drop = nn.Dropout(dropout)
        self.rope = rope

    def forward(self, x, key_padding_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        cos, sin = self.rope(T, x.device); cos = cos.to(q.dtype); sin = sin.to(q.dtype)
        q, k = apply_rope(q, k, cos, sin)

        if key_padding_mask is not None:
            # combine causal mask with padding mask → (B, 1, T, T) bool
            causal = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
            )                                                       # (T, T)
            pad = key_padding_mask[:, None, None, :]                # (B, 1, 1, T)
            attn_mask = causal[None, None, :, :] | pad              # (B, 1, T, T)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.attn_drop if self.training else 0.0,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.attn_drop if self.training else 0.0,
            )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class CrossAttention(nn.Module):
    """Standard cross-attention (no RoPE) — Q from decoder, K/V from encoder."""
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, memory, memory_padding_mask=None):
        B, T, C = x.shape
        _, S, _ = memory.shape
        q  = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(memory).view(B, S, 2, self.n_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)
        attn_mask = None
        if memory_padding_mask is not None:
            attn_mask = memory_padding_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class EncoderBlockRoPE(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, rope):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = BiSelfAttentionRoPE(d_model, n_heads, dropout, rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.ffn (self.norm2(x))
        return x


class DecoderBlockRoPE(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, rope):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = CausalSelfAttentionRoPE(d_model, n_heads, dropout, rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
    def forward(self, x, memory, tgt_padding_mask=None, memory_padding_mask=None):
        x = x + self.self_attn(self.norm1(x), key_padding_mask=tgt_padding_mask)
        x = x + self.cross_attn(self.norm2(x), memory, memory_padding_mask=memory_padding_mask)
        x = x + self.ffn(self.norm3(x))
        return x


class LineModel(nn.Module):
    """Encoder-Decoder Transformer with RoPE in self-attention."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg     = cfg
        self.enc_emb = nn.Embedding(cfg.vocab, cfg.d_model, padding_idx=0)
        self.dec_emb = nn.Embedding(cfg.vocab, cfg.d_model, padding_idx=0)
        head_dim     = cfg.d_model // cfg.n_heads
        # one RoPE shared across both encoder + decoder self-attention
        self.rope    = RotaryEmbedding(head_dim, max_seq_len=max(cfg.max_len, 1024), base=cfg.rope_base)

        self.enc_blocks = nn.ModuleList([
            EncoderBlockRoPE(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, self.rope)
            for _ in range(cfg.n_layers)
        ])
        self.dec_blocks = nn.ModuleList([
            DecoderBlockRoPE(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, self.rope)
            for _ in range(cfg.n_layers)
        ])
        self.enc_norm = nn.LayerNorm(cfg.d_model)
        self.dec_norm = nn.LayerNorm(cfg.d_model)
        self.head     = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.dec_emb.weight = self.head.weight                     # weight tying
        self.drop = nn.Dropout(cfg.dropout)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def encode(self, src, src_key_padding_mask=None):
        h = self.drop(self.enc_emb(src))
        for block in self.enc_blocks:
            h = block(h, key_padding_mask=src_key_padding_mask)
        return self.enc_norm(h)

    def forward(self, src, tgt,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None):
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        h = self.drop(self.dec_emb(tgt))
        for block in self.dec_blocks:
            h = block(h, memory,
                      tgt_padding_mask=tgt_key_padding_mask,
                      memory_padding_mask=src_key_padding_mask)
        return self.head(self.dec_norm(h))

    @torch.no_grad()
    def generate(self, prefix_ids, max_new=64, temperature=0.7, top_k=40, tokenizer=None):
        self.eval()
        dev = next(self.parameters()).device
        src = torch.tensor([prefix_ids], dtype=torch.long, device=dev)
        src_pad = (src == 0)
        memory  = self.encode(src, src_key_padding_mask=src_pad)

        bos = tokenizer.bos_id if tokenizer is not None else SPECIAL["<BOS>"]
        eos = tokenizer.eos_id if tokenizer is not None else SPECIAL["<EOS>"]
        dec_ids = [bos]; out_ids = []
        for _ in range(max_new):
            tgt = torch.tensor([dec_ids], dtype=torch.long, device=dev)
            h = self.drop(self.dec_emb(tgt))
            for block in self.dec_blocks:
                h = block(h, memory, memory_padding_mask=src_pad)
            logits = self.head(self.dec_norm(h))[0, -1] / max(temperature, 1e-6)
            if top_k:
                topk_v, _ = torch.topk(logits, top_k)
                logits[logits < topk_v[-1]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            if nxt == eos: break
            dec_ids.append(nxt); out_ids.append(nxt)
        return out_ids