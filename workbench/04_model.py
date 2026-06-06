"""PhoneFM model — GPT-style transformer with cardio classification head.

Design notes:
  - Pre-norm decoder-only transformer, no causal mask on the classification path
    (we use bidirectional attention over the 30-day window and pool the final
    token's hidden state for the head — same shape contract used by GPT-EHR).
  - Rotary position embeddings via positions[] tensor — positions = calendar
    day index within the window (0..30), so the model learns time structure
    independently of token-stream length.
  - ~50M-param config by default; bump to ~100M if Workbench GPU is A100+.

  Param budget at default config:
    d=512, n_layers=12, n_heads=8 → ~38M backbone + 5M embedding + 0.5M head
    ≈ 44M trainable. Use d=768/n_layers=12 for ~80M.

Pasted into Workbench. Importable as workbench_model if renamed.
"""

import json
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CELL 1 — config
# ============================================================
@dataclass
class PhoneFMConfig:
    vocab_size:   int
    max_position: int = 64      # 30 days + slack; positions are calendar-day indices
    d_model:      int = 512
    n_layers:     int = 12
    n_heads:      int = 8
    d_ff:         int = 2048
    dropout:      float = 0.1
    max_seq_len:  int = 4096
    pad_id:       int = 0


def load_config_from_vocab(vocab_path: str = "/tmp/vocab.json", **overrides) -> PhoneFMConfig:
    vocab = json.load(open(vocab_path))
    vocab_size = max(vocab.values()) + 1
    return PhoneFMConfig(vocab_size=vocab_size, **overrides)


# ============================================================
# CELL 2 — rotary positional embedding
# ============================================================
class RotaryEmbedding(nn.Module):
    # NOTE: base=100 (not the LLM-standard 10000) because PhoneFM positions are
    # calendar-day indices in [0, ~31]. With base=10000 and head_dim=64, half
    # the rotation dimensions see freq < 1e-3 over the position range — cos≈1,
    # sin≈0 — so they encode no position info. base=100 compresses the
    # frequency range so all dimensions carry signal across [0, 31].
    def __init__(self, dim: int, max_position: int = 8192, base: int = 100):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position = max_position

    def forward(self, positions: torch.Tensor):
        # positions: [B, L] integer positions (e.g., 0..30 day indices)
        freqs = positions.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: [B, H, L, D];  cos,sin: [B, L, D]
    cos = cos.unsqueeze(1); sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


# ============================================================
# CELL 3 — transformer block
# ============================================================
class Block(nn.Module):
    def __init__(self, cfg: PhoneFMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        assert cfg.d_model % cfg.n_heads == 0

    def forward(self, x, cos, sin, key_padding_mask=None):
        # x: [B, L, D]; key_padding_mask: [B, L] (True = real token)
        B, L, D = x.shape
        h = self.ln1(x)
        q = self.q_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)

        # attn_mask: [B, 1, 1, L]; True positions are valid
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
        # F.scaled_dot_product_attention expects mask where True = keep
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.drop.p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        x = x + self.drop(self.o_proj(out))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


# ============================================================
# CELL 4 — full model
# ============================================================
class PhoneFM(nn.Module):
    def __init__(self, cfg: PhoneFMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, max_position=cfg.max_position * 256)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # Classification head: mean-pool over real tokens → linear → logit
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()

    def forward(self, input_ids, positions, attn_mask):
        """input_ids: [B, L] long
           positions: [B, L] long  (calendar-day index)
           attn_mask: [B, L] bool  (True = real token, False = pad)
        """
        h = self.tok_emb(input_ids)
        cos, sin = self.rope(positions)
        for blk in self.blocks:
            h = blk(h, cos, sin, key_padding_mask=attn_mask)
        h = self.ln_f(h)
        # Masked mean pool
        m = attn_mask.float().unsqueeze(-1)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        logit = self.head(pooled).squeeze(-1)
        return logit

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# CELL 5 — quick build smoke test
# ============================================================
if __name__ == "__main__":
    cfg = load_config_from_vocab()
    model = PhoneFM(cfg)
    print(f"vocab_size={cfg.vocab_size}  n_params={model.num_params()/1e6:.1f}M")
    B, L = 2, 1024
    ids = torch.randint(0, cfg.vocab_size, (B, L))
    pos = torch.randint(0, 32, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool); mask[:, 800:] = False
    out = model(ids, pos, mask)
    print(f"logit shape: {tuple(out.shape)}  values: {out.tolist()}")
