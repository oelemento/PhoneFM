"""PhoneFM v2 model — mixed continuous (wearable) + discrete (EHR) tokens.

Architecture decisions driven by:
  - Master 2022 Nat Med (s41591-022-02012-w): daily steps suffice to detect chronic-disease signal in AoU.
  - Zheng 2024 Nat Med (s41591-024-03155-8): daily sleep stages (REM%, deep%, light%), irregularity,
    and HRV (SDANN from minute-HR) are the strongest cardio-relevant wearable signals in AoU.
  - v1 post-mortem: 99.3% wearable / 0.7% EHR token density caused attention dilution. v2 collapses
    each day of wearable into one float-feature-vector token, giving ~78% wearable / 22% EHR ratio.

Sequence layout per window (180-day input → +30d prediction):
  [CLS]  [DAY_0_WEARABLE]  [DAY_1_WEARABLE]  ...  [DAY_179_WEARABLE]
         + EHR event tokens at their actual day_index (interleaved by sort key)
         [EOS] [PAD]...

CLS holds projected demographics (age, sex, BMI, baseline CAD/cancer/smoking/alcohol/SBP).

Output: 4 endpoint heads (AFib, MI, HF, CV death) + 1 composite head = 5 sigmoids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Config
# ============================================================

@dataclass
class PhoneFMV2Config:
    # token vocabulary (EHR side only; wearable is float)
    vocab_size: int = 6007  # matches v1 EHR vocab (IDs 0-6006); wearable tokens are NOT in vocab
    pad_id: int = 0
    cls_id: int = 1
    eos_id: int = 2
    wearable_marker_id: int = 3  # placeholder id for wearable positions in the input_ids tensor
    # ehr token-type IDs (just for type embedding; the actual id is the standard vocab id)

    # transformer
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    max_seq_len: int = 512

    # wearable feature vector
    n_wearable_features: int = 11
    # ordering MUST match the tokenizer's column order:
    #   0: total_steps
    #   1: mean_hr
    #   2: resting_hr
    #   3: max_hr
    #   4: sdann   (HRV from minute-HR)
    #   5: rem_pct
    #   6: deep_pct
    #   7: light_pct
    #   8: sleep_duration_hr
    #   9: sleep_onset_hour
    #  10: sleep_irregularity_7d_sd

    # demographics / baseline confounders
    n_confounders: int = 8
    # ordering:
    #   0: age (years, standardised)
    #   1: sex_female (0/1)
    #   2: bmi (standardised)
    #   3: baseline_cad (0/1)
    #   4: baseline_cancer (0/1)
    #   5: smoking_100cigs (0/1)
    #   6: alcohol_ever (0/1)
    #   7: sbp_systolic_mmHg (standardised)

    # heads
    n_endpoint_heads: int = 4   # afib, mi, hf, cv_death
    use_composite_head: bool = True   # extra composite head (OR of all 4); shared backbone

    # token-type embedding cardinality:
    #   0: PAD, 1: CLS, 2: EOS, 3: WEARABLE, 4: DX10, 5: MED, 6: PX10, 7: LAB
    n_token_types: int = 8


# ============================================================
# Building blocks
# ============================================================

class RotaryPositionalEmbedding(nn.Module):
    """RoPE on a small max position range. Day-index goes 0..179 (or up to ~210 if EHR fences extend)."""

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, d_head: int, max_positions: int = 256, base: float = 100.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        positions = torch.arange(max_positions).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: (B, L) int
        return self.cos[positions], self.sin[positions]


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q,k: (B, n_heads, L, d_head); cos,sin: (B, L, d_head/2)
    cos = cos.unsqueeze(1)  # (B,1,L,d_head/2)
    sin = sin.unsqueeze(1)
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: PhoneFMV2Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout_p = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        # key_padding_mask: (B, L) bool, True = valid (keep), False = pad
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = (~key_padding_mask)[:, None, None, :]  # (B,1,1,L), True = mask out
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: PhoneFMV2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), cos, sin, key_padding_mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


# ============================================================
# PhoneFM v2
# ============================================================

class PhoneFMV2(nn.Module):
    """Mixed-input transformer for cardiovascular risk with separate heads per endpoint.

    Inputs (all batched):
      input_ids:        (B, L) int64 — EHR vocab ids for EHR positions, wearable_marker_id for wearable
                        positions, PAD elsewhere. CLS at position 0, EOS at last valid.
      token_types:      (B, L) int64 — see cfg.n_token_types (0=PAD,1=CLS,2=EOS,3=WEARABLE,4=DX10,...).
      day_index:        (B, L) int64 — 0..179 (or 0 for CLS/EOS).
      wearable_feats:   (B, L, n_wearable_features) float — valid only where token_types==WEARABLE; 0 elsewhere.
      confounders:      (B, n_confounders) float — projected and added to the CLS embedding.
      attn_mask:        (B, L) bool — True = keep, False = pad.
    """

    def __init__(self, cfg: PhoneFMV2Config):
        super().__init__()
        self.cfg = cfg

        # EHR / special token embedding (PAD, CLS, EOS, WEARABLE_MARKER, then vocab IDs)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        # Type embedding (distinguishes wearable vs DX10 vs MED ...)
        self.type_emb = nn.Embedding(cfg.n_token_types, cfg.d_model)

        # Wearable feature projection: float vector -> d_model
        self.wearable_proj = nn.Sequential(
            nn.Linear(cfg.n_wearable_features, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # Confounder projection: float vector -> d_model (added to CLS)
        self.confounder_proj = nn.Sequential(
            nn.Linear(cfg.n_confounders, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # RoPE on day_index
        self.rope = RotaryPositionalEmbedding(d_head=cfg.d_model // cfg.n_heads, max_positions=256)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)

        # Heads
        head_names = ["afib", "mi", "hf", "cv_death"][:cfg.n_endpoint_heads]
        self.head_names = head_names + (["composite"] if cfg.use_composite_head else [])
        self.heads = nn.ModuleDict({
            name: nn.Linear(cfg.d_model, 1) for name in self.head_names
        })

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, L)
        token_types: torch.Tensor,      # (B, L)
        day_index: torch.Tensor,        # (B, L)
        wearable_feats: torch.Tensor,   # (B, L, F)
        confounders: torch.Tensor,      # (B, C)
        attn_mask: torch.Tensor,        # (B, L) bool
    ) -> dict[str, torch.Tensor]:
        # Base embedding
        h = self.tok_emb(input_ids) + self.type_emb(token_types)  # (B, L, d)

        # Wearable mix-in: where token_type == WEARABLE, replace tok_emb contribution with wearable projection
        wearable_mask = (token_types == 3).unsqueeze(-1)  # (B, L, 1)
        wearable_h = self.wearable_proj(wearable_feats)   # (B, L, d)
        h = torch.where(wearable_mask, wearable_h + self.type_emb(token_types), h)

        # Confounders added to CLS (position 0)
        h[:, 0, :] = h[:, 0, :] + self.confounder_proj(confounders)

        # RoPE
        cos, sin = self.rope(day_index)  # (B, L, d_head/2) each

        for blk in self.blocks:
            h = blk(h, cos, sin, key_padding_mask=attn_mask)

        h = self.ln_final(h)

        # Pool from CLS (position 0)
        cls = h[:, 0, :]

        logits = {name: self.heads[name](cls).squeeze(-1) for name in self.head_names}
        return logits


# ============================================================
# Loss helper
# ============================================================

def multihead_bce_loss(
    logits: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    pos_weights: dict[str, float],
    head_weights: Optional[dict[str, float]] = None,
) -> torch.Tensor:
    """Sum of BCE-with-logits across heads, optionally weighted.

    labels:       per-head (B,) float in {0,1}
    pos_weights:  per-head positive-class weight (use train pos_rate^-1 minus 1, like v1)
    head_weights: per-head loss weight (default 1.0 each)
    """
    head_weights = head_weights or {k: 1.0 for k in logits}
    any_logit = next(iter(logits.values()))
    total = torch.zeros((), device=any_logit.device, dtype=any_logit.dtype)
    for name, logit in logits.items():
        if name not in labels:
            continue
        pw = torch.tensor(pos_weights[name], device=logit.device, dtype=logit.dtype)
        l = F.binary_cross_entropy_with_logits(
            logit, labels[name].float(), pos_weight=pw, reduction="mean"
        )
        total = total + head_weights.get(name, 1.0) * l
    return total
