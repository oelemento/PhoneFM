"""PhoneFM v3 model — same backbone as v2, 13 supervised heads at multiple horizons.

Architecture (per docs/v3_spec.md §6):
- Backbone identical to v2 (d_model=384, n_layers=6, n_heads=6, BatchNorm inputs, RoPE).
- 13 independent Linear(d_model, 1) heads, one per (endpoint, horizon).
- n_confounders=9 (was 8 in v2; adds baseline_osa from §2.1).
- Per-endpoint baseline-exclusion masks live OUTSIDE the model (in dataset / loss);
  the model emits all 13 logits unconditionally and the loss applies the mask.

Head list (locked in v3_spec.md §13):
  Primary (10 heads with backprop):
    mortality_30d, mortality_180d, mortality_365d                (no baseline mask)
    t2d_180d, t2d_365d                                            (baseline mask: prior T2D)
    dep_180d, dep_365d                                            (baseline mask: prior depression)
    cv_composite_30d, cv_composite_180d, cv_composite_365d        (no baseline mask, v2 continuity)
  Negative controls (3 heads with backprop, AUROC≈0.5 expected per §11.2):
    skin_neoplasm_365d, refractive_errors_365d, dental_caries_365d  (baseline mask: prior diagnosis)

Backbone state_dict loaded from 04_pretrain_v3.py output via `strict=False` —
pretrain model has no confounder layers and no supervised heads, so those load
randomly while the wearable/EHR/transformer stack inherits the SSL backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse v2 building blocks verbatim — RoPE, MHSA, TransformerBlock are unchanged.
# Importing keeps a single source of truth and means pretrain backbone weights
# share exact parameter names with v3.
from phonefm_model_v2 import (  # type: ignore[import-not-found]
    RotaryPositionalEmbedding,
    TransformerBlock,
)


# ============================================================
# Head list (locked per v3_spec.md §13)
# ============================================================

# (endpoint, horizon_days). Order matters for state_dict layout — DO NOT REORDER
# without re-saving config.json.
PRIMARY_HEADS: tuple[tuple[str, int], ...] = (
    ("mortality", 30),
    ("mortality", 180),
    ("mortality", 365),
    ("t2d", 180),
    ("t2d", 365),
    ("dep", 180),
    ("dep", 365),
    ("cv_composite", 30),
    ("cv_composite", 180),
    ("cv_composite", 365),
)

NEGATIVE_CONTROL_HEADS: tuple[tuple[str, int], ...] = (
    ("skin_neoplasm", 365),
    ("refractive_errors", 365),
    ("dental_caries", 365),
)

ALL_HEADS: tuple[tuple[str, int], ...] = PRIMARY_HEADS + NEGATIVE_CONTROL_HEADS

# Endpoints whose label is only valid for participants without prior diagnosis.
# Tokenizer emits mask_{ep}_{h}d = 0 for these when baseline flag is set.
BASELINE_EXCLUDED_ENDPOINTS = frozenset({
    "t2d", "dep", "skin_neoplasm", "refractive_errors", "dental_caries",
})

# v3_spec.md §14.10: best.pt selection metric — sum of these AUROCs.
PRIMARY_BEST_METRIC_HEADS: tuple[str, ...] = (
    "cv_composite_30d", "mortality_365d", "t2d_365d", "dep_365d",
)


def head_name(endpoint: str, horizon_days: int) -> str:
    """Canonical head name used for state_dict, labels dict, and config.json."""
    return f"{endpoint}_{horizon_days}d"


# ============================================================
# Config
# ============================================================

@dataclass
class PhoneFMV3Config:
    # Vocab + special tokens (unchanged from v2)
    vocab_size: int = 6007
    pad_id: int = 0
    cls_id: int = 1
    eos_id: int = 2
    wearable_marker_id: int = 3

    # Transformer (same as v2 default)
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.15      # v2 lesson §14.2: 0.10 → 0.15 to slow HF/composite overfit
    max_seq_len: int = 512

    # Wearable feature vector (unchanged — same tokenizer output)
    n_wearable_features: int = 11

    # Confounders — n_confounders = 9 in v3 (adds baseline_osa per §2.1)
    # Order MUST match 02_tokenizer_v3.precompute_confounders output:
    #   0: age (years, raw)
    #   1: sex_female (0/1)
    #   2: bmi (kg/m^2, raw)
    #   3: sbp (mmHg, raw)
    #   4: dbp (mmHg, raw)
    #   5: baseline_cad (0/1)
    #   6: baseline_cancer (0/1)
    #   7: prior_afib (0/1)
    #   8: baseline_osa (0/1)  <-- NEW in v3
    n_confounders: int = 9

    # Token-type embedding cardinality (same as v2; v3 doesn't add new token types
    # because pretraining MASK_ID is a vocab id, not a type)
    n_token_types: int = 8

    # Heads — list of (endpoint, horizon_days) tuples, persisted to config.json
    head_specs: tuple[tuple[str, int], ...] = ALL_HEADS


# ============================================================
# PhoneFM v3
# ============================================================

class PhoneFMV3(nn.Module):
    """v3 multi-horizon, multi-domain transformer.

    Inputs (same shapes as v2 forward signature):
      input_ids:        (B, L) int64
      token_types:      (B, L) int64
      day_index:        (B, L) int64
      wearable_feats:   (B, L, n_wearable_features) float
      confounders:      (B, n_confounders) float
      attn_mask:        (B, L) bool   — True = keep, False = pad

    Output:
      dict {head_name: (B,) float logit} for all 13 heads.
    """

    def __init__(self, cfg: PhoneFMV3Config):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.type_emb = nn.Embedding(cfg.n_token_types, cfg.d_model)

        # BatchNorm inputs — v2 stability lesson §14.2 (steps range 0-100k → bf16 overflow without)
        self.wearable_input_norm = nn.BatchNorm1d(cfg.n_wearable_features)
        self.wearable_proj = nn.Sequential(
            nn.Linear(cfg.n_wearable_features, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        self.confounder_input_norm = nn.BatchNorm1d(cfg.n_confounders)
        self.confounder_proj = nn.Sequential(
            nn.Linear(cfg.n_confounders, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        self.rope = RotaryPositionalEmbedding(
            d_head=cfg.d_model // cfg.n_heads, max_positions=256
        )

        # Reuse v2 TransformerBlock via duck-typed config attribute access
        # (TransformerBlock reads cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout).
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)

        # Heads — one per (endpoint, horizon_days) tuple. nn.ModuleDict so we can
        # iterate by name and so state_dict keys match what 05_train_v3.py expects.
        self.head_names: list[str] = [head_name(ep, h) for (ep, h) in cfg.head_specs]
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
        h = self.tok_emb(input_ids) + self.type_emb(token_types)

        wearable_mask = (token_types == 3).unsqueeze(-1)  # 3 = TYPE_WEARABLE
        B, L, F = wearable_feats.shape
        wearable_norm = self.wearable_input_norm(
            wearable_feats.reshape(B * L, F)
        ).reshape(B, L, F)
        wearable_h = self.wearable_proj(wearable_norm)
        h = torch.where(wearable_mask, wearable_h + self.type_emb(token_types), h)

        confounders_norm = self.confounder_input_norm(confounders)
        h[:, 0, :] = h[:, 0, :] + self.confounder_proj(confounders_norm)

        cos, sin = self.rope(day_index)
        for blk in self.blocks:
            h = blk(h, cos, sin, key_padding_mask=attn_mask)
        h = self.ln_final(h)

        cls = h[:, 0, :]  # (B, d_model)
        return {name: self.heads[name](cls).squeeze(-1) for name in self.head_names}


# ============================================================
# Loss — masked multi-head BCE per v3_spec.md §7
# ============================================================

def multihead_masked_bce_loss(
    logits: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    pos_weights: dict[str, float],
    head_weights: Optional[dict[str, float]] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sum of masked BCE-with-logits across heads.

    Per head:
        bce_per_sample = BCE(logit, label, pos_weight)             # (B,)
        loss_head     = (bce_per_sample * mask).sum() / mask.sum().clamp(min=1)

    The mask is binary (1=use, 0=skip). For heads without baseline exclusion
    (mortality, cv_composite), pass mask = torch.ones_like(label) — handled
    automatically if masks dict omits the key.

    Args:
      logits:       dict head_name -> (B,) float
      labels:       dict head_name -> (B,) float in {0,1}
      masks:        dict head_name -> (B,) float in {0,1}; missing key -> all-ones
      pos_weights:  dict head_name -> scalar
      head_weights: dict head_name -> scalar (default 1.0 each; 0.0 = drop)

    Returns:
      (total_loss, per_head_loss_dict_for_logging)
    """
    head_weights = head_weights or {k: 1.0 for k in logits}
    any_logit = next(iter(logits.values()))
    total = torch.zeros((), device=any_logit.device, dtype=any_logit.dtype)
    per_head_log: dict[str, float] = {}

    for name, logit in logits.items():
        if name not in labels:
            continue
        hw = head_weights.get(name, 1.0)
        if hw == 0.0:
            continue
        label = labels[name].to(logit.dtype)
        mask = masks.get(name, torch.ones_like(label))
        pw = torch.tensor(pos_weights[name], device=logit.device, dtype=logit.dtype)

        bce = F.binary_cross_entropy_with_logits(
            logit, label, pos_weight=pw, reduction="none"
        )  # (B,)
        denom = mask.sum().clamp(min=1.0)
        loss_head = (bce * mask).sum() / denom

        total = total + hw * loss_head
        per_head_log[name] = float(loss_head.detach().item())

    return total, per_head_log


# ============================================================
# Pretrain backbone loading (v3_spec.md §11.1)
# ============================================================

def load_pretrained_backbone(
    model: PhoneFMV3,
    backbone_state_path: str,
) -> tuple[list[str], list[str]]:
    """Load a backbone_only.pt produced by 04_pretrain_v3.py into a fresh v3 model.

    Pretrain model has fewer parameters than v3 supervised:
      - same: tok_emb, type_emb, wearable_input_norm, wearable_proj, rope, blocks, ln_final
      - missing: confounder_input_norm, confounder_proj (pretrain has no confounder input)
      - missing: heads.* (pretrain has its own MSE regression head, stripped out by 04_pretrain_v3.py
                  when saving backbone_only.pt)

    Uses strict=False so randomly-init confounder + heads survive. Returns the
    (missing_keys, unexpected_keys) tuple so the training script can log it and
    abort if unexpected keys appear (which would indicate a v2/v3 mismatch).
    """
    state = torch.load(backbone_state_path, map_location="cpu")
    result = model.load_state_dict(state, strict=False)
    return list(result.missing_keys), list(result.unexpected_keys)
