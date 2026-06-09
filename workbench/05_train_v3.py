"""PhoneFM v3 trainer — 13 heads, multi-horizon BCE with per-endpoint masks.

Initialises backbone weights from 04_pretrain_v3.py's backbone_only.pt, then
fine-tunes 13 (endpoint, horizon) heads + the backbone for a small number of
epochs. Best-checkpoint selection uses the sum of AUROCs across primary heads
(cv_composite_30d + mortality_365d + t2d_365d + dep_365d) per v3_spec.md §14.10.

Run on AoU Workbench A100:
    nohup python -u 05_train_v3.py > ~/workspace/phonefm-data/phonefm_v3/run.log 2>&1 &

Inputs:
  ~/workspace/phonefm-data/tokenized_v3/{train,val}_*.parquet
  ~/workspace/phonefm-data/phonefm_pretrain_v3/backbone_only.pt

Outputs:
  ~/workspace/phonefm-data/phonefm_v3/
    best.pt        backbone + all heads + optimizer at the best-metric epoch
    last.pt        same at the latest epoch (so a crashed run can resume)
    config.json    config + HP + val_metrics_at_save + head_names
    metrics.json   per-epoch val metrics
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-not-found]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phonefm_dataset_v3 import (   # type: ignore[import-not-found]  # noqa: E402
    HEAD_NAMES,
    make_loader_v3,
    report_label_distribution,
)
from phonefm_model_v3 import (     # type: ignore[import-not-found]  # noqa: E402
    PhoneFMV3,
    PhoneFMV3Config,
    PRIMARY_BEST_METRIC_HEADS,
    load_pretrained_backbone,
    multihead_masked_bce_loss,
)


# ============================================================
# Config — inherits v2 stability defaults per v3_spec.md §14
# ============================================================

HP: dict[str, Any] = dict(
    # Backbone (must match pretrain HP for state_dict compatibility)
    d_model=384, n_layers=6, n_heads=6, d_ff=1536, dropout=0.15, max_seq_len=512,
    n_wearable_features=11, n_confounders=9,
    # Optimisation — v2 lessons §14.2
    batch_size=32, grad_accum=2,           # effective batch 64
    lr=1e-4, weight_decay=0.15, warmup_steps=1000, epochs=5, grad_clip=1.0,
    # Early stopping per §14.1: stop if best metric hasn't improved for 3 epochs
    early_stop_patience=3,
    # Head-drop threshold: any head with <50 train positives gets head_weight=0
    min_pos_for_training=50,
    # Best-metric formula
    best_metric="sum_primary_auroc",
    seed=20260609,
)

DATA_DIR = Path("/home/jupyter/workspace/phonefm-data/tokenized_v3")
PRETRAIN_DIR = Path("/home/jupyter/workspace/phonefm-data/phonefm_pretrain_v3")
OUT_DIR = Path("/home/jupyter/workspace/phonefm-data/phonefm_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BACKBONE_INIT_PATH = PRETRAIN_DIR / "backbone_only.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(HP["seed"]); np.random.seed(HP["seed"])

print(f"device={DEVICE}  out={OUT_DIR}", flush=True)


# ============================================================
# Data
# ============================================================

print("loading train/val...", flush=True)
train_loader = make_loader_v3(
    str(DATA_DIR / "train_*.parquet"),
    batch_size=HP["batch_size"], num_workers=4, shuffle=True,
    max_seq_len=HP["max_seq_len"], n_confounders=HP["n_confounders"],
)
val_loader = make_loader_v3(
    str(DATA_DIR / "val_*.parquet"),
    batch_size=HP["batch_size"], num_workers=2, shuffle=False,
    max_seq_len=HP["max_seq_len"], n_confounders=HP["n_confounders"],
)

train_stats = report_label_distribution(str(DATA_DIR / "train_*.parquet"))
print("train label distribution (after baseline-exclusion mask):")
for name, s in train_stats.items():
    print(f"  {name:<28s} n_valid={s['n_valid']:>7,d}  n_pos={s['n_pos']:>6,d}"
          f"  pos_rate_among_valid={s['pos_rate_among_valid']:.4f}"
          f"  pos_weight={s['pos_weight_for_bce']:.2f}",
          flush=True)

POS_WEIGHTS = {name: train_stats[name]["pos_weight_for_bce"] for name in HEAD_NAMES}
HEAD_WEIGHTS = {
    name: (1.0 if train_stats[name]["n_pos"] >= HP["min_pos_for_training"] else 0.0)
    for name in HEAD_NAMES
}
_dropped = [n for n, w in HEAD_WEIGHTS.items() if w == 0.0]
if _dropped:
    print(f"WARNING: dropping heads {_dropped} from loss "
          f"(<{HP['min_pos_for_training']} positives in train)", flush=True)

# Preflight: verify val has enough positives for the best-metric primary heads.
# If we skip this, a tiny val + rare endpoint produces NaN AUROC for every
# primary head → sum_primary_auroc = 0.0 every epoch → best.pt saves at epoch 0
# with random-init heads, then early-stops with effectively no supervised
# training applied. Adversarial reviewer's silent-failure scenario.
val_stats = report_label_distribution(str(DATA_DIR / "val_*.parquet"))
print("val label distribution (after baseline-exclusion mask):")
for name, s in val_stats.items():
    print(f"  {name:<28s} n_valid={s['n_valid']:>7,d}  n_pos={s['n_pos']:>6,d}", flush=True)

_VAL_NPOS_MIN = 5
_primary_npos = {h: val_stats[h]["n_pos"] for h in PRIMARY_BEST_METRIC_HEADS if h in val_stats}
_primary_viable = [h for h, n in _primary_npos.items() if n >= _VAL_NPOS_MIN]
_primary_starved = [h for h, n in _primary_npos.items() if n < _VAL_NPOS_MIN]
if not _primary_viable:
    raise RuntimeError(
        f"All primary best-metric heads have val n_pos < {_VAL_NPOS_MIN}: "
        f"{_primary_npos}. The sum-of-primary-AUROC metric would be 0.0 every "
        f"epoch and best.pt would never reflect actual training. Either lower "
        f"_VAL_NPOS_MIN, switch best-metric, or fix the val split."
    )
if _primary_starved:
    print(f"WARNING: primary heads starved in val (n_pos<{_VAL_NPOS_MIN}): {_primary_starved}. "
          f"Best-metric will be sum over the {len(_primary_viable)} viable primary heads "
          f"{_primary_viable}.", flush=True)


# ============================================================
# Model — fresh v3, then load pretrained backbone
# ============================================================

cfg = PhoneFMV3Config(
    d_model=HP["d_model"], n_layers=HP["n_layers"], n_heads=HP["n_heads"],
    d_ff=HP["d_ff"], dropout=HP["dropout"], max_seq_len=HP["max_seq_len"],
    n_wearable_features=HP["n_wearable_features"], n_confounders=HP["n_confounders"],
)
model = PhoneFMV3(cfg).to(DEVICE)
print(f"model: {model.num_params()/1e6:.1f}M params  heads={model.head_names}", flush=True)

if BACKBONE_INIT_PATH.exists():
    print(f"loading pretrained backbone from {BACKBONE_INIT_PATH}", flush=True)
    missing, unexpected = load_pretrained_backbone(model, str(BACKBONE_INIT_PATH))
    # Expected missing keys: confounder_*.* and heads.*.* (pretrain has neither)
    # Unexpected keys = pretrain state has something v3 doesn't — should be empty
    if unexpected:
        raise RuntimeError(
            f"pretrained backbone has UNEXPECTED keys not in v3 model: {unexpected}. "
            f"This indicates a v2/v3 architecture mismatch — abort training."
        )
    expected_missing_prefixes = ("confounder_input_norm.", "confounder_proj.", "heads.")
    unexpected_missing = [
        k for k in missing if not k.startswith(expected_missing_prefixes)
    ]
    if unexpected_missing:
        raise RuntimeError(
            f"missing keys NOT in expected new-layer prefixes: {unexpected_missing}. "
            f"Pretrained backbone is incomplete — abort."
        )
    print(f"  loaded backbone OK. {len(missing)} missing keys (confounder layers + heads): randomly initialised.",
          flush=True)
else:
    print(f"WARNING: no pretrained backbone at {BACKBONE_INIT_PATH}. "
          f"Training from random init — v3 will likely underperform v2.", flush=True)


# ============================================================
# Optimiser + schedule
# ============================================================

decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (no_decay if p.ndim < 2 or n.endswith(".bias") else decay).append(p)
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": HP["weight_decay"]},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=HP["lr"], betas=(0.9, 0.95),
)

total_steps = HP["epochs"] * math.ceil(len(train_loader) / HP["grad_accum"])
def lr_at(step: int) -> float:
    if step < HP["warmup_steps"]:
        return HP["lr"] * step / max(1, HP["warmup_steps"])
    progress = (step - HP["warmup_steps"]) / max(1, total_steps - HP["warmup_steps"])
    return HP["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

print(f"total_optimiser_steps={total_steps:,}", flush=True)


# ============================================================
# Eval helper — per-head AUROC/AUPRC with mask-aware filtering
# ============================================================

@torch.no_grad()
def evaluate(loader) -> dict:
    model.eval()
    preds: dict[str, list[np.ndarray]] = {name: [] for name in HEAD_NAMES}
    labels: dict[str, list[np.ndarray]] = {name: [] for name in HEAD_NAMES}
    masks: dict[str, list[np.ndarray]] = {name: [] for name in HEAD_NAMES}
    for batch in loader:
        out = model(
            batch["input_ids"].to(DEVICE),
            batch["token_types"].to(DEVICE),
            batch["day_index"].to(DEVICE),
            batch["wearable_feats"].to(DEVICE),
            batch["confounders"].to(DEVICE),
            batch["attn_mask"].to(DEVICE),
        )
        for name in HEAD_NAMES:
            preds[name].append(torch.sigmoid(out[name].float()).cpu().numpy())
            labels[name].append(batch["labels"][name].numpy())
            masks[name].append(batch["masks"][name].numpy())
    metrics: dict[str, float] = {}
    for name in HEAD_NAMES:
        p = np.concatenate(preds[name])
        y = np.concatenate(labels[name]).astype(np.int64)
        m = np.concatenate(masks[name]).astype(bool)
        if m.sum() < 50 or y[m].sum() < 5 or y[m].sum() > m.sum() - 5:
            metrics[f"{name}_auroc"] = float("nan")
            metrics[f"{name}_auprc"] = float("nan")
            metrics[f"{name}_n_valid"] = int(m.sum())
            metrics[f"{name}_n_pos"] = int(y[m].sum())
            continue
        metrics[f"{name}_auroc"] = float(roc_auc_score(y[m], p[m]))
        metrics[f"{name}_auprc"] = float(average_precision_score(y[m], p[m]))
        metrics[f"{name}_pos_rate"] = float(y[m].mean())
        metrics[f"{name}_mean_pred"] = float(p[m].mean())
        metrics[f"{name}_n_valid"] = int(m.sum())
        metrics[f"{name}_n_pos"] = int(y[m].sum())
    model.train()
    return metrics


def sum_primary_auroc(metrics: dict[str, float]) -> tuple[float, int]:
    """Sum of AUROCs across PRIMARY_BEST_METRIC_HEADS, skipping NaN heads.

    Returns (sum, n_contributing_heads). Callers should treat n=0 as a
    degenerate epoch (no primary head returned a real AUROC) and NOT save
    best.pt against it — see the eval block below.
    """
    score = 0.0
    n_contrib = 0
    for h in PRIMARY_BEST_METRIC_HEADS:
        v = metrics.get(f"{h}_auroc", float("nan"))
        if not math.isnan(v):
            score += v
            n_contrib += 1
    return score, n_contrib


# ============================================================
# Train loop with NaN guard + early stopping
# ============================================================

metrics_log: list[dict] = []
best_score = -1.0
best_epoch = -1
no_improve = 0
step = 0
t0 = time.time()
model.train()

from torch.cuda.amp import autocast as cuda_autocast  # noqa: E402  # v2 lesson §14.3

for epoch in range(HP["epochs"]):
    for i, batch in enumerate(train_loader):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        with cuda_autocast(enabled=(DEVICE == "cuda"), dtype=torch.bfloat16):
            out = model(
                batch["input_ids"].to(DEVICE),
                batch["token_types"].to(DEVICE),
                batch["day_index"].to(DEVICE),
                batch["wearable_feats"].to(DEVICE),
                batch["confounders"].to(DEVICE),
                batch["attn_mask"].to(DEVICE),
            )
            labels_d = {k: v.to(DEVICE) for k, v in batch["labels"].items()}
            masks_d = {k: v.to(DEVICE) for k, v in batch["masks"].items()}
            loss, per_head = multihead_masked_bce_loss(
                out, labels_d, masks_d, POS_WEIGHTS, HEAD_WEIGHTS,
            )
            loss = loss / HP["grad_accum"]

        # v2 lesson §14.6: hard-halt on NaN
        if not torch.isfinite(loss).item():
            print(f"NaN loss at step {step}; halting", flush=True)
            raise RuntimeError("NaN loss")

        loss.backward()

        if (i + 1) % HP["grad_accum"] == 0:
            nn.utils.clip_grad_norm_(model.parameters(), HP["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % 50 == 0:
                el = (time.time() - t0) / 60
                print(f"e{epoch} step {step}/{total_steps}  loss={loss.item()*HP['grad_accum']:.4f}"
                      f"  lr={lr_at(step):.2e}  elapsed={el:.1f}min", flush=True)
            step += 1

    # End-of-epoch eval + checkpoint
    print(f"--- evaluating epoch {epoch} ---", flush=True)
    val_m = evaluate(val_loader)
    score, n_contrib = sum_primary_auroc(val_m)
    summary_lines = []
    for name in HEAD_NAMES:
        au = val_m.get(f"{name}_auroc", float("nan"))
        if not math.isnan(au):
            summary_lines.append(f"{name}:AUROC={au:.4f}")
    print(f"=== epoch {epoch}  sum_primary_auroc={score:.4f}  (n_heads_contributing={n_contrib}/{len(PRIMARY_BEST_METRIC_HEADS)})  ===", flush=True)
    print("  " + "  ".join(summary_lines), flush=True)
    metrics_log.append({"epoch": epoch, "step": step,
                        "sum_primary_auroc": score,
                        "n_primary_heads_contributing": n_contrib, **val_m})
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    # last.pt every epoch (cheap insurance against crash mid-run)
    torch.save(model.state_dict(), OUT_DIR / "last.pt")

    # Refuse to save best.pt if no primary head contributed a real AUROC.
    # Adversarial reviewer's failure mode: tiny val + rare endpoint → all primary
    # AUROCs NaN → score=0.0 every epoch → epoch-0 random-init heads saved as best.
    if n_contrib == 0:
        print(f"  REFUSING to save best.pt: 0 primary heads contributed AUROC. "
              f"score=0.0 is a degenerate metric, not a real improvement.", flush=True)
    elif score > best_score:
        best_score = score
        best_epoch = epoch
        no_improve = 0
        torch.save(model.state_dict(), OUT_DIR / "best.pt")
        with open(OUT_DIR / "config.json", "w") as f:
            cfg_dict = cfg.__dict__.copy()
            # tuple → list for JSON
            cfg_dict["head_specs"] = [list(s) for s in cfg_dict["head_specs"]]
            json.dump({
                "config": cfg_dict, "hp": HP, "val_at_save": val_m,
                "head_names": list(model.head_names),
                "best_epoch": best_epoch, "best_score": best_score,
            }, f, indent=2)
        print(f"  saved best.pt  ({HP['best_metric']}={best_score:.4f})", flush=True)
    else:
        no_improve += 1
        if no_improve >= HP["early_stop_patience"]:
            print(f"early stop: no improvement for {HP['early_stop_patience']} epochs", flush=True)
            break

print(f"training complete  best_epoch={best_epoch}  best_score={best_score:.4f}", flush=True)
