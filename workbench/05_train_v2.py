"""PhoneFM v2 trainer — multi-head BCE, per-endpoint val AUROC/AUPRC.

Run on AoU Workbench A100 from the workbench directory:
    nohup python -u 05_train_v2.py > /tmp/train_v2.log 2>&1 &

Reads tokenized shards from ~/workspace/phonefm-data/tokenized_v2/{train,val,test}_*.parquet
(produced by 02_tokenizer_v2.py).

Saves checkpoint + metrics to ~/workspace/phonefm-data/phonefm_v2/ which is GCS-FUSE backed
so survives machine-type changes.
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
from phonefm_dataset_v2 import (   # type: ignore[import-not-found]  # noqa: E402
    ENDPOINTS,
    make_loader_v2,
    report_label_distribution,
)
from phonefm_model_v2 import (     # type: ignore[import-not-found]  # noqa: E402
    PhoneFMV2,
    PhoneFMV2Config,
    multihead_bce_loss,
)


# ============================================================
# CELL 1 — config
# ============================================================

HP: dict[str, Any] = dict(
    # backbone
    d_model=384, n_layers=6, n_heads=6, d_ff=1536, dropout=0.1, max_seq_len=512,
    # optimisation
    batch_size=32, grad_accum=2,             # effective batch 64
    lr=3e-4, weight_decay=0.1, warmup_steps=500, epochs=10, grad_clip=1.0,
    # data
    balance_on="composite",
    # which head to use for "best" tracking; sum of endpoint AUROCs is more
    # discriminating than composite alone since multi-head training may differ
    # per endpoint
    best_metric="afib_auroc",
    seed=20260609,
)

DATA_DIR = Path("/home/jupyter/workspace/phonefm-data/tokenized_v2")
OUT_DIR = Path("/home/jupyter/workspace/phonefm-data/phonefm_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(HP["seed"]); np.random.seed(HP["seed"])

print(f"device={DEVICE}  out={OUT_DIR}", flush=True)


# ============================================================
# CELL 2 — data
# ============================================================

print("loading train/val/test...", flush=True)
train_loader = make_loader_v2(
    str(DATA_DIR / "train_*.parquet"),
    batch_size=HP["batch_size"], num_workers=4, balance_on=HP["balance_on"],
    max_seq_len=HP["max_seq_len"],
)
val_loader = make_loader_v2(
    str(DATA_DIR / "val_*.parquet"),
    batch_size=HP["batch_size"], num_workers=2, balance_on=None, shuffle=False,
    max_seq_len=HP["max_seq_len"],
)

# class-balance weights for each head (from train labels, no sampler)
train_stats = report_label_distribution(str(DATA_DIR / "train_*.parquet"))
print("train label distribution:")
for ep, s in train_stats.items():
    print(f"  {ep:<12s} n={s['n']:>7,d}  n_pos={s['n_pos']:>6,d}"
          f"  pos_rate={s['pos_rate']:.4f}  pos_weight={s['pos_weight_for_bce']:.2f}")

POS_WEIGHTS = {ep: train_stats[ep]["pos_weight_for_bce"] for ep in ENDPOINTS}
# Drop any endpoint with zero training positives from the loss (head_weight=0).
# Training such a head would push it toward all-negative, wasting capacity without
# learning anything useful. The head still computes a logit at eval time but does
# not contribute to gradients.
HEAD_WEIGHTS = {ep: (1.0 if train_stats[ep]["n_pos"] > 0 else 0.0) for ep in ENDPOINTS}
_dropped = [ep for ep, w in HEAD_WEIGHTS.items() if w == 0.0]
if _dropped:
    print(f"WARNING: dropping heads {_dropped} from loss (zero positives in train)", flush=True)


# ============================================================
# CELL 3 — model + optimiser
# ============================================================

cfg = PhoneFMV2Config(
    d_model=HP["d_model"], n_layers=HP["n_layers"], n_heads=HP["n_heads"],
    d_ff=HP["d_ff"], dropout=HP["dropout"], max_seq_len=HP["max_seq_len"],
)
model = PhoneFMV2(cfg).to(DEVICE)
print(f"model: {model.num_params()/1e6:.1f}M params  heads={model.head_names}", flush=True)

# AdamW with decoupled weight decay; no decay on bias/LayerNorm
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

# Linear warmup + cosine decay
total_steps = HP["epochs"] * math.ceil(len(train_loader) / HP["grad_accum"])
def lr_at(step: int) -> float:
    if step < HP["warmup_steps"]:
        return HP["lr"] * step / max(1, HP["warmup_steps"])
    progress = (step - HP["warmup_steps"]) / max(1, total_steps - HP["warmup_steps"])
    return HP["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

print(f"total_optimiser_steps={total_steps:,}", flush=True)


# ============================================================
# CELL 4 — eval helper
# ============================================================

@torch.no_grad()
def evaluate(loader) -> dict:
    model.eval()
    preds = {ep: [] for ep in ENDPOINTS}
    labels = {ep: [] for ep in ENDPOINTS}
    for batch in loader:
        out = model(
            batch["input_ids"].to(DEVICE),
            batch["token_types"].to(DEVICE),
            batch["day_index"].to(DEVICE),
            batch["wearable_feats"].to(DEVICE),
            batch["confounders"].to(DEVICE),
            batch["attn_mask"].to(DEVICE),
        )
        for ep, logit in out.items():
            preds[ep].append(torch.sigmoid(logit).cpu().numpy())
            labels[ep].append(batch["labels"][ep].numpy())
    metrics = {}
    for ep in ENDPOINTS:
        p = np.concatenate(preds[ep])
        y = np.concatenate(labels[ep])
        if y.sum() < 5 or y.sum() > len(y) - 5:
            metrics[f"{ep}_auroc"] = float("nan")
            metrics[f"{ep}_auprc"] = float("nan")
            continue
        metrics[f"{ep}_auroc"] = float(roc_auc_score(y, p))
        metrics[f"{ep}_auprc"] = float(average_precision_score(y, p))
        metrics[f"{ep}_pos_rate"] = float(y.mean())
        metrics[f"{ep}_mean_pred"] = float(p.mean())
    model.train()
    return metrics


# ============================================================
# CELL 5 — train loop
# ============================================================

metrics_log = []
best_score = -1.0
step = 0
t0 = time.time()
model.train()

# bf16 on A100: GradScaler is unnecessary (bf16 has fp32 exponent range), but harmless.
# Using torch.cuda.amp for compatibility with torch 2.0.1 on the AoU Jupyter image.
from torch.cuda.amp import autocast as cuda_autocast  # noqa: E402

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
            labels = {k: v.to(DEVICE) for k, v in batch["labels"].items()}
            loss = multihead_bce_loss(out, labels, POS_WEIGHTS, HEAD_WEIGHTS) / HP["grad_accum"]

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

    # end-of-epoch val
    print(f"--- evaluating epoch {epoch} ---", flush=True)
    val_m = evaluate(val_loader)
    summary = "  ".join(
        f"{ep}:AUROC={val_m.get(ep+'_auroc', float('nan')):.4f}/AUPRC={val_m.get(ep+'_auprc', float('nan')):.4f}"
        for ep in ENDPOINTS if not math.isnan(val_m.get(ep + "_auroc", float("nan")))
    )
    print(f"=== epoch {epoch}: {summary} ===", flush=True)
    metrics_log.append({"epoch": epoch, "step": step, **val_m})

    score = val_m.get(HP["best_metric"], float("-inf"))
    if not math.isnan(score) and score > best_score:
        best_score = score
        torch.save(model.state_dict(), OUT_DIR / "best.pt")
        with open(OUT_DIR / "config.json", "w") as f:
            json.dump({"config": cfg.__dict__, "hp": HP, "val": val_m,
                       "head_names": model.head_names}, f, indent=2)
        print(f"  saved best.pt  ({HP['best_metric']}={best_score:.4f})", flush=True)

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

print("training complete", flush=True)
