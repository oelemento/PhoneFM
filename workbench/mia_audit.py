"""
mia_audit.py — SKELETON: loss-based membership-inference audit for the PhoneFM v3
checkpoint, to support the AoU/Verily Controlled-Tier egress request.

Idea: if the model memorizes training participants it assigns them systematically
lower loss than comparable held-out (test) windows. We compute per-window loss
for a balanced sample of TRAIN (members) and TEST (non-members) windows and
report the AUROC of (-loss -> member). ~0.50 = no detectable memorization.

>>> STATUS: SKELETON. Needs (1) code review, (2) one GPU pass, and (3) the loss
    reconstruction matched to the EXACT training objective before the number is
    trustworthy. See docs/membership_inference_audit_plan.md.

TODOs flagged inline:
  - confirm the TRAIN parquet glob (train_*.parquet?) and that it is the SAME
    split the checkpoint was trained on;
  - confirm the per-head loss matches training (masked BCE, head weighting,
    pos_weight / class balancing, label smoothing, etc.).

Run (GPU pod):  cd ~/repos/PhoneFM/workbench && python3 -u mia_audit.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))
from phonefm_dataset_v3 import make_loader_v3
from phonefm_model_v3 import PhoneFMV3, PhoneFMV3Config

DATA_DIR = Path("/home/jupyter/workspace/phonefm-data/tokenized_v3")
MODEL_DIR = Path("/home/jupyter/workspace/phonefm-data/phonefm_v3")
OUT_PATH = MODEL_DIR / "mia_audit.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
SEED = 20260612
N_PER_GROUP = 8000          # members (train) and non-members (test) to sample
HEADS = ("cv_composite_30d", "cv_composite_180d", "cv_composite_365d",
         "afib_30d", "mortality_30d")   # TODO: use the model's actual head set

# TODO confirm: the split the model was TRAINED on.
TRAIN_GLOB = str(DATA_DIR / "train_*.parquet")
TEST_GLOB = str(DATA_DIR / "test_*.parquet")


def per_window_loss(model, batch) -> np.ndarray:
    """Sum of masked per-head BCE for each window in the batch -> (B,) array.
    TODO: match this to the EXACT training loss (head weights, pos_weight,
    smoothing). The membership SIGNAL is robust to mild mismatch, but the
    reported loss-gap magnitude is not — fix before quoting the gap."""
    input_ids = batch["input_ids"].to(DEVICE)
    token_types = batch["token_types"].to(DEVICE)
    day_index = batch["day_index"].to(DEVICE)
    wearable = batch["wearable_feats"].to(DEVICE)
    confounders = batch["confounders"].to(DEVICE)
    attn_mask = batch["attn_mask"].to(DEVICE)
    logits = model(input_ids, token_types, day_index, wearable, confounders, attn_mask)
    B = input_ids.shape[0]
    total = torch.zeros(B, device=DEVICE)
    count = torch.zeros(B, device=DEVICE)
    for h in HEADS:
        if h not in logits:
            continue
        y = batch["labels"][h].to(DEVICE).float()
        m = batch["masks"][h].to(DEVICE).bool()
        lh = F.binary_cross_entropy_with_logits(
            logits[h].float().reshape(-1), y.reshape(-1), reduction="none")
        lh = lh * m.reshape(-1).float()
        total += lh
        count += m.reshape(-1).float()
    return (total / count.clamp(min=1)).detach().cpu().numpy()


@torch.no_grad()
def collect(model, glob_pattern, n_target, cfg) -> np.ndarray:
    torch.manual_seed(SEED)
    loader = make_loader_v3(glob_pattern, batch_size=BATCH_SIZE, num_workers=4,
                            shuffle=True, max_seq_len=cfg.max_seq_len,
                            n_confounders=cfg.n_confounders)
    out = []
    for batch in loader:
        out.append(per_window_loss(model, batch))
        if sum(len(x) for x in out) >= n_target:
            break
    return np.concatenate(out)[:n_target]


@torch.no_grad()
def main():
    print(f"device={DEVICE}", flush=True)
    with open(MODEL_DIR / "config.json") as f:
        saved = json.load(f)
    cfg_d = saved["config"].copy()
    cfg_d["head_specs"] = tuple(tuple(s) for s in cfg_d["head_specs"])
    cfg = PhoneFMV3Config(**cfg_d)
    model = PhoneFMV3(cfg).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location=DEVICE), strict=True)
    model.eval()

    members = collect(model, TRAIN_GLOB, N_PER_GROUP, cfg)      # train = in
    nonmembers = collect(model, TEST_GLOB, N_PER_GROUP, cfg)    # test  = out
    print(f"members={len(members)} non-members={len(nonmembers)}", flush=True)

    loss = np.concatenate([members, nonmembers])
    is_member = np.concatenate([np.ones(len(members)), np.zeros(len(nonmembers))])
    # attacker predicts membership from LOW loss -> score = -loss
    mia_auroc = float(roc_auc_score(is_member, -loss))
    gap = float(members.mean() - nonmembers.mean())   # negative => members lower loss

    print(f"\nMIA AUROC (-loss -> member): {mia_auroc:.4f}   (0.50 = no memorization)")
    print(f"mean loss  members={members.mean():.4f}  non-members={nonmembers.mean():.4f}")
    print(f"train-test loss gap: {gap:+.4f}")
    verdict = "no detectable memorization" if abs(mia_auroc - 0.5) < 0.03 else "INVESTIGATE"
    print(f"verdict: {verdict}")

    with open(OUT_PATH, "w") as f:
        json.dump({"mia_auroc": mia_auroc, "loss_gap_train_minus_test": gap,
                   "n_per_group": int(N_PER_GROUP), "heads": list(HEADS),
                   "verdict": verdict}, f, indent=2)
    print(f"saved {OUT_PATH}")
    print("\nNOTE: SKELETON — match per_window_loss() to the training objective and "
          "review before quoting numbers; consider a calendar/utilization-matched "
          "control set to rule out train/test drift (see audit plan).")


if __name__ == "__main__":
    main()
