"""PhoneFM membership-inference audit — required before model-export request.

What we're proving: an attacker holding the trained model weights cannot
reliably tell whether a given participant's data was in the training set.

Methodology (matches All-of-Us export-review expectations):

1. Hold-out attack. Compare per-participant loss on train members vs val/test
   non-members. If the model memorized, members will have systematically lower
   loss → AUROC of (loss → "is_member?") will be high.

2. Shadow-model attack (optional, stronger). Train a small shadow model on a
   disjoint subset, then train a meta-classifier on shadow-member vs shadow-non
   loss features and apply it to the target model. Skipped here unless the
   simple hold-out attack already shows leakage.

Acceptance criteria for the export request:
   - Hold-out attack AUROC  < 0.55  (chance = 0.50)
   - 95th-percentile per-participant loss member-vs-non difference < 0.02

Output:
   /tmp/phonefm_v1/mia_audit.json
"""

# ============================================================
# CELL 1 — load model + data
# ============================================================
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score

from phonefm_dataset import CardioWindowsDataset
from phonefm_model import PhoneFM, PhoneFMConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUN_DIR = Path("/tmp/phonefm_v1")

cfg_dict = json.load(open(RUN_DIR / "config.json"))["config"]
cfg = PhoneFMConfig(**cfg_dict)
model = PhoneFM(cfg).to(DEVICE)
model.load_state_dict(torch.load(RUN_DIR / "best.pt", map_location=DEVICE))
model.eval()


# ============================================================
# CELL 2 — per-participant loss
# ============================================================
@torch.no_grad()
def per_participant_loss(split: str, max_participants: int = 5000):
    ds = CardioWindowsDataset(split)
    # Aggregate loss per person_id (each person has multiple windows)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    by_pid = {}
    for idx in range(len(ds)):
        sample = ds[idx]
        si, ri = ds.index[idx]
        pid = int(ds.frames[si].iloc[ri]["person_id"])
        if pid not in by_pid and len(by_pid) >= max_participants:
            continue
        ids  = sample["input_ids"].unsqueeze(0).to(DEVICE)
        pos  = sample["positions"].unsqueeze(0).to(DEVICE)
        mask = sample["attn_mask"].unsqueeze(0).to(DEVICE)
        y    = sample["label"].unsqueeze(0).to(DEVICE)
        logit = model(ids, pos, mask)
        loss = loss_fn(logit, y).item()
        by_pid.setdefault(pid, []).append(loss)
    return {pid: float(np.mean(v)) for pid, v in by_pid.items()}


print("computing per-participant loss on train members...")
train_loss = per_participant_loss("train", max_participants=5000)
print(f"  n_train_members = {len(train_loss):,}")

print("computing per-participant loss on val non-members...")
val_loss = per_participant_loss("val", max_participants=5000)
print(f"  n_val_nonmembers = {len(val_loss):,}")


# ============================================================
# CELL 3 — hold-out attack: can attacker distinguish members from non-members?
# ============================================================
all_losses = list(train_loss.values()) + list(val_loss.values())
is_member = [1] * len(train_loss) + [0] * len(val_loss)

# Lower loss → predicted "member". So attack uses -loss as the score.
mia_auroc = float(roc_auc_score(is_member, [-l for l in all_losses]))

tl = np.array(list(train_loss.values()))
vl = np.array(list(val_loss.values()))
diff_p95 = float(np.percentile(vl, 95) - np.percentile(tl, 95))
diff_med = float(np.median(vl) - np.median(tl))

print(f"\n=== MIA hold-out attack ===")
print(f"  AUROC (member vs non):     {mia_auroc:.4f}   (chance = 0.5; threshold = 0.55)")
print(f"  median loss difference:    {diff_med:+.4f}")
print(f"  p95   loss difference:     {diff_p95:+.4f}")


# ============================================================
# CELL 4 — decision
# ============================================================
PASS_AUROC = 0.55
PASS_LOSS_DIFF = 0.02

passes = (mia_auroc < PASS_AUROC) and (abs(diff_p95) < PASS_LOSS_DIFF)
print(f"\nVerdict: {'PASS' if passes else 'FAIL'} for All-of-Us export request")
if not passes:
    print("Next steps: add stronger regularization (more dropout, less epochs),")
    print("           sample more participants per batch, or train on a larger cohort.")


# ============================================================
# CELL 5 — write report
# ============================================================
report = {
    "model": "phonefm_v1",
    "checkpoint": str(RUN_DIR / "best.pt"),
    "n_train_members_sampled": len(train_loss),
    "n_val_nonmembers_sampled": len(val_loss),
    "mia_holdout_auroc": mia_auroc,
    "median_loss_diff_nonmember_minus_member": diff_med,
    "p95_loss_diff_nonmember_minus_member": diff_p95,
    "thresholds": {"auroc": PASS_AUROC, "p95_loss_diff": PASS_LOSS_DIFF},
    "passes_export_criteria": passes,
}
out = RUN_DIR / "mia_audit.json"
json.dump(report, open(out, "w"), indent=2)
print(f"\nwrote {out}")


# ============================================================
# CELL 6 — sync to bucket so the export request can reference it
# ============================================================
import os
from google.cloud import storage
BUCKET = os.environ["WORKSPACE_BUCKET"]
gcs = storage.Client(); bkt = gcs.bucket(BUCKET.replace("gs://", ""))
bkt.blob("phonefm/checkpoints/v1/mia_audit.json").upload_from_filename(str(out))
print("Synced. Attach mia_audit.json to the model-export request.")
