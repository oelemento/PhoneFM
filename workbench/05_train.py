"""PhoneFM training loop — runs inside Workbench on a single A100 / L4.

Run order: 01_cohort_extraction → 02_tokenizer → (this).
Defaults sized for 1 GPU, ~12 hours. To resize: bump batch_size + grad_accum
if you have more VRAM, or shrink d_model to 384 for a faster prototype.

Outputs:
  /tmp/phonefm_v1/best.pt           ← state_dict
  /tmp/phonefm_v1/config.json
  /tmp/phonefm_v1/metrics.json      ← per-epoch val AUROC
  gs://$WORKSPACE_BUCKET/phonefm/checkpoints/v1/ ← synced
"""

# ============================================================
# CELL 1 — imports + workspace
# ============================================================
import os, json, time, math
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, average_precision_score

# These two files were just authored — if you saved them as 03_dataset.py /
# 04_model.py in Jupyter, rename to phonefm_dataset.py / phonefm_model.py first
# (module names can't start with a digit). The cells below assume rename.
from phonefm_dataset import make_loader, VOCAB_SIZE, PAD_ID
from phonefm_model import PhoneFM, PhoneFMConfig

OUT_DIR = Path("/tmp/phonefm_v1"); OUT_DIR.mkdir(exist_ok=True)
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}  vocab_size: {VOCAB_SIZE}")


# ============================================================
# CELL 2 — hyperparams
# ============================================================
HP = dict(
    d_model      = 512,
    n_layers     = 12,
    n_heads      = 8,
    d_ff         = 2048,
    dropout      = 0.1,
    max_seq_len  = 4096,

    batch_size   = 16,
    grad_accum   = 4,           # effective batch = 64
    lr           = 3e-4,
    weight_decay = 0.1,
    warmup_steps = 500,
    epochs       = 6,
    grad_clip    = 1.0,
    pos_loss_weight = None,     # set later from train pos-rate
    seed         = 20260606,
)
torch.manual_seed(HP["seed"]); np.random.seed(HP["seed"])


# ============================================================
# CELL 3 — model + optimizer
# ============================================================
cfg = PhoneFMConfig(
    vocab_size=VOCAB_SIZE,
    d_model=HP["d_model"], n_layers=HP["n_layers"], n_heads=HP["n_heads"],
    d_ff=HP["d_ff"], dropout=HP["dropout"], max_seq_len=HP["max_seq_len"],
    pad_id=PAD_ID,
)
model = PhoneFM(cfg).to(DEVICE)
print(f"model: {model.num_params()/1e6:.1f}M params")

# AdamW with decoupled weight decay; no decay on bias/LayerNorm
decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad: continue
    (no_decay if p.ndim < 2 or n.endswith(".bias") else decay).append(p)
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": HP["weight_decay"]},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=HP["lr"], betas=(0.9, 0.95),
)
# bf16 has fp32 dynamic range and does NOT need loss scaling — keep GradScaler
# disabled so .scale/.unscale_/.step/.update are all no-ops. Re-enable only if
# switching autocast dtype to fp16.
scaler = GradScaler(enabled=False)


# ============================================================
# CELL 4 — data
# ============================================================
train_dl = make_loader("train", batch_size=HP["batch_size"], num_workers=4, balance=True)
val_dl   = make_loader("val",   batch_size=HP["batch_size"], num_workers=2, shuffle=False, balance=False)

# Class balance is already handled by WeightedRandomSampler in 03_dataset.py
# (positives upsampled to ~50% per batch). Adding BCE pos_weight on top would
# double-correct by ~((1-p)/p)^2 — observed via inflated positive predictions
# and broken calibration. Use pos_weight=1.0 when balanced sampling is on.
sample_labels = np.array([train_dl.dataset.frames[si].iloc[ri]["label"]
                          for si, ri in train_dl.dataset.index[:5000]])
pos_rate = float(sample_labels.mean())
HP["pos_loss_weight"] = 1.0
print(f"raw train pos_rate (pre-sampler) = {pos_rate:.4f}  "
      f"BCE pos_weight = {HP['pos_loss_weight']:.2f}  "
      f"(class balance handled by WeightedRandomSampler)")

loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(HP["pos_loss_weight"], device=DEVICE))


# ============================================================
# CELL 5 — LR schedule (linear warmup + cosine decay)
# ============================================================
total_steps = HP["epochs"] * math.ceil(len(train_dl) / HP["grad_accum"])
def lr_at(step):
    if step < HP["warmup_steps"]:
        return HP["lr"] * step / max(1, HP["warmup_steps"])
    progress = (step - HP["warmup_steps"]) / max(1, total_steps - HP["warmup_steps"])
    return HP["lr"] * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


# ============================================================
# CELL 6 — eval helper
# ============================================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    logits, labels = [], []
    for batch in loader:
        b = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
        with autocast(enabled=(DEVICE == "cuda"), dtype=torch.bfloat16):
            out = model(b["input_ids"], b["positions"], b["attn_mask"])
        logits.append(out.float().cpu().numpy())
        labels.append(b["label"].cpu().numpy())
    model.train()
    logits = np.concatenate(logits); labels = np.concatenate(labels)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return {
        "auroc": float(roc_auc_score(labels, logits)),
        "auprc": float(average_precision_score(labels, logits)),
        "pos_rate": float(labels.mean()),
        "mean_pred_prob": float(probs.mean()),  # should ≈ pos_rate if calibrated
        "n": int(len(labels)),
    }


# ============================================================
# CELL 7 — train
# ============================================================
metrics_log = []
best_auroc = 0.0
step = 0
model.train()
t0 = time.time()

for epoch in range(HP["epochs"]):
    opt.zero_grad(set_to_none=True)
    epoch_loss = 0.0; n_seen = 0
    for i, batch in enumerate(train_dl):
        b = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
        ctx = autocast(enabled=(DEVICE == "cuda"), dtype=torch.bfloat16) if DEVICE == "cuda" else nullcontext()
        with ctx:
            logit = model(b["input_ids"], b["positions"], b["attn_mask"])
            loss = loss_fn(logit, b["label"]) / HP["grad_accum"]

        scaler.scale(loss).backward()
        epoch_loss += loss.item() * HP["grad_accum"]; n_seen += b["label"].numel()

        if (i + 1) % HP["grad_accum"] == 0:
            for g in opt.param_groups: g["lr"] = lr_at(step)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), HP["grad_clip"])
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"e{epoch} step {step}/{total_steps}  "
                      f"loss={epoch_loss/n_seen:.4f}  lr={lr_at(step):.2e}  "
                      f"elapsed={(time.time()-t0)/60:.1f}min")

    val = evaluate(model, val_dl)
    # Calibration sanity: mean predicted prob should be close to val pos_rate.
    # If it's wildly off (e.g., 0.5 when pos_rate=0.05), class-balance handling
    # is double-correcting — see C5 in the pre-deploy review.
    cal_gap = abs(val["mean_pred_prob"] - val["pos_rate"])
    print(f"=== epoch {epoch}: val AUROC={val['auroc']:.4f}  AUPRC={val['auprc']:.4f} "
          f"(pos_rate={val['pos_rate']:.4f}, mean_pred={val['mean_pred_prob']:.4f}, "
          f"cal_gap={cal_gap:.3f}, n={val['n']:,}) ===")
    if cal_gap > 0.15:
        print(f"  ⚠ calibration gap > 0.15 — class-balance handling may be wrong")
    metrics_log.append({"epoch": epoch, "step": step, **val,
                        "train_loss": epoch_loss / max(n_seen, 1)})
    if val["auroc"] > best_auroc:
        best_auroc = val["auroc"]
        torch.save(model.state_dict(), OUT_DIR / "best.pt")
        json.dump({"config": cfg.__dict__, "hp": HP, "val": val},
                  open(OUT_DIR / "config.json", "w"), indent=2)
        print(f"  ↳ saved best.pt  ({best_auroc:.4f})")

    json.dump(metrics_log, open(OUT_DIR / "metrics.json", "w"), indent=2)

print(f"DONE. best val AUROC = {best_auroc:.4f}  total time = {(time.time()-t0)/3600:.2f}h")


# ============================================================
# CELL 8 — final test set eval + sync to bucket
# ============================================================
model.load_state_dict(torch.load(OUT_DIR / "best.pt"))
test_dl = make_loader("test", batch_size=HP["batch_size"], num_workers=2, shuffle=False, balance=False)
test_metrics = evaluate(model, test_dl)
print(f"TEST: AUROC={test_metrics['auroc']:.4f}  AUPRC={test_metrics['auprc']:.4f}  n={test_metrics['n']:,}")
json.dump({**test_metrics, "split": "test"},
          open(OUT_DIR / "test_metrics.json", "w"), indent=2)

# Sync to workspace bucket so 06_eval_mia.py + the export request can pick it up
BUCKET = os.environ["WORKSPACE_BUCKET"]
from google.cloud import storage  # noqa: E402
gcs = storage.Client(); bkt = gcs.bucket(BUCKET.replace("gs://", ""))
for p in OUT_DIR.glob("*"):
    bkt.blob(f"phonefm/checkpoints/v1/{p.name}").upload_from_filename(str(p))
print(f"Checkpoint synced to {BUCKET}/phonefm/checkpoints/v1/")
