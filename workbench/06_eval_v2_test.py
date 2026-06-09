"""PhoneFM v2 test-set evaluation with cluster bootstrap 95% CIs.

Per the v3 spec §13.2 — this is the decision gate before launching v3:
  - If v2 test AFib AUROC lower 95% CI >= 0.85 → proceed to v3
  - If < 0.85 → pause and diagnose generalization gap

Loads /home/jupyter/workspace/phonefm-data/phonefm_v2/best.pt (saved at
epoch 2 with afib_auroc=0.9285), runs on test_*.parquet shards (13 shards,
~64K windows from ~1.8K participants), and reports per-endpoint AUROC +
AUPRC + cluster-bootstrap 95% CI.

Cluster bootstrap is mandatory: each participant contributes ~5 windows
that are highly correlated (same demographics, baseline conditions).
Window-level bootstrap would underestimate CIs by ~sqrt(5) = 2.2×,
inflating the chance of spuriously passing the 0.85 gate.

Output: /home/jupyter/workspace/phonefm-data/phonefm_v2/test_results.json

Run on AoU Workbench A100 (or any GPU machine after v2 training finishes):
    python3 06_eval_v2_test.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-not-found]
import torch
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-not-found]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phonefm_dataset_v2 import (  # type: ignore[import-not-found]
    ENDPOINTS,
    make_loader_v2,
)
from phonefm_model_v2 import (  # type: ignore[import-not-found]
    PhoneFMV2,
    PhoneFMV2Config,
)


# ============================================================
# Config
# ============================================================

DATA_DIR = Path("/home/jupyter/workspace/phonefm-data/tokenized_v2")
MODEL_DIR = Path("/home/jupyter/workspace/phonefm-data/phonefm_v2")
OUT_PATH = MODEL_DIR / "test_results.json"

BATCH_SIZE = 32
# num_workers=0 is required so DataLoader returns batches in deterministic
# shard order, allowing per-row person_id lookup from pre-scanned shards.
NUM_WORKERS = 0

# Decision gate from v3 spec §13.2
AFIB_DECISION_THRESHOLD = 0.85
BOOTSTRAP_N = 1000
CI_LEVEL = 0.95

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEVICE}  best_pt={MODEL_DIR / 'best.pt'}", flush=True)


# ============================================================
# Cluster bootstrap (resample person_ids, not rows)
# ============================================================

def cluster_bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    person_ids: np.ndarray,
    metric_fn,
    n_boot: int = BOOTSTRAP_N,
    ci: float = CI_LEVEL,
    rng_seed: int = 20260609,
) -> dict:
    """Cluster bootstrap a metric (AUROC or AUPRC) at the person level.

    Standard window-level i.i.d. bootstrap underestimates variance because
    a participant's multiple windows are highly correlated. Cluster
    bootstrap samples PERSON IDs with replacement and aggregates each
    sampled person's full set of windows.

    Returns dict with point, lo, hi, n_valid (number of resamples where
    the metric was computable — degenerate resamples with <5 positives
    are dropped).
    """
    rng = np.random.RandomState(rng_seed)
    point = metric_fn(y_true, y_pred)

    # Map person_id → list of row indices
    unique_pids = np.unique(person_ids)
    pid_to_rows: dict = {}
    for pid in unique_pids:
        pid_to_rows[pid] = np.where(person_ids == pid)[0]

    scores = []
    n_pid = len(unique_pids)
    for _ in range(n_boot):
        # Resample person_ids with replacement
        sampled_pids = unique_pids[rng.randint(0, n_pid, size=n_pid)]
        # Concatenate all rows from sampled persons
        rows = np.concatenate([pid_to_rows[pid] for pid in sampled_pids])
        yt = y_true[rows]
        yp = y_pred[rows]
        if yt.sum() < 5 or yt.sum() > len(yt) - 5:
            continue
        scores.append(metric_fn(yt, yp))

    if len(scores) < 100:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan"),
                "n_valid": len(scores)}
    lo = float(np.quantile(scores, (1 - ci) / 2))
    hi = float(np.quantile(scores, 1 - (1 - ci) / 2))
    return {"point": float(point), "lo": lo, "hi": hi, "n_valid": len(scores)}


# ============================================================
# Main
# ============================================================

@torch.no_grad()
def main() -> None:
    if not (MODEL_DIR / "best.pt").exists():
        raise FileNotFoundError(f"Missing {MODEL_DIR / 'best.pt'} — v2 training must finish first")
    if not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(f"Missing {MODEL_DIR / 'config.json'}")

    # Load v2 config — derive MAX_SEQ_LEN from there, NOT a hardcoded constant
    with open(MODEL_DIR / "config.json") as f:
        saved = json.load(f)
    cfg_d = saved["config"]
    cfg = PhoneFMV2Config(**cfg_d)
    max_seq_len = cfg.max_seq_len
    print(f"loaded config: d_model={cfg.d_model} n_layers={cfg.n_layers} "
          f"max_seq_len={max_seq_len}", flush=True)
    print(f"val metrics at save time: {saved.get('val', {})}", flush=True)
    print(f"head_names: {saved.get('head_names', '?')}", flush=True)

    # Pre-scan test shards to collect person_id per row (in shard order).
    # This must match the order the loader will visit rows. PhoneFMV2Dataset
    # reads shards in sorted glob order, then iloc rows 0..N-1 per shard.
    print("\npre-scanning test shards for person_ids...", flush=True)
    test_glob = str(DATA_DIR / "test_*.parquet")
    person_ids_per_row: list = []
    # Per-shard boundary tracking for the cross-check below
    shard_first_pids: list = []
    shard_row_counts: list = []
    for path in sorted(glob.glob(test_glob)):
        df = pd.read_parquet(path, columns=["person_id"])
        pids = df["person_id"].tolist()
        shard_first_pids.append(pids[0] if pids else None)
        shard_row_counts.append(len(pids))
        person_ids_per_row.extend(pids)
    person_ids_array = np.array(person_ids_per_row, dtype=np.int64)
    n_pid = len(np.unique(person_ids_array))
    print(f"  n_rows = {len(person_ids_array):,}   n_unique_persons = {n_pid:,}", flush=True)

    # Build model and load weights with strict=True (no silent random-head fallback)
    model = PhoneFMV2(cfg).to(DEVICE)
    state = torch.load(MODEL_DIR / "best.pt", map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"model loaded (strict=True): {model.num_params() / 1e6:.1f}M params", flush=True)

    # BatchNorm sanity check — running stats should not be NaN/inf
    bn = model.wearable_input_norm  # type: ignore[attr-defined]
    if bn.running_mean is None or bn.running_var is None:
        raise RuntimeError("wearable_input_norm has no running stats — model was loaded incorrectly")
    if not torch.isfinite(bn.running_mean).all() or not torch.isfinite(bn.running_var).all():
        raise RuntimeError("wearable_input_norm running stats contain NaN/inf")
    print(f"  BN running_mean (first 3 dims) = {bn.running_mean[:3].tolist()}", flush=True)
    print(f"  BN running_var  (first 3 dims) = {bn.running_var[:3].tolist()}", flush=True)

    # Test loader — shuffle=False with num_workers=0 for deterministic shard order
    test_loader = make_loader_v2(
        test_glob,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, balance_on=None,
        shuffle=False, max_seq_len=max_seq_len,
    )
    n_dataset = len(test_loader.dataset)
    print(f"\ntest_loader: {len(test_loader)} batches, {n_dataset:,} windows total", flush=True)
    if n_dataset != len(person_ids_array):
        raise RuntimeError(
            f"person_id count {len(person_ids_array)} != dataset size {n_dataset}; "
            f"test shards changed since pre-scan?"
        )
    # Per-shard first-row sanity check — guards against any future pyarrow/engine
    # change that reorders rows on read. We verify the dataset's row at each
    # shard's starting index matches the person_id we recorded during pre-scan.
    cum = 0
    for shard_idx, (first_pid, n_rows) in enumerate(zip(shard_first_pids, shard_row_counts)):
        if first_pid is None:
            cum += n_rows
            continue
        ds_first_pid = int(test_loader.dataset.frames[shard_idx].iloc[0]["person_id"])
        if ds_first_pid != first_pid:
            raise RuntimeError(
                f"shard {shard_idx} pre-scan first person_id {first_pid} != "
                f"dataset.frames[{shard_idx}].iloc[0].person_id {ds_first_pid}. "
                f"parquet row-order differs between pre-scan and dataset read — "
                f"cluster bootstrap would mis-assign person_ids. ABORT."
            )
        cum += n_rows
    print(f"  per-shard first-row person_id parity verified across {len(shard_first_pids)} shards  (OK)", flush=True)

    # Forward pass — collect predictions per endpoint, preserving row order
    preds: dict[str, list] = {ep: [] for ep in ENDPOINTS}
    labels: dict[str, list] = {ep: [] for ep in ENDPOINTS}

    t0 = time.time()
    for i, batch in enumerate(test_loader):
        # Match training-time eval semantics: bf16 autocast
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda"), dtype=torch.bfloat16):
            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["token_types"].to(DEVICE),
                batch["day_index"].to(DEVICE),
                batch["wearable_feats"].to(DEVICE),
                batch["confounders"].to(DEVICE),
                batch["attn_mask"].to(DEVICE),
            )
        for ep in ENDPOINTS:
            if ep in logits:
                preds[ep].append(torch.sigmoid(logits[ep].float()).cpu().numpy())
                labels[ep].append(batch["labels"][ep].numpy())
        if i % 100 == 0:
            el = (time.time() - t0) / 60
            print(f"  batch {i}/{len(test_loader)}  elapsed={el:.1f}min", flush=True)
    el = (time.time() - t0) / 60
    print(f"forward done in {el:.1f} min", flush=True)

    # Per-endpoint metrics with cluster bootstrap
    results: dict[str, dict] = {}
    print("\n=== v2 TEST RESULTS (cluster-bootstrap CIs, person-level) ===", flush=True)
    print(f"{'endpoint':<12} {'n':>7} {'n_pos':>6} {'rate':>7} "
          f"{'AUROC (95% CI)':<32} {'AUPRC (95% CI)':<32} {'n_valid':>7}", flush=True)
    for ep in ENDPOINTS:
        if not preds[ep]:
            print(f"{ep:<12}  no predictions (head missing from model)", flush=True)
            results[ep] = {"error": "head missing"}
            continue
        y_pred = np.concatenate(preds[ep])
        y_true = np.concatenate(labels[ep]).astype(np.int64)
        n = len(y_true)
        n_pos = int(y_true.sum())
        assert n == len(person_ids_array), \
            f"row count mismatch: predictions {n} vs person_ids {len(person_ids_array)}"
        if n_pos < 5 or n_pos > n - 5:
            print(f"{ep:<12} {n:>7,} {n_pos:>6,} {n_pos/n:>7.4f}  "
                  f"insufficient positives for AUROC", flush=True)
            results[ep] = {
                "n": n, "n_pos": n_pos, "pos_rate": n_pos / n,
                "auroc": None, "auprc": None,
                "skipped": "n_pos < 5 or n_pos > n - 5",
            }
            continue
        auroc = cluster_bootstrap_metric(y_true, y_pred, person_ids_array, roc_auc_score)
        auprc = cluster_bootstrap_metric(y_true, y_pred, person_ids_array, average_precision_score)
        print(f"{ep:<12} {n:>7,} {n_pos:>6,} {n_pos/n:>7.4f}  "
              f"{auroc['point']:.4f} ({auroc['lo']:.4f}-{auroc['hi']:.4f})  "
              f"{auprc['point']:.4f} ({auprc['lo']:.4f}-{auprc['hi']:.4f})  "
              f"{auroc['n_valid']:>7,}", flush=True)
        results[ep] = {
            "n": n, "n_pos": n_pos, "pos_rate": n_pos / n,
            "auroc": auroc, "auprc": auprc,
        }

    # Decision gate per v3 spec §13.2 — NaN-aware
    afib_result = results.get("afib", {})
    afib_auroc = afib_result.get("auroc", {}) if isinstance(afib_result.get("auroc"), dict) else {}
    afib_auroc_lo = afib_auroc.get("lo")
    print("\n=== v3 SPEC §13.2 DECISION GATE ===", flush=True)
    if afib_auroc_lo is None or (isinstance(afib_auroc_lo, float) and math.isnan(afib_auroc_lo)):
        print(f"  AFib AUROC lower CI bound is NaN/missing → MANUAL_REVIEW required", flush=True)
        decision = "MANUAL_REVIEW"
    elif afib_auroc_lo >= AFIB_DECISION_THRESHOLD:
        print(f"  AFib AUROC lower 95% CI = {afib_auroc_lo:.4f} >= {AFIB_DECISION_THRESHOLD}  "
              f"→ PROCEED TO v3", flush=True)
        decision = "PROCEED"
    else:
        print(f"  AFib AUROC lower 95% CI = {afib_auroc_lo:.4f} < {AFIB_DECISION_THRESHOLD}  "
              f"→ PAUSE FOR GENERALIZATION DIAGNOSIS", flush=True)
        decision = "PAUSE"

    # Save full results
    final_results = {
        "model_dir": str(MODEL_DIR),
        "test_shards_glob": test_glob,
        "n_test_persons": int(n_pid),
        "n_test_windows": len(person_ids_array),
        "decision": decision,
        "afib_decision_threshold": AFIB_DECISION_THRESHOLD,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_kind": "cluster_at_person_level",
        "ci_level": CI_LEVEL,
        "endpoints": results,
        "val_at_save_time": saved.get("val", {}),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nresults saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
