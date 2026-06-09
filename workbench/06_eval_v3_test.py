"""PhoneFM v3 test-set evaluation — multi-horizon cluster bootstrap CIs.

For each of the 13 (endpoint, horizon) heads, computes person-level cluster-
bootstrap 95% CIs on AUROC + AUPRC.

Bootstrap RNG matches 06_eval_v2_test.py (np.random.RandomState seed=20260609)
so the CIs are bit-comparable to v2.

Cox PH sensitivity analysis (v3_spec.md §11.3) and subgroup analyses (§13.3)
live in separate scripts (06_cox_sensitivity.py, 06_subgroup_analysis.py) that
load the same best.pt + test_results.json artifacts produced here.

Run on AoU Workbench A100:
    cd ~/repos/PhoneFM/workbench && python3 06_eval_v3_test.py

Output:
    ~/workspace/phonefm-data/phonefm_v3/test_results.json
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-not-found]
import torch
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-not-found]

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))

# (The earlier draft imported v2's cluster_bootstrap_metric via importlib but
# never used it — cluster_bootstrap_masked below subsumes it and supports the
# per-endpoint mask. Removed per correctness-review finding M5.)

from phonefm_dataset_v3 import (  # type: ignore[import-not-found]
    HEAD_NAMES,
    make_loader_v3,
)
from phonefm_model_v3 import (  # type: ignore[import-not-found]
    PRIMARY_HEADS,
    NEGATIVE_CONTROL_HEADS,
    PhoneFMV3,
    PhoneFMV3Config,
    head_name,
)


# ============================================================
# Config — overrides v3 train defaults for eval
# ============================================================

DATA_DIR = Path("/home/jupyter/workspace/phonefm-data/tokenized_v3")
MODEL_DIR = Path("/home/jupyter/workspace/phonefm-data/phonefm_v3")
OUT_PATH = MODEL_DIR / "test_results.json"
SUBGROUP_DEF_PATH = HERE / "subgroup_definitions.json"

BATCH_SIZE = 32
NUM_WORKERS = 0
BOOTSTRAP_N = 1000
CI_LEVEL = 0.95
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Decision gates (mirror v2 §13.2 but per-domain)
DECISION_GATES = {
    "cv_composite_30d_auroc_lo":  0.85,
    "mortality_365d_auroc_lo":    0.70,
    "t2d_365d_auroc_lo":          0.65,
    "dep_365d_auroc_lo":          0.60,
}


# ============================================================
# Helpers
# ============================================================

def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def cluster_bootstrap_masked(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    person_ids: np.ndarray,
    mask: np.ndarray,
    metric_fn,
    n_boot: int = BOOTSTRAP_N,
    ci: float = CI_LEVEL,
) -> dict:
    """Cluster bootstrap that respects the per-head mask.

    Persons whose ALL rows have mask=0 contribute nothing. Resamples persons
    with replacement; for each resample, concatenates only the mask=1 rows of
    the resampled persons.
    """
    if not mask.any():
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_valid": 0}
    rows_buf: dict[int, list[int]] = {}
    valid_idx = np.where(mask)[0]
    for ri in valid_idx:
        pid = int(person_ids[ri])
        if pid not in rows_buf:
            rows_buf[pid] = []
        rows_buf[pid].append(int(ri))
    # Sort unique_pids for deterministic, v2-comparable bootstrap (M6 fix).
    unique_pids = np.array(sorted(rows_buf.keys()), dtype=np.int64)
    person_to_rows: dict[int, np.ndarray] = {
        pid: np.array(idxs, dtype=np.int64) for pid, idxs in rows_buf.items()
    }

    # Point estimate on the masked rows
    point = float(metric_fn(y_true[mask], y_pred[mask]))

    # np.random.RandomState (legacy MT19937) for parity with v2's
    # cluster_bootstrap_metric — CIs are bit-comparable when mask is all-ones.
    rng = np.random.RandomState(20260609)
    scores = []
    for _ in range(n_boot):
        sample_pids = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idx = np.concatenate([person_to_rows[int(p)] for p in sample_pids])
        if y_true[idx].sum() < 5 or y_true[idx].sum() > len(idx) - 5:
            continue
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    # Stricter "enough samples" threshold than v2's <100 to surface rare-endpoint
    # CI fragility — but still report n_bootstrap_valid even when the CI is NaN.
    n_ok = len(scores)
    if n_ok < 100:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "n_valid": int(mask.sum()), "n_bootstrap_valid": n_ok}
    lo = float(np.quantile(scores, (1 - ci) / 2))
    hi = float(np.quantile(scores, 1 - (1 - ci) / 2))
    return {"point": point, "lo": lo, "hi": hi,
            "n_valid": int(mask.sum()), "n_bootstrap_valid": n_ok}


# ============================================================
# Main
# ============================================================

@torch.no_grad()
def main() -> None:
    print(f"device={DEVICE}", flush=True)

    # Subgroup-definition file integrity check (per §13.3)
    if SUBGROUP_DEF_PATH.exists():
        sha = file_sha1(SUBGROUP_DEF_PATH)
        print(f"subgroup_definitions.json sha1 = {sha}", flush=True)
    else:
        print(f"WARNING: {SUBGROUP_DEF_PATH} not found — subgroup analysis skipped", flush=True)

    # ---- Load v3 model
    with open(MODEL_DIR / "config.json") as f:
        saved = json.load(f)
    cfg_d = saved["config"].copy()
    # Restore head_specs from JSON list-of-lists to tuple-of-tuples
    # (round-trip preserves *outer* tuple via tuple() but inner items stay as
    # lists if we don't explicitly coerce — M7 fix).
    cfg_d["head_specs"] = tuple(tuple(s) for s in cfg_d["head_specs"])
    cfg = PhoneFMV3Config(**cfg_d)
    max_seq_len = cfg.max_seq_len

    # ---- Pre-scan test shards to build person_id row vector
    test_glob = str(DATA_DIR / "test_*.parquet")
    print(f"\npre-scanning test shards from {test_glob}...", flush=True)
    person_ids_per_row: list = []
    shard_first_pids: list = []
    for path in sorted(glob.glob(test_glob)):
        df = pd.read_parquet(path, columns=["person_id"])
        pids = df["person_id"].tolist()
        shard_first_pids.append(pids[0] if pids else None)
        person_ids_per_row.extend(pids)
    person_ids_array = np.array(person_ids_per_row, dtype=np.int64)
    n_pid = len(np.unique(person_ids_array))
    print(f"  n_rows = {len(person_ids_array):,}   n_unique_persons = {n_pid:,}", flush=True)

    # ---- Model
    model = PhoneFMV3(cfg).to(DEVICE)
    state = torch.load(MODEL_DIR / "best.pt", map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"model loaded: {model.num_params() / 1e6:.1f}M params", flush=True)

    test_loader = make_loader_v3(
        test_glob, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        shuffle=False, max_seq_len=max_seq_len, n_confounders=cfg.n_confounders,
    )
    n_dataset = len(test_loader.dataset)
    print(f"test_loader: {len(test_loader)} batches, {n_dataset:,} windows", flush=True)
    if n_dataset != len(person_ids_array):
        raise RuntimeError(f"row count mismatch: {n_dataset} vs {len(person_ids_array)}")

    # ---- Per-shard parity check (catches reorder bugs)
    for shard_idx, first_pid in enumerate(shard_first_pids):
        if first_pid is None:
            continue
        ds_first = int(test_loader.dataset.frames[shard_idx].iloc[0]["person_id"])
        if ds_first != first_pid:
            raise RuntimeError(f"shard {shard_idx} parity check failed: ds_first={ds_first} vs first_pid={first_pid}")
    print(f"  per-shard parity verified across {len(shard_first_pids)} shards (OK)", flush=True)

    # ---- Forward pass — collect preds + labels + masks for all 13 heads
    preds: dict[str, list[np.ndarray]] = {name: [] for name in HEAD_NAMES}
    labels: dict[str, list[np.ndarray]] = {name: [] for name in HEAD_NAMES}
    masks: dict[str, list[np.ndarray]] = {name: [] for name in HEAD_NAMES}
    t0 = time.time()
    for i, batch in enumerate(test_loader):
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda"), dtype=torch.bfloat16):
            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["token_types"].to(DEVICE),
                batch["day_index"].to(DEVICE),
                batch["wearable_feats"].to(DEVICE),
                batch["confounders"].to(DEVICE),
                batch["attn_mask"].to(DEVICE),
            )
        for name in HEAD_NAMES:
            preds[name].append(torch.sigmoid(logits[name].float()).cpu().numpy())
            labels[name].append(batch["labels"][name].numpy())
            masks[name].append(batch["masks"][name].numpy())
        if i % 100 == 0:
            print(f"  batch {i}/{len(test_loader)}  elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    print(f"forward done in {(time.time()-t0)/60:.1f} min", flush=True)

    # ---- Cluster bootstrap per head
    results: dict[str, dict] = {}
    print(f"\n=== v3 TEST RESULTS (cluster-bootstrap CIs, person-level) ===", flush=True)
    print(f"{'head':<28} {'n_valid':>8} {'n_pos':>6} {'rate':>7}  "
          f"{'AUROC (95% CI)':<30} {'AUPRC (95% CI)':<30}", flush=True)

    for name in HEAD_NAMES:
        p = np.concatenate(preds[name])
        y = np.concatenate(labels[name]).astype(np.int64)
        m = np.concatenate(masks[name]).astype(bool)
        n_valid = int(m.sum())
        n_pos = int(y[m].sum()) if n_valid > 0 else 0
        rate = n_pos / max(n_valid, 1)
        if n_pos < 5 or n_pos > n_valid - 5:
            print(f"{name:<28} {n_valid:>8,} {n_pos:>6,} {rate:>7.4f}  "
                  f"insufficient positives", flush=True)
            results[name] = {"n_valid": n_valid, "n_pos": n_pos,
                             "auroc": None, "auprc": None,
                             "skipped": "n_pos < 5"}
            continue
        auroc = cluster_bootstrap_masked(
            y, p, person_ids_array, m, roc_auc_score, n_boot=BOOTSTRAP_N, ci=CI_LEVEL,
        )
        auprc = cluster_bootstrap_masked(
            y, p, person_ids_array, m, average_precision_score, n_boot=BOOTSTRAP_N, ci=CI_LEVEL,
        )
        print(f"{name:<28} {n_valid:>8,} {n_pos:>6,} {rate:>7.4f}  "
              f"{auroc['point']:.4f} ({auroc['lo']:.4f}-{auroc['hi']:.4f})  "
              f"{auprc['point']:.4f} ({auprc['lo']:.4f}-{auprc['hi']:.4f})", flush=True)
        results[name] = {"n_valid": n_valid, "n_pos": n_pos, "pos_rate": rate,
                         "auroc": auroc, "auprc": auprc}

    # ---- Negative-control sanity check (§11.2 expects AUROC ≈ 0.5)
    print(f"\n=== NEGATIVE CONTROL CHECK (expected AUROC ≈ 0.5) ===", flush=True)
    for (ep, h) in NEGATIVE_CONTROL_HEADS:
        name = head_name(ep, h)
        r = results.get(name, {})
        au = r.get("auroc")
        if au is None:
            print(f"  {name}: skipped", flush=True)
            continue
        verdict = "OK" if au["lo"] <= 0.6 else "WARN (>0.6 suggests utilization confound)"
        print(f"  {name}: AUROC {au['point']:.4f} ({au['lo']:.4f}-{au['hi']:.4f})  {verdict}", flush=True)

    # ---- v3 decision gates per (endpoint, horizon)
    print(f"\n=== v3 DECISION GATES ===", flush=True)
    gate_outcomes: dict[str, str] = {}
    for gate_key, threshold in DECISION_GATES.items():
        head_key = gate_key.replace("_auroc_lo", "")
        r = results.get(head_key, {}).get("auroc")
        if r is None:
            gate_outcomes[gate_key] = "SKIPPED"
            print(f"  {gate_key}: SKIPPED (no AUROC)", flush=True)
            continue
        passed = r["lo"] >= threshold
        gate_outcomes[gate_key] = "PASS" if passed else "PAUSE"
        marker = "✓" if passed else "✗"
        print(f"  {marker} {gate_key}: lo={r['lo']:.4f} vs threshold {threshold:.2f} → "
              f"{gate_outcomes[gate_key]}", flush=True)

    # ---- Final write
    final_results = {
        "model_dir": str(MODEL_DIR),
        "test_shards_glob": test_glob,
        "n_test_persons": int(n_pid),
        "n_test_windows": int(len(person_ids_array)),
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_kind": "cluster_at_person_level_masked",
        "ci_level": CI_LEVEL,
        "endpoints": results,
        "decision_gates": gate_outcomes,
        "subgroup_def_sha1": file_sha1(SUBGROUP_DEF_PATH) if SUBGROUP_DEF_PATH.exists() else None,
        "head_specs": [list(s) for s in (PRIMARY_HEADS + NEGATIVE_CONTROL_HEADS)],
    }
    with open(OUT_PATH, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nresults saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
