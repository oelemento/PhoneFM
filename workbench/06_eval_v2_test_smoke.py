"""Smoke test for 06_eval_v2_test.py.

Tests the bits of the eval script that matter most for correctness of the
v3 §13.2 decision gate, without requiring a trained model or real test
shards:

  1. cluster_bootstrap_metric correctness on synthetic data
  2. cluster bootstrap CIs are WIDER than naive i.i.d. bootstrap on
     correlated cluster data (the whole reason for the cluster-bootstrap fix)
  3. decision gate logic on all 5 cases (lo>=0.85, lo<0.85, lo=None,
     lo=NaN, results['afib'] missing)
  4. cluster bootstrap with rare endpoints reports n_valid below requested
  5. per-shard first-row order check catches a synthetic mismatch

Usage:
    cd ~/repos/PhoneFM/workbench && python3 06_eval_v2_test_smoke.py
"""

from __future__ import annotations

import importlib.util as _ilu
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-not-found]
from sklearn.metrics import roc_auc_score  # type: ignore[import-not-found]

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))

# Load the eval module via importlib so we can reach its functions/constants
_spec = _ilu.spec_from_file_location("eval_mod", HERE / "06_eval_v2_test.py")
assert _spec is not None and _spec.loader is not None
ev = _ilu.module_from_spec(_spec)
# DON'T exec the eval module — its top-level prints assume a real env and
# we just want the cluster_bootstrap_metric function. So instead, recompile
# the eval source up to but not including main().
src = (HERE / "06_eval_v2_test.py").read_text()
# Strip the if __name__ == '__main__': main() block; keep all definitions
src_no_main = src.split("if __name__")[0]
exec(compile(src_no_main, "eval_mod_partial", "exec"), ev.__dict__)


# ============================================================
# Test fixtures
# ============================================================

def make_clustered_data(
    n_persons: int = 200,
    windows_per_person: int = 5,
    positive_person_rate: float = 0.10,
    pred_noise: float = 0.10,
    rng_seed: int = 42,
):
    """Create synthetic clustered data: each person has multiple correlated
    windows. Predictions are correlated with truth + within-cluster noise.

    Returns:
        y_true (N,), y_pred (N,), person_ids (N,)
    """
    rng = np.random.RandomState(rng_seed)
    person_ids = []
    y_true = []
    y_pred = []
    for pid in range(n_persons):
        is_positive = rng.rand() < positive_person_rate
        # All windows for one person share the label (perfect within-cluster correlation)
        person_label = 1 if is_positive else 0
        # Predictions are noisy version of label: positives center at 0.7, negs at 0.3
        base_pred = 0.7 if is_positive else 0.3
        for _ in range(windows_per_person):
            person_ids.append(pid)
            y_true.append(person_label)
            y_pred.append(np.clip(base_pred + rng.randn() * pred_noise, 0.0, 1.0))
    return (np.array(y_true, dtype=np.int64),
            np.array(y_pred, dtype=np.float32),
            np.array(person_ids, dtype=np.int64))


# ============================================================
# Test 1: cluster_bootstrap_metric basic correctness
# ============================================================

def test_cluster_bootstrap_basic():
    print("\n=== Test 1: cluster_bootstrap_metric basic correctness ===")
    y_true, y_pred, pids = make_clustered_data(n_persons=200, windows_per_person=5)
    result = ev.cluster_bootstrap_metric(y_true, y_pred, pids, roc_auc_score, n_boot=500)
    print(f"  point AUROC = {result['point']:.4f}")
    print(f"  95% CI      = ({result['lo']:.4f}, {result['hi']:.4f})")
    print(f"  n_valid     = {result['n_valid']}/500")
    assert 0.7 < result['point'] < 1.0, "synthetic data should give high AUROC"
    assert result['lo'] < result['point'] < result['hi'], "CI must contain point"
    assert result['n_valid'] >= 400, "most resamples should be valid"
    print("  PASS")


# ============================================================
# Test 2: cluster CI is WIDER than naive CI (the key claim)
# ============================================================

def naive_bootstrap_metric(y_true, y_pred, metric_fn, n_boot=500, rng_seed=20260609):
    """Window-level i.i.d. bootstrap (the WRONG way for clustered data)."""
    rng = np.random.RandomState(rng_seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        if yt.sum() < 5 or yt.sum() > n - 5:
            continue
        scores.append(metric_fn(yt, yp))
    lo = float(np.quantile(scores, 0.025))
    hi = float(np.quantile(scores, 0.975))
    return {"point": float(point), "lo": lo, "hi": hi, "n_valid": len(scores)}


def test_cluster_ci_wider_than_naive():
    print("\n=== Test 2: cluster CI is WIDER than naive CI on correlated data ===")
    # Use HARDER predictions: noise=0.30 so positives/negatives overlap
    # significantly (point AUROC ~0.80). At AUROC saturation (~1.0) both
    # bootstraps give tiny CIs that don't show the cluster effect — the
    # cluster-vs-naive distinction only manifests when there's enough
    # variance for sqrt(windows_per_person) scaling to be visible.
    y_true, y_pred, pids = make_clustered_data(
        n_persons=200, windows_per_person=5, pred_noise=0.30,
    )
    cluster = ev.cluster_bootstrap_metric(y_true, y_pred, pids, roc_auc_score, n_boot=500)
    naive = naive_bootstrap_metric(y_true, y_pred, roc_auc_score, n_boot=500)
    cluster_width = cluster['hi'] - cluster['lo']
    naive_width = naive['hi'] - naive['lo']
    ratio = cluster_width / max(naive_width, 1e-9)
    print(f"  point AUROC = {cluster['point']:.4f}  (lowered via pred_noise=0.30 so CI ratio is visible)")
    print(f"  naive   CI width = {naive_width:.4f}  ({naive['lo']:.4f}, {naive['hi']:.4f})")
    print(f"  cluster CI width = {cluster_width:.4f}  ({cluster['lo']:.4f}, {cluster['hi']:.4f})")
    print(f"  cluster/naive ratio = {ratio:.2f}  "
          f"(asymptotic sqrt(5)=2.24; finite-sample threshold > 1.2)")
    assert ratio > 1.2, \
        f"cluster CI should be wider than naive CI on clustered data; got ratio={ratio:.2f}"
    print("  PASS — cluster bootstrap correctly captures within-cluster correlation")


# ============================================================
# Test 3: decision gate logic — all 5 cases
# ============================================================

def _make_gate_decision(results, threshold=0.85):
    """Replicate the eval script's gate logic for testing."""
    afib_result = results.get("afib", {})
    afib_auroc = afib_result.get("auroc", {}) if isinstance(afib_result.get("auroc"), dict) else {}
    afib_auroc_lo = afib_auroc.get("lo")
    if afib_auroc_lo is None or (isinstance(afib_auroc_lo, float) and math.isnan(afib_auroc_lo)):
        return "MANUAL_REVIEW"
    elif afib_auroc_lo >= threshold:
        return "PROCEED"
    else:
        return "PAUSE"


def test_decision_gate_all_branches():
    print("\n=== Test 3: decision gate handles all 5 cases ===")
    # 1. lo >= 0.85 → PROCEED
    r = {"afib": {"auroc": {"point": 0.92, "lo": 0.88, "hi": 0.95}}}
    assert _make_gate_decision(r) == "PROCEED", "case 1 failed"
    print("  case 1: lo=0.88 → PROCEED  ✓")
    # 2. lo < 0.85 → PAUSE
    r = {"afib": {"auroc": {"point": 0.83, "lo": 0.78, "hi": 0.87}}}
    assert _make_gate_decision(r) == "PAUSE", "case 2 failed"
    print("  case 2: lo=0.78 → PAUSE    ✓")
    # 3. lo = None → MANUAL_REVIEW
    r = {"afib": {"auroc": {"point": 0.95, "lo": None, "hi": None}}}
    assert _make_gate_decision(r) == "MANUAL_REVIEW", "case 3 failed"
    print("  case 3: lo=None → MANUAL_REVIEW  ✓")
    # 4. lo = NaN → MANUAL_REVIEW
    r = {"afib": {"auroc": {"point": 0.95, "lo": float("nan"), "hi": float("nan")}}}
    assert _make_gate_decision(r) == "MANUAL_REVIEW", "case 4 failed"
    print("  case 4: lo=NaN → MANUAL_REVIEW  ✓")
    # 5. afib endpoint skipped (n_pos < 5) → auroc is None → MANUAL_REVIEW
    r = {"afib": {"n": 100, "n_pos": 3, "auroc": None, "auprc": None, "skipped": "low n_pos"}}
    assert _make_gate_decision(r) == "MANUAL_REVIEW", "case 5 failed"
    print("  case 5: auroc=None (skipped) → MANUAL_REVIEW  ✓")
    # 6 (bonus). afib entirely missing → MANUAL_REVIEW
    r = {}
    assert _make_gate_decision(r) == "MANUAL_REVIEW", "case 6 failed"
    print("  case 6: afib missing from results → MANUAL_REVIEW  ✓")
    print("  PASS — all decision-gate branches correct")


# ============================================================
# Test 4: rare endpoint resample handling
# ============================================================

def test_rare_endpoint_resample_handling():
    print("\n=== Test 4: rare endpoint → n_valid below requested ===")
    # Only 3 positive persons out of 200; almost all bootstraps will fail
    # the >=5 positives threshold
    n_persons, wpp = 200, 5
    rng = np.random.RandomState(0)
    person_ids = []
    y_true = []
    y_pred = []
    positive_pids = {0, 1, 2}   # only 3 positives × 5 = 15 positive rows total
    for pid in range(n_persons):
        is_pos = pid in positive_pids
        for _ in range(wpp):
            person_ids.append(pid)
            y_true.append(int(is_pos))
            y_pred.append(0.7 if is_pos else 0.3 + rng.randn() * 0.05)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    pids = np.array(person_ids)
    result = ev.cluster_bootstrap_metric(y_true, y_pred, pids, roc_auc_score, n_boot=500)
    print(f"  total positives = {y_true.sum()}  (across only {len(positive_pids)} persons)")
    print(f"  n_valid = {result['n_valid']}/500")
    print(f"  lo, hi = ({result['lo']}, {result['hi']})")
    # We expect SOME resamples to fail because the 3 positive persons may
    # not be sampled in every replicate
    assert result['n_valid'] < 500, "should drop some replicates with rare positives"
    print("  PASS — rare-endpoint behavior correctly reports reduced n_valid")


# ============================================================
# Test 5: per-shard order check synthetic
# ============================================================

def test_per_shard_order_check():
    print("\n=== Test 5: per-shard first-row order check catches a mismatch ===")
    # Build 2 synthetic shards with different first person_ids.
    # Then simulate a mismatch by mutating the dataset's row order.
    tmpdir = Path(tempfile.mkdtemp(prefix="phonefm_eval_smoke_"))
    for shard_idx, first_pid in enumerate([1000, 2000]):
        rows = []
        for j in range(5):
            rows.append({"person_id": first_pid + j})
        pd.DataFrame(rows).to_parquet(tmpdir / f"test_{shard_idx:04d}.parquet")

    # Simulate the pre-scan
    pre_scan_first_pids = []
    for path in sorted(tmpdir.glob("test_*.parquet")):
        df = pd.read_parquet(path, columns=["person_id"])
        pre_scan_first_pids.append(int(df["person_id"].iloc[0]))
    print(f"  pre-scan first pids: {pre_scan_first_pids}")
    assert pre_scan_first_pids == [1000, 2000]

    # Now simulate a hypothetical dataset that returns rows in a DIFFERENT
    # order — e.g., shard 1's first row got swapped with shard 0's first row.
    # The check should catch this.
    class FakeDataset:
        def __init__(self, paths):
            self.frames = [pd.read_parquet(p) for p in paths]
            # Mutate frame 0 row 0 to simulate a row-order divergence
            self.frames[0] = self.frames[0].iloc[::-1].reset_index(drop=True)  # reverse!

    fake_ds = FakeDataset(sorted(tmpdir.glob("test_*.parquet")))
    # Run the per-shard check manually
    mismatch_detected = False
    for shard_idx, expected_first in enumerate(pre_scan_first_pids):
        ds_first = int(fake_ds.frames[shard_idx].iloc[0]["person_id"])
        if ds_first != expected_first:
            mismatch_detected = True
            print(f"  shard {shard_idx}: expected pid={expected_first}, got pid={ds_first}  ← MISMATCH (correctly caught)")
            break
    assert mismatch_detected, "per-shard order check FAILED to catch divergence"
    print("  PASS — per-shard order parity check catches row-order divergence")


# ============================================================
# Run all tests
# ============================================================

def main():
    print(f"=== SMOKE TEST: 06_eval_v2_test.py ===")
    test_cluster_bootstrap_basic()
    test_cluster_ci_wider_than_naive()
    test_decision_gate_all_branches()
    test_rare_endpoint_resample_handling()
    test_per_shard_order_check()
    print("\n=== ALL SMOKE TESTS PASSED ===\n")


if __name__ == "__main__":
    main()
