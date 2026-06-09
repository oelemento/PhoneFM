"""Val-set cluster bootstrap CI computation for v2.

Mirror image of 06_eval_v2_test.py but evaluates on val_*.parquet shards.
Purpose: compare val vs test cluster-bootstrap CIs to determine whether the
test val→test gap is a real generalization issue or a measurement artifact
of using val point estimates without uncertainty quantification.

If val cluster-bootstrap CI is similar width to test, the model is
generalizing fine and the saved val point of 0.9285 was misleading us
about the actual measurement uncertainty.

Output: /home/jupyter/workspace/phonefm-data/phonefm_v2/val_results.json
"""

from __future__ import annotations

import glob
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

# Load the test script as a module to reuse cluster_bootstrap_metric
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("eval_test", HERE / "06_eval_v2_test.py")
assert _spec is not None and _spec.loader is not None
eval_test = _ilu.module_from_spec(_spec)
# Don't exec the full main() — strip the __main__ guard
src = (HERE / "06_eval_v2_test.py").read_text().split("if __name__")[0]
exec(compile(src, "eval_test_partial", "exec"), eval_test.__dict__)

from phonefm_dataset_v2 import (  # type: ignore[import-not-found]
    ENDPOINTS,
    make_loader_v2,
)
from phonefm_model_v2 import PhoneFMV2, PhoneFMV2Config  # type: ignore[import-not-found]

# ============================================================
# Config — overrides test script defaults
# ============================================================

DATA_DIR = Path("/home/jupyter/workspace/phonefm-data/tokenized_v2")
MODEL_DIR = Path("/home/jupyter/workspace/phonefm-data/phonefm_v2")
OUT_PATH = MODEL_DIR / "val_results.json"
BATCH_SIZE = 32
NUM_WORKERS = 0
BOOTSTRAP_N = 1000
CI_LEVEL = 0.95
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main() -> None:
    print(f"device={DEVICE}", flush=True)

    with open(MODEL_DIR / "config.json") as f:
        saved = json.load(f)
    cfg = PhoneFMV2Config(**saved["config"])
    max_seq_len = cfg.max_seq_len

    # Pre-scan val shards for person_ids
    val_glob = str(DATA_DIR / "val_*.parquet")
    print(f"\npre-scanning val shards from {val_glob}...", flush=True)
    person_ids_per_row: list = []
    shard_first_pids: list = []
    for path in sorted(glob.glob(val_glob)):
        df = pd.read_parquet(path, columns=["person_id"])
        pids = df["person_id"].tolist()
        shard_first_pids.append(pids[0] if pids else None)
        person_ids_per_row.extend(pids)
    person_ids_array = np.array(person_ids_per_row, dtype=np.int64)
    n_pid = len(np.unique(person_ids_array))
    print(f"  n_rows = {len(person_ids_array):,}   n_unique_persons = {n_pid:,}", flush=True)

    # Load model
    model = PhoneFMV2(cfg).to(DEVICE)
    state = torch.load(MODEL_DIR / "best.pt", map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"model loaded: {model.num_params() / 1e6:.1f}M params", flush=True)

    val_loader = make_loader_v2(
        val_glob,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, balance_on=None,
        shuffle=False, max_seq_len=max_seq_len,
    )
    n_dataset = len(val_loader.dataset)
    print(f"val_loader: {len(val_loader)} batches, {n_dataset:,} windows", flush=True)
    if n_dataset != len(person_ids_array):
        raise RuntimeError(f"row count mismatch: {n_dataset} vs {len(person_ids_array)}")

    # Per-shard parity check
    for shard_idx, first_pid in enumerate(shard_first_pids):
        if first_pid is None:
            continue
        ds_first = int(val_loader.dataset.frames[shard_idx].iloc[0]["person_id"])
        if ds_first != first_pid:
            raise RuntimeError(f"shard {shard_idx} parity check failed")
    print(f"  per-shard parity verified across {len(shard_first_pids)} shards (OK)", flush=True)

    # Forward
    preds: dict = {ep: [] for ep in ENDPOINTS}
    labels: dict = {ep: [] for ep in ENDPOINTS}
    t0 = time.time()
    for i, batch in enumerate(val_loader):
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
            print(f"  batch {i}/{len(val_loader)}  elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    print(f"forward done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Cluster bootstrap
    results: dict = {}
    print("\n=== v2 VAL RESULTS (cluster-bootstrap CIs, person-level) ===", flush=True)
    print(f"{'endpoint':<12} {'n':>7} {'n_pos':>6} {'rate':>7} "
          f"{'AUROC (95% CI)':<32} {'AUPRC (95% CI)':<32}", flush=True)
    for ep in ENDPOINTS:
        if not preds[ep]:
            results[ep] = {"error": "head missing"}
            continue
        y_pred = np.concatenate(preds[ep])
        y_true = np.concatenate(labels[ep]).astype(np.int64)
        n = len(y_true)
        n_pos = int(y_true.sum())
        if n_pos < 5 or n_pos > n - 5:
            print(f"{ep:<12} {n:>7,} {n_pos:>6,} {n_pos/n:>7.4f}  "
                  f"insufficient positives", flush=True)
            results[ep] = {"n": n, "n_pos": n_pos, "auroc": None, "auprc": None,
                          "skipped": "n_pos < 5"}
            continue
        auroc = eval_test.cluster_bootstrap_metric(
            y_true, y_pred, person_ids_array, roc_auc_score,
            n_boot=BOOTSTRAP_N, ci=CI_LEVEL,
        )
        auprc = eval_test.cluster_bootstrap_metric(
            y_true, y_pred, person_ids_array, average_precision_score,
            n_boot=BOOTSTRAP_N, ci=CI_LEVEL,
        )
        print(f"{ep:<12} {n:>7,} {n_pos:>6,} {n_pos/n:>7.4f}  "
              f"{auroc['point']:.4f} ({auroc['lo']:.4f}-{auroc['hi']:.4f})  "
              f"{auprc['point']:.4f} ({auprc['lo']:.4f}-{auprc['hi']:.4f})", flush=True)
        results[ep] = {
            "n": n, "n_pos": n_pos, "pos_rate": n_pos / n,
            "auroc": auroc, "auprc": auprc,
        }

    final_results = {
        "model_dir": str(MODEL_DIR),
        "val_shards_glob": val_glob,
        "n_val_persons": int(n_pid),
        "n_val_windows": len(person_ids_array),
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
