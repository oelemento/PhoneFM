"""
07_save_predictions.py — re-run the v3 model forward pass on the test set and
persist per-window predictions + (person_id, end_date) metadata so we can build
risk trajectories aligned to event timing.

Why a separate script?  06_eval_v3_test.py computes aggregate AUROCs and writes
test_results.json, but does NOT keep per-window predictions on disk.  We need
those to plot "risk over time leading up to a cardiovascular event."  This
script is otherwise a near-clone of 06_eval_v3_test.py's forward-pass loop —
same loader, same autocast pattern (already patched to torch.amp.autocast for
CPU compatibility), same checkpoint.

Inputs:
  - ~/workspace/phonefm-data/tokenized_v3/test_*.parquet      (13 shards)
  - ~/workspace/phonefm-data/phonefm_v3/best.pt + config.json (locked checkpoint)

Output:
  - ~/workspace/phonefm-data/phonefm_v3/test_predictions.parquet
    with columns:
        person_id              int64
        end_date               object (string "YYYY-MM-DD")
        cv30_pred              float32 (sigmoid prob)
        cv30_label             int8
        cv30_mask              int8    (1 if this window was a valid eval row)
        cv180_pred / label / mask  (likewise for 180d horizon)
        cv365_pred / label / mask  (likewise for 365d horizon)

Sanity checks (all run in-process before the parquet is written):
  - per-shard first-person_id parity between pre-scan and loader.dataset.frames
    (ported from 06_eval_v3_test.py — catches any silent shard reordering)
  - row-count parity between metadata pre-scan and loader.dataset
  - prediction-count parity per head
  - cv_composite_30d AUROC reproduction check: should match
    the locked test_results.json value of 0.8857 within 1e-6

Run AFTER 06_eval_v3_test.py is done (uses same best.pt):
    cd ~/repos/PhoneFM/workbench && nohup python3 -u 07_save_predictions.py \
        > ~/workspace/phonefm-data/phonefm_v3/save_predictions.log 2>&1 &

Runtime: ~5h on n1-highmem-2 CPU (matches 06_eval_v3_test.py pace).
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))

from phonefm_dataset_v3 import HEAD_NAMES, make_loader_v3
from phonefm_model_v3 import PhoneFMV3, PhoneFMV3Config


# ===========================================================================
# Config
# ===========================================================================

DATA_DIR = Path("/home/jupyter/workspace/phonefm-data/tokenized_v3")
MODEL_DIR = Path("/home/jupyter/workspace/phonefm-data/phonefm_v3")
OUT_PATH = MODEL_DIR / "test_predictions.parquet"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_WORKERS = 0  # match 06_eval_v3_test.py

# Heads we want to persist (cv composite only, since trajectory analysis
# centers on cardiovascular risk).  All three horizons so the analysis can
# pick whichever lead time is most informative.
HEADS_TO_SAVE = ("cv_composite_30d", "cv_composite_180d", "cv_composite_365d")

# Published reference AUROC for cv_composite_30d on this test set, written by
# 06_eval_v3_test.py to test_results.json on 2026-06-10.  The new script must
# reproduce this within 1e-6 — if it doesn't, the predictions are wrong even
# though every other guard passed.
REFERENCE_AUROC_CV30D = 0.8856857329969465


# ===========================================================================
# Main
# ===========================================================================

@torch.no_grad()
def main() -> None:
    print(f"device={DEVICE}", flush=True)
    print(f"out={OUT_PATH}", flush=True)
    for head in HEADS_TO_SAVE:
        assert head in HEAD_NAMES, f"{head} not in HEAD_NAMES"

    # ---- Load v3 model from locked checkpoint
    with open(MODEL_DIR / "config.json") as f:
        saved = json.load(f)
    cfg_d = saved["config"].copy()
    cfg_d["head_specs"] = tuple(tuple(s) for s in cfg_d["head_specs"])
    cfg = PhoneFMV3Config(**cfg_d)
    model = PhoneFMV3(cfg).to(DEVICE)
    state = torch.load(MODEL_DIR / "best.pt", map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"model loaded: {model.num_params() / 1e6:.1f}M params", flush=True)

    # ---- Pre-scan test shards to build a row-aligned metadata table.
    #      Order MUST match make_loader_v3's iteration order (no shuffling
    #      in test mode → shards in sorted glob order, rows in shard order).
    test_glob = str(DATA_DIR / "test_*.parquet")
    meta_cols = ["person_id", "end_date"] + [f"label_{h}" for h in HEADS_TO_SAVE] + [f"mask_{h}" for h in HEADS_TO_SAVE]
    print(f"\npre-scanning test shards from {test_glob}...", flush=True)
    meta_rows: list[pd.DataFrame] = []
    shard_first_pids: list[int] = []   # for the per-shard parity check
    for path in sorted(glob.glob(test_glob)):
        df = pd.read_parquet(path, columns=meta_cols)
        shard_first_pids.append(int(df["person_id"].iloc[0]))
        meta_rows.append(df)
    meta_df = pd.concat(meta_rows, ignore_index=True)
    n_rows = len(meta_df)
    print(f"  n_rows = {n_rows}  n_unique_persons = {meta_df['person_id'].nunique()}", flush=True)

    # ---- Build the loader (positional first arg, like 06_eval_v3_test.py).
    #      Pass n_confounders explicitly so a future cfg change can't silently
    #      desync loader/model shapes.
    test_loader = make_loader_v3(
        test_glob,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        max_seq_len=cfg.max_seq_len,
        n_confounders=cfg.n_confounders,
    )
    n_dataset = len(test_loader.dataset)
    print(f"test_loader: {len(test_loader)} batches, {n_dataset} windows", flush=True)
    if n_dataset != n_rows:
        raise RuntimeError(
            f"row-alignment guard FAILED: loader has {n_dataset} rows, "
            f"metadata has {n_rows}.  Predictions cannot be aligned to (person_id, end_date)."
        )

    # ---- Per-shard parity check (ported from 06_eval_v3_test.py).
    #      Catches any silent shard reordering between the pre-scan glob and
    #      the dataset's internal shard order.  13 integer comparisons; if
    #      the loader was ever changed to length-bucket or distribute shards
    #      differently, this fires immediately instead of after 5h.
    for shard_idx, first_pid in enumerate(shard_first_pids):
        ds_first = int(test_loader.dataset.frames[shard_idx].iloc[0]["person_id"])
        if ds_first != first_pid:
            raise RuntimeError(
                f"per-shard parity FAILED at shard {shard_idx}: "
                f"pre-scan first person_id={first_pid}, loader first person_id={ds_first}.  "
                f"Predictions would be silently misaligned."
            )
    print(f"  per-shard parity verified across {len(shard_first_pids)} shards (OK)", flush=True)

    # ---- Forward pass — collect sigmoid probabilities, in loader order.
    preds: dict[str, list[np.ndarray]] = {h: [] for h in HEADS_TO_SAVE}
    t0 = time.time()
    for i, batch in enumerate(test_loader):
        with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda"), dtype=torch.bfloat16):
            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["token_types"].to(DEVICE),
                batch["day_index"].to(DEVICE),
                batch["wearable_feats"].to(DEVICE),
                batch["confounders"].to(DEVICE),
                batch["attn_mask"].to(DEVICE),
            )
        for h in HEADS_TO_SAVE:
            preds[h].append(torch.sigmoid(logits[h].float()).cpu().numpy())
        if i % 100 == 0:
            print(f"  batch {i}/{len(test_loader)}  elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    print(f"forward done in {(time.time()-t0)/60:.1f} min", flush=True)

    # ---- Stitch predictions + metadata.  Force end_date to "YYYY-MM-DD"
    #      regardless of whether parquet round-trips as object-of-date or
    #      datetime64[ns] (depends on pyarrow/pandas versions).
    end_date_str = pd.to_datetime(meta_df["end_date"]).dt.strftime("%Y-%m-%d").to_numpy()
    out_cols: dict[str, np.ndarray] = {
        "person_id": meta_df["person_id"].to_numpy(dtype=np.int64),
        "end_date": end_date_str,
    }
    for h in HEADS_TO_SAVE:
        short = h.replace("cv_composite_", "cv")
        p = np.concatenate(preds[h]).astype(np.float32).reshape(-1)
        if len(p) != n_rows:
            raise RuntimeError(
                f"prediction-count guard FAILED for {h}: "
                f"got {len(p)} preds for {n_rows} rows."
            )
        out_cols[f"{short}_pred"] = p
        out_cols[f"{short}_label"] = meta_df[f"label_{h}"].to_numpy(dtype=np.int8)
        out_cols[f"{short}_mask"] = meta_df[f"mask_{h}"].to_numpy(dtype=np.int8)

    # ---- AUROC reproduction guard.  Apply the same masking the eval script
    #      used (mask == 1) and recompute cv_composite_30d AUROC.  Must match
    #      the published 0.8857 within 1e-6 — anything else means the
    #      predictions are wrong even though every structural guard passed.
    cv30_pred = out_cols["cv30_pred"]
    cv30_label = out_cols["cv30_label"]
    cv30_mask = out_cols["cv30_mask"].astype(bool)
    auroc = roc_auc_score(cv30_label[cv30_mask], cv30_pred[cv30_mask])
    print(f"\nreproduction check: cv_composite_30d AUROC = {auroc:.10f}  "
          f"(reference {REFERENCE_AUROC_CV30D:.10f}, delta {auroc - REFERENCE_AUROC_CV30D:+.2e})",
          flush=True)
    if abs(auroc - REFERENCE_AUROC_CV30D) > 1e-6:
        raise RuntimeError(
            f"AUROC reproduction FAILED: got {auroc}, expected ~{REFERENCE_AUROC_CV30D}.  "
            f"Predictions diverge from 06_eval_v3_test.py — do not write output."
        )

    # ---- Mean-pred sanity print (free check that predictions weren't, e.g.,
    #      all 0.5 or constant).  Should track the true positive rate.
    for h in HEADS_TO_SAVE:
        short = h.replace("cv_composite_", "cv")
        m = out_cols[f"{short}_mask"].astype(bool)
        if m.sum() == 0:
            continue
        true_rate = out_cols[f"{short}_label"][m].mean()
        mean_pred = out_cols[f"{short}_pred"][m].mean()
        print(f"  {h:25s}: n_valid={int(m.sum()):>6d}  true_rate={true_rate:.4f}  mean_pred={mean_pred:.4f}", flush=True)

    out_df = pd.DataFrame(out_cols)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(out_df, preserve_index=False), OUT_PATH)
    print(f"\nsaved predictions: {OUT_PATH}  rows={len(out_df)}  cols={list(out_df.columns)}", flush=True)


if __name__ == "__main__":
    main()
