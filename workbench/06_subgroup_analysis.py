"""PhoneFM v3 subgroup analysis — per-cell AUROC/AUPRC + cluster bootstrap CIs.

Implements the pre-registered subgroup analysis declared in
`workbench/subgroup_definitions.json` per `docs/v3_spec.md` §13.3.

Loads the model + test shards once, computes per-cell metrics over the
declared marginal subgroups (race, age, sex, SES) plus the one pre-registered
interaction (race × age_at_end_date). For each (head, cell) combination,
reports AUROC, AUPRC, Brier score, and calibration slope/intercept with
cluster-bootstrap 95% CIs at the person level.

Inputs:
  ~/workspace/phonefm-data/tokenized_v3/test_*.parquet
  ~/workspace/phonefm-data/phonefm_v3/best.pt
  workbench/subgroup_definitions.json

Output:
  ~/workspace/phonefm-data/phonefm_v3/subgroup_results.json

Run AFTER 06_eval_v3_test.py:
    cd ~/repos/PhoneFM/workbench && python3 06_subgroup_analysis.py
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
from sklearn.metrics import (  # type: ignore[import-not-found]
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))

from phonefm_dataset_v3 import (  # type: ignore[import-not-found]
    HEAD_NAMES,
    make_loader_v3,
)
from phonefm_model_v3 import (  # type: ignore[import-not-found]
    PhoneFMV3,
    PhoneFMV3Config,
    PRIMARY_BEST_METRIC_HEADS,
)


# ============================================================
# Config
# ============================================================

DATA_DIR = Path("/home/jupyter/workspace/phonefm-data/tokenized_v3")
MODEL_DIR = Path("/home/jupyter/workspace/phonefm-data/phonefm_v3")
OUT_PATH = MODEL_DIR / "subgroup_results.json"
SUBGROUP_DEF_PATH = HERE / "subgroup_definitions.json"

BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def cluster_bootstrap(
    y_true: np.ndarray, y_pred: np.ndarray, person_ids: np.ndarray, metric_fn,
    n_boot: int, ci: float = 0.95, seed: int = 20260609,
) -> dict:
    """Person-level cluster bootstrap CI for `metric_fn(y_true, y_pred)`."""
    if len(y_true) < 30 or y_true.sum() < 5 or y_true.sum() > len(y_true) - 5:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": int(len(y_true)), "n_pos": int(y_true.sum())}
    point = float(metric_fn(y_true, y_pred))
    rows_buf: dict[int, list[int]] = {}
    for ri in range(len(person_ids)):
        rows_buf.setdefault(int(person_ids[ri]), []).append(ri)
    unique_pids = np.array(sorted(rows_buf.keys()), dtype=np.int64)
    person_to_rows = {p: np.array(idxs, dtype=np.int64) for p, idxs in rows_buf.items()}
    rng = np.random.RandomState(seed)
    scores: list[float] = []
    for _ in range(n_boot):
        sample = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idx = np.concatenate([person_to_rows[int(p)] for p in sample])
        if y_true[idx].sum() < 5 or y_true[idx].sum() > len(idx) - 5:
            continue
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    if len(scores) < 100:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "n": int(len(y_true)), "n_pos": int(y_true.sum()),
                "n_bootstrap_valid": len(scores)}
    lo = float(np.quantile(scores, (1 - ci) / 2))
    hi = float(np.quantile(scores, 1 - (1 - ci) / 2))
    return {"point": point, "lo": lo, "hi": hi,
            "n": int(len(y_true)), "n_pos": int(y_true.sum()),
            "n_bootstrap_valid": len(scores)}


def calibration_slope_intercept(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Logistic-regression calibration: predicted log-odds vs observed label.

    Slope=1, intercept=0 is perfect calibration. Returns (intercept, slope).
    """
    eps = 1e-8
    p = np.clip(y_pred, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    X = np.column_stack([np.ones_like(logit), logit])
    # OLS on logit -> y_true (close enough to logistic for diagnostics)
    try:
        beta, *_ = np.linalg.lstsq(X, y_true.astype(np.float64), rcond=None)
        return float(beta[0]), float(beta[1])
    except Exception:
        return float("nan"), float("nan")


def load_subgroup_definitions() -> dict:
    """Load + sha1 the pre-registered subgroup definitions JSON."""
    sha = file_sha1(SUBGROUP_DEF_PATH)
    print(f"subgroup_definitions.json sha1 = {sha}", flush=True)
    with open(SUBGROUP_DEF_PATH) as f:
        defs = json.load(f)
    return {"sha1": sha, "defs": defs}


def bucket_age(age: int, age_buckets: dict[str, list[int]]) -> str | None:
    """Map a single age (years) to the first matching bucket label."""
    for name, (lo, hi) in age_buckets.items():
        if lo <= age <= hi:
            return name
    return None


def bucket_race(rcid: int, race_buckets: dict[str, list[int]]) -> str | None:
    for name, ids in race_buckets.items():
        if int(rcid) in [int(x) for x in ids]:
            return name
    return None


@torch.no_grad()
def main() -> None:
    print(f"device={DEVICE}", flush=True)
    subgroup_meta = load_subgroup_definitions()
    defs = subgroup_meta["defs"]

    # ---- Load v3 model
    with open(MODEL_DIR / "config.json") as f:
        saved = json.load(f)
    cfg_d = saved["config"].copy()
    cfg_d["head_specs"] = tuple(tuple(s) for s in cfg_d["head_specs"])
    cfg = PhoneFMV3Config(**cfg_d)
    max_seq_len = cfg.max_seq_len

    model = PhoneFMV3(cfg).to(DEVICE)
    state = torch.load(MODEL_DIR / "best.pt", map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"model loaded: {model.num_params() / 1e6:.1f}M params", flush=True)

    # ---- Pre-scan test shards for subgroup metadata + person_id index
    test_glob = str(DATA_DIR / "test_*.parquet")
    print(f"\npre-scanning test shards from {test_glob}...", flush=True)
    meta_cols = ["person_id", "race_concept_id", "sex_at_birth_concept_id",
                 "ses_income_quartile", "age_at_end_date"]
    meta_rows = []
    for path in sorted(glob.glob(test_glob)):
        meta_rows.append(pd.read_parquet(path, columns=meta_cols))
    meta_df = pd.concat(meta_rows, ignore_index=True)
    person_ids_array = meta_df["person_id"].to_numpy(dtype=np.int64)
    race_array = meta_df["race_concept_id"].to_numpy(dtype=np.int64)
    sex_array = meta_df["sex_at_birth_concept_id"].to_numpy(dtype=np.int64)
    ses_array = meta_df["ses_income_quartile"].to_numpy(dtype=np.int64)
    age_array = meta_df["age_at_end_date"].to_numpy(dtype=np.int64)
    print(f"  n_rows = {len(meta_df):,}   n_unique_persons = {meta_df['person_id'].nunique():,}",
          flush=True)

    test_loader = make_loader_v3(
        test_glob, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        shuffle=False, max_seq_len=max_seq_len, n_confounders=cfg.n_confounders,
    )
    if len(test_loader.dataset) != len(meta_df):
        raise RuntimeError(
            f"meta count {len(meta_df)} != loader count {len(test_loader.dataset)}"
        )

    # ---- Forward pass — gather preds + labels + masks for all heads
    print("\nforward pass...", flush=True)
    preds = {n: [] for n in HEAD_NAMES}
    labels = {n: [] for n in HEAD_NAMES}
    masks = {n: [] for n in HEAD_NAMES}
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
        for n in HEAD_NAMES:
            preds[n].append(torch.sigmoid(logits[n].float()).cpu().numpy())
            labels[n].append(batch["labels"][n].numpy())
            masks[n].append(batch["masks"][n].numpy())
        if i % 100 == 0:
            print(f"  batch {i}/{len(test_loader)}  elapsed={(time.time()-t0)/60:.1f}min",
                  flush=True)
    print(f"forward done in {(time.time()-t0)/60:.1f} min", flush=True)

    preds = {n: np.concatenate(v) for n, v in preds.items()}
    labels = {n: np.concatenate(v).astype(np.int64) for n, v in labels.items()}
    masks = {n: np.concatenate(v).astype(bool) for n, v in masks.items()}

    # ---- Subgroup loop. Focus on primary heads for the manuscript figure;
    # negative controls reported in full eval, not stratified.
    min_n = int(defs.get("min_n_per_cell", 30))
    bootstrap_n = int(defs.get("bootstrap_n", 1000))
    ci_level = float(defs.get("ci_level", 0.95))

    out: dict = {
        "_subgroup_def_sha1": subgroup_meta["sha1"],
        "_bootstrap_n": bootstrap_n,
        "_ci_level": ci_level,
        "_min_n_per_cell": min_n,
        "head_results": {},
    }

    for head in PRIMARY_BEST_METRIC_HEADS:
        if head not in preds:
            print(f"skip {head}: not in head names", flush=True)
            continue
        p_all = preds[head]
        y_all = labels[head]
        m_all = masks[head]
        head_out: dict = {"marginal": {}, "interaction": {}}

        # Marginals
        for dim_name, cells in [
            ("race", defs["race"]),
            ("sex_at_birth", defs["sex_at_birth"]),
            ("ses_income_quartile", defs["ses_income_quartile"]),
        ]:
            head_out["marginal"][dim_name] = {}
            for cell_label, ids in cells.items():
                if dim_name == "race":
                    in_cell = np.isin(race_array, [int(x) for x in ids])
                elif dim_name == "sex_at_birth":
                    in_cell = np.isin(sex_array, [int(x) for x in ids])
                else:
                    in_cell = np.isin(ses_array, [int(x) for x in ids])
                sel = in_cell & m_all
                head_out["marginal"][dim_name][cell_label] = _cell_metrics(
                    y_all[sel], p_all[sel], person_ids_array[sel],
                    min_n, bootstrap_n, ci_level,
                )

        # Age — uses range buckets, not exact-match
        head_out["marginal"]["age_at_end_date"] = {}
        for cell_label, (lo, hi) in defs["age_at_end_date"].items():
            in_cell = (age_array >= int(lo)) & (age_array <= int(hi))
            sel = in_cell & m_all
            head_out["marginal"]["age_at_end_date"][cell_label] = _cell_metrics(
                y_all[sel], p_all[sel], person_ids_array[sel],
                min_n, bootstrap_n, ci_level,
            )

        # Race × age interaction (the one pre-registered pairwise)
        for race_cell, race_ids in defs["race"].items():
            for age_cell, (lo, hi) in defs["age_at_end_date"].items():
                in_cell = (
                    np.isin(race_array, [int(x) for x in race_ids]) &
                    (age_array >= int(lo)) & (age_array <= int(hi))
                )
                sel = in_cell & m_all
                head_out["interaction"][f"{race_cell} | {age_cell}"] = _cell_metrics(
                    y_all[sel], p_all[sel], person_ids_array[sel],
                    min_n, bootstrap_n, ci_level,
                )

        out["head_results"][head] = head_out
        print(f"completed {head}", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults saved to {OUT_PATH}", flush=True)


def _cell_metrics(y: np.ndarray, p: np.ndarray, pids: np.ndarray,
                  min_n: int, n_boot: int, ci: float) -> dict:
    """Per-cell metric block. Returns NaN-filled block if cell is below min_n."""
    if len(y) < min_n or y.sum() < 5 or y.sum() > len(y) - 5:
        return {"n": int(len(y)), "n_pos": int(y.sum()),
                "skipped": f"n<{min_n} or n_pos<5"}
    intercept, slope = calibration_slope_intercept(y, p)
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "pos_rate": float(y.mean()),
        "AUROC": cluster_bootstrap(y, p, pids, roc_auc_score, n_boot, ci),
        "AUPRC": cluster_bootstrap(y, p, pids, average_precision_score, n_boot, ci),
        "Brier": float(brier_score_loss(y, p)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


if __name__ == "__main__":
    main()
