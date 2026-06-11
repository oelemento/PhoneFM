"""
09_ablation_wearables.py — quantify the importance of the wearable signal for
cardiovascular prediction, in AUROC terms, by input ablation at inference.

No retraining.  We evaluate, in a single GPU pass over the test set:
  - FULL      : inputs unchanged.
  - WEAR_PERM : wearable_feats permuted across windows (K independent random
                derangements within each shuffled batch, so every window's
                wearable block is reassigned to a DIFFERENT person).  This is
                permutation importance: it breaks the wearable<->outcome
                association while keeping the exact input distribution (no OOD
                artifact).  full - mean_k(WEAR_PERM_k) = wearable importance.
  - WEAR_ZERO : wearable_feats set to zero → BatchNorm(eval) maps to a CONSTANT
                embedding at every wearable position → zero per-window wearable
                info.  Mildly OOD; reported as a presence/structure cross-check.

Only wearable_feats changes; EHR tokens, confounders, day_index, attn_mask are
held fixed.  So the AUROC drop is the MARGINAL importance of correctly-valued
wearables CONDITIONAL ON EHR + confounders.  Read carefully:
  - It is NOT wearable sufficiency (wearables alone) — EHR/confounders remain.
  - A small value could mean "wearables add little" OR "wearables are redundant
    with EHR which compensates."  The single ablation cannot separate these.
  - The PERMUTATION delta is the defensible "wearable value importance"
    (in-distribution); the ZERO delta also removes wearable presence/structure
    and is mildly OOD, so a large perm-vs-zero gap is itself diagnostic.

Uncertainty is reported two ways (both review findings):
  - permutation SD: spread of importance across the K random permutations.
  - person-level cluster bootstrap 95% CI: resamples PERSONS (not windows),
    matching the clustering of the data (~44 windows/person) and the headline
    test AUROC's resampling unit.  Window-level bootstrap would be
    anti-conservative (pseudo-replication).

Runs FP32 on GPU (no autocast) so the full-condition AUROC reproduces the
0.8857 CPU fp32 baseline and there is no bf16 ranking noise in the delta.

Inputs:
  - ~/workspace/phonefm-data/tokenized_v3/test_*.parquet
  - ~/workspace/phonefm-data/phonefm_v3/best.pt + config.json
Output:
  - ~/workspace/phonefm-data/phonefm_v3/ablation_wearables.json

Run (GPU pod):
    cd ~/repos/PhoneFM/workbench && python3 09_ablation_wearables.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))

from phonefm_dataset_v3 import make_loader_v3
from phonefm_model_v3 import PhoneFMV3, PhoneFMV3Config


# ===========================================================================
# Config
# ===========================================================================

DATA_DIR = Path("/home/jupyter/workspace/phonefm-data/tokenized_v3")
MODEL_DIR = Path("/home/jupyter/workspace/phonefm-data/phonefm_v3")
OUT_PATH = MODEL_DIR / "ablation_wearables.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 20260611

HEADS = ("cv_composite_30d", "cv_composite_180d", "cv_composite_365d")
HEADLINE = "cv_composite_30d"
N_PERM = 10            # independent random derangements per batch
N_BOOTSTRAP = 2000     # person-level cluster bootstrap on the importance delta
CI = 0.95
REFERENCE_AUROC_CV30D = 0.8856857329969465  # CPU fp32 baseline


def auroc_masked(pred: np.ndarray, y: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, pred))


def random_derangement(n: int, g: torch.Generator) -> torch.Tensor:
    """A permutation of range(n) with no fixed points (so every window's
    wearables are reassigned to a different window).  n>=2 expected."""
    if n < 2:
        return torch.arange(n)
    while True:
        p = torch.randperm(n, generator=g)
        if not torch.any(p == torch.arange(n)):
            return p


@torch.no_grad()
def main() -> None:
    print(f"device={DEVICE}", flush=True)
    if DEVICE != "cuda":
        print("WARNING: CPU run does ~12 forward passes/batch and will be very slow; "
              "GPU strongly recommended.", flush=True)

    # ---- Load model
    with open(MODEL_DIR / "config.json") as f:
        saved = json.load(f)
    cfg_d = saved["config"].copy()
    cfg_d["head_specs"] = tuple(tuple(s) for s in cfg_d["head_specs"])
    cfg = PhoneFMV3Config(**cfg_d)
    model = PhoneFMV3(cfg).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location=DEVICE), strict=True)
    model.eval()
    print(f"model loaded: {model.num_params() / 1e6:.1f}M params", flush=True)

    # ---- Loader: shuffle=True so each batch mixes persons (verified that the
    #      dataset index is global across shards, so the within-batch derangement
    #      reassigns wearables across persons).  Seed for reproducibility.
    torch.manual_seed(SEED)
    test_glob = str(DATA_DIR / "test_*.parquet")
    loader = make_loader_v3(
        test_glob, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True,
        max_seq_len=cfg.max_seq_len, n_confounders=cfg.n_confounders,
    )
    print(f"loader: {len(loader)} batches, {len(loader.dataset)} windows", flush=True)

    g = torch.Generator()
    g.manual_seed(SEED)

    preds_full = {h: [] for h in HEADS}
    preds_perm = {h: [[] for _ in range(N_PERM)] for h in HEADS}
    preds_zero = {h: [] for h in HEADS}
    labels = {h: [] for h in HEADS}
    masks = {h: [] for h in HEADS}
    person_ids: list[np.ndarray] = []
    same_person_neighbor = 0
    total_pairs = 0

    def fwd(wf: torch.Tensor) -> dict:
        logits = model(input_ids, token_types, day_index, wf, confounders, attn_mask)
        return {h: torch.sigmoid(logits[h].float()).cpu().numpy() for h in HEADS}

    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE)
        token_types = batch["token_types"].to(DEVICE)
        day_index = batch["day_index"].to(DEVICE)
        wearable = batch["wearable_feats"].to(DEVICE)
        confounders = batch["confounders"].to(DEVICE)
        attn_mask = batch["attn_mask"].to(DEVICE)
        B = wearable.shape[0]
        pid = batch["person_id"].numpy()
        person_ids.append(pid)

        # full
        of = fwd(wearable)
        for h in HEADS:
            preds_full[h].append(of[h])
        # K permutations
        for k in range(N_PERM):
            d = random_derangement(B, g)
            same_person_neighbor += int((pid[d.numpy()] == pid).sum())
            total_pairs += B
            op = fwd(wearable[d.to(DEVICE)])
            for h in HEADS:
                preds_perm[h][k].append(op[h])
        # zero
        oz = fwd(torch.zeros_like(wearable))
        for h in HEADS:
            preds_zero[h].append(oz[h])

        for h in HEADS:
            labels[h].append(batch["labels"][h].numpy())
            masks[h].append(batch["masks"][h].numpy())
        if i % 100 == 0:
            print(f"  batch {i}/{len(loader)}", flush=True)

    # ---- Concatenate
    pid_all = np.concatenate(person_ids)
    lab = {h: np.concatenate(labels[h]).astype(np.int64) for h in HEADS}
    msk = {h: np.concatenate(masks[h]).astype(bool) for h in HEADS}
    full = {h: np.concatenate(preds_full[h]).astype(np.float64).reshape(-1) for h in HEADS}
    zero = {h: np.concatenate(preds_zero[h]).astype(np.float64).reshape(-1) for h in HEADS}
    perm = {h: [np.concatenate(preds_perm[h][k]).astype(np.float64).reshape(-1) for k in range(N_PERM)] for h in HEADS}

    frac_same = same_person_neighbor / max(total_pairs, 1)
    print(f"\nsame-person neighbor fraction across permutations: {frac_same:.4f} "
          f"(want ~0; high → shuffle not mixing persons)", flush=True)

    # ---- AUROC per head per condition + importance
    summary: dict = {
        "config": {"seed": SEED, "n_perm": N_PERM, "n_bootstrap": N_BOOTSTRAP, "ci": CI,
                   "device": DEVICE, "precision": "fp32", "bootstrap_unit": "person (cluster)",
                   "reference_auroc_cv30d_cpu_fp32": REFERENCE_AUROC_CV30D,
                   "same_person_neighbor_frac": frac_same,
                   "interpretation": "marginal AUROC importance of wearable VALUES conditional on EHR+confounders; "
                                     "NOT sufficiency; permutation delta is the defensible number, zero is presence/OOD cross-check"},
        "heads": {},
    }
    print(f"\n{'head':<22} {'n_valid':>8} {'n_pos':>6}  {'full':>7} "
          f"{'perm_mean':>10} {'perm_sd':>8} {'zero':>7}  {'imp(full-perm)':>15}", flush=True)
    for h in HEADS:
        m = msk[h]
        y = lab[h][m]
        n_valid, n_pos = int(m.sum()), int(y.sum())
        a_full = auroc_masked(full[h][m], y)
        a_perm_k = [auroc_masked(perm[h][k][m], y) for k in range(N_PERM)]
        a_zero = auroc_masked(zero[h][m], y)
        perm_mean = float(np.nanmean(a_perm_k))
        perm_sd = float(np.nanstd(a_perm_k))
        imp = a_full - perm_mean
        zero_imp = a_full - a_zero
        print(f"{h:<22} {n_valid:>8} {n_pos:>6}  {a_full:>7.4f} {perm_mean:>10.4f} "
              f"{perm_sd:>8.4f} {a_zero:>7.4f}  {imp:>+15.4f}", flush=True)
        summary["heads"][h] = {
            "n_valid": n_valid, "n_pos": n_pos, "auroc_full": a_full,
            "auroc_perm_mean": perm_mean, "auroc_perm_sd": perm_sd,
            "auroc_perm_per_k": a_perm_k, "auroc_zero": a_zero,
            "wearable_importance_perm": imp, "permutation_sd": perm_sd,
            "wearable_importance_zero": zero_imp,
            "perm_vs_zero_gap": imp - zero_imp,
        }

    # ---- Person-level cluster bootstrap on the HEADLINE importance delta.
    #      Resample PERSONS with replacement; recompute (full - perm_0) AUROC on
    #      their pooled windows.  Uses permutation 0 for speed; permutation_sd
    #      above separately captures between-permutation variance.
    h = HEADLINE
    m = msk[h]
    idx_all = np.where(m)[0]
    y_all = lab[h][idx_all]
    pf = full[h][idx_all]
    pp0 = perm[h][0][idx_all]
    pids_masked = pid_all[idx_all]
    # group masked-window positions by person
    order = np.argsort(pids_masked, kind="stable")
    uniq, starts = np.unique(pids_masked[order], return_index=True)
    groups = np.split(order, starts[1:])  # list of index-arrays into idx_all space
    rng = np.random.RandomState(SEED)
    deltas = np.full(N_BOOTSTRAP, np.nan)
    n_persons = len(uniq)
    for b in range(N_BOOTSTRAP):
        pick = rng.randint(0, n_persons, n_persons)
        sel = np.concatenate([groups[j] for j in pick])
        yb = y_all[sel]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        deltas[b] = roc_auc_score(yb, pf[sel]) - roc_auc_score(yb, pp0[sel])
    n_boot_valid = int(np.count_nonzero(~np.isnan(deltas)))
    if n_boot_valid >= 100:
        lo = float(np.nanpercentile(deltas, (1 - CI) / 2 * 100))
        hi = float(np.nanpercentile(deltas, (1 + CI) / 2 * 100))
    else:
        lo = hi = float("nan")
    point = summary["heads"][h]["wearable_importance_perm"]
    summary["heads"][h]["importance_cluster_ci"] = {"lo": lo, "hi": hi, "n_bootstrap_valid": n_boot_valid}
    summary["n_persons"] = n_persons

    print(f"\n=== {HEADLINE} WEARABLE IMPORTANCE ===", flush=True)
    print(f"  importance (full - perm_mean) = {point:+.4f}", flush=True)
    print(f"  permutation SD (across {N_PERM} shuffles) = {summary['heads'][h]['permutation_sd']:.4f}", flush=True)
    print(f"  person-cluster 95% CI = [{lo:+.4f}, {hi:+.4f}]  (n_boot_valid={n_boot_valid}, n_persons={n_persons})", flush=True)
    print(f"  {'CI excludes 0 → significant' if lo > 0 else 'CI includes 0'}", flush=True)
    print(f"  zero-ablation importance (cross-check) = {summary['heads'][h]['wearable_importance_zero']:+.4f} "
          f"(perm-vs-zero gap {summary['heads'][h]['perm_vs_zero_gap']:+.4f})", flush=True)
    gap = abs(summary["heads"][h]["auroc_full"] - REFERENCE_AUROC_CV30D)
    print(f"  full AUROC (GPU fp32) = {summary['heads'][h]['auroc_full']:.6f}  "
          f"vs CPU fp32 reference {REFERENCE_AUROC_CV30D:.6f}  (|Δ|={gap:.2e}, "
          f"{'OK' if gap < 5e-3 else 'WARN: setup mismatch'})", flush=True)

    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_san(x) for x in o]
        if isinstance(o, float) and np.isnan(o):
            return None
        return o

    with open(OUT_PATH, "w") as f:
        json.dump(_san(summary), f, indent=2)
    print(f"\nsaved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
