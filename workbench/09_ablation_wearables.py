"""
09_ablation_wearables.py — quantify, in AUROC terms, both the IMPORTANCE and
the SUFFICIENCY of the wearable signal for cardiovascular prediction, by input
ablation at inference (no retraining).  Single GPU pass, fp32.

Input layout (from collate_v3): position 0 = CLS; positions 1..180 = wearable
day tokens (token_types==3); positions 181.. = EHR event tokens
(token_types>=4: DX/MED/PX/LAB); then EOS; then PAD.  attn_mask gates which
positions the transformer attends to.  Confounders (age/sex/BMI/...) are added
to the CLS hidden state.

Conditions evaluated per batch (cv_composite heads):
  FULL       : inputs unchanged.
  WEAR_PERM  : wearable_feats permuted across windows (K random derangements in
               each shuffled batch → each window's wearables go to a different
               person), EHR + confounders intact.  IMPORTANCE: full - perm =
               marginal AUROC value of correctly-paired wearables GIVEN EHR.
  WEAR_ZERO  : wearable_feats zeroed (BatchNorm eval → constant embedding).
               Presence/structure cross-check for the permutation (mildly OOD).
  WEAR_DEMO  : EHR event tokens masked out (attn_mask &= token_types<4),
               confounders KEPT.  SUFFICIENCY (deployment): wearables + self-
               reported demographics, no hospital EHR — the realistic on-device
               scenario.  Report AUROC directly.
  WEAR_ONLY  : EHR masked AND confounders zeroed.  Pure wearable physiology
               floor.  Report AUROC directly.

Read carefully (review caveats, load-bearing):
  - IMPORTANCE (full - perm) is the marginal value of wearable VALUES
    CONDITIONAL on EHR + confounders.  A small number can mean "wearables add
    little" OR "wearables are redundant with EHR which compensates" — the
    importance ablation alone cannot separate these.  The SUFFICIENCY numbers
    (wear_demo / wear_only) disambiguate: if wearables alone are still strong,
    the signal is real regardless of redundancy.
  - The PERMUTATION delta (in-distribution) is the defensible importance; ZERO
    also removes wearable presence/structure and is mildly OOD.  The
    perm-vs-zero gap is logged as a diagnostic.

Uncertainty (importance, headline cv_composite_30d):
  - permutation SD across the K derangements (permutation variance), and
  - person-level CLUSTER bootstrap 95% CI (resamples persons, matching the
    ~44-windows/person clustering; window-level would be anti-conservative).

fp32 on GPU so FULL AUROC reproduces the 0.8857 CPU baseline (sanity) and the
delta carries no bf16 ranking noise.

In:  ~/workspace/phonefm-data/tokenized_v3/test_*.parquet, .../phonefm_v3/best.pt + config.json
Out: ~/workspace/phonefm-data/phonefm_v3/ablation_wearables.json
Run (GPU pod):  cd ~/repos/PhoneFM/workbench && python3 09_ablation_wearables.py
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
EHR_TYPE_MIN = 4           # token_types >= 4 are EHR event tokens (DX/MED/PX/LAB)
N_PERM = 10
N_BOOTSTRAP = 2000
CI = 0.95
REFERENCE_AUROC_CV30D = 0.8856857329969465  # CPU fp32 baseline

# Conditions whose AUROC we report (perm handled separately as a K-list).
SINGLE_CONDS = ("full", "wear_zero", "wear_demo", "wear_only")


def auroc_masked(pred: np.ndarray, y: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, pred))


def random_derangement(n: int, g: torch.Generator) -> torch.Tensor:
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
        print("WARNING: CPU run does ~14 forward passes/batch — very slow; GPU strongly recommended.", flush=True)

    with open(MODEL_DIR / "config.json") as f:
        saved = json.load(f)
    cfg_d = saved["config"].copy()
    cfg_d["head_specs"] = tuple(tuple(s) for s in cfg_d["head_specs"])
    cfg = PhoneFMV3Config(**cfg_d)
    model = PhoneFMV3(cfg).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location=DEVICE), strict=True)
    model.eval()
    print(f"model loaded: {model.num_params() / 1e6:.1f}M params", flush=True)

    torch.manual_seed(SEED)
    loader = make_loader_v3(
        str(DATA_DIR / "test_*.parquet"), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        shuffle=True, max_seq_len=cfg.max_seq_len, n_confounders=cfg.n_confounders,
    )
    print(f"loader: {len(loader)} batches, {len(loader.dataset)} windows", flush=True)
    g = torch.Generator(); g.manual_seed(SEED)

    preds = {c: {h: [] for h in HEADS} for c in SINGLE_CONDS}
    preds_perm = {h: [[] for _ in range(N_PERM)] for h in HEADS}
    labels = {h: [] for h in HEADS}
    masks = {h: [] for h in HEADS}
    person_ids: list[np.ndarray] = []
    same_person_neighbor = 0
    total_pairs = 0

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

        zero_wear = torch.zeros_like(wearable)
        zero_conf = torch.zeros_like(confounders)
        attn_noehr = attn_mask & (token_types < EHR_TYPE_MIN)  # mask EHR event tokens

        def fwd(wf, conf, amask):
            logits = model(input_ids, token_types, day_index, wf, conf, amask)
            return {h: torch.sigmoid(logits[h].float()).cpu().numpy() for h in HEADS}

        out = {
            "full":      fwd(wearable, confounders, attn_mask),
            "wear_zero": fwd(zero_wear, confounders, attn_mask),
            "wear_demo": fwd(wearable, confounders, attn_noehr),   # EHR masked, demo kept
            "wear_only": fwd(wearable, zero_conf, attn_noehr),     # EHR masked, demo zeroed
        }
        for c in SINGLE_CONDS:
            for h in HEADS:
                preds[c][h].append(out[c][h])

        for k in range(N_PERM):
            d = random_derangement(B, g)
            same_person_neighbor += int((pid[d.numpy()] == pid).sum()); total_pairs += B
            op = fwd(wearable[d.to(DEVICE)], confounders, attn_mask)
            for h in HEADS:
                preds_perm[h][k].append(op[h])

        for h in HEADS:
            labels[h].append(batch["labels"][h].numpy())
            masks[h].append(batch["masks"][h].numpy())
        if i % 100 == 0:
            print(f"  batch {i}/{len(loader)}", flush=True)

    # ---- Concatenate
    pid_all = np.concatenate(person_ids)
    lab = {h: np.concatenate(labels[h]).astype(np.int64) for h in HEADS}
    msk = {h: np.concatenate(masks[h]).astype(bool) for h in HEADS}
    P = {c: {h: np.concatenate(preds[c][h]).astype(np.float64).reshape(-1) for h in HEADS} for c in SINGLE_CONDS}
    perm = {h: [np.concatenate(preds_perm[h][k]).astype(np.float64).reshape(-1) for k in range(N_PERM)] for h in HEADS}

    frac_same = same_person_neighbor / max(total_pairs, 1)
    print(f"\nsame-person neighbor fraction across permutations: {frac_same:.4f} (want ~0)", flush=True)

    summary: dict = {
        "config": {"seed": SEED, "n_perm": N_PERM, "n_bootstrap": N_BOOTSTRAP, "ci": CI,
                   "device": DEVICE, "precision": "fp32", "bootstrap_unit": "person (cluster)",
                   "reference_auroc_cv30d_cpu_fp32": REFERENCE_AUROC_CV30D,
                   "same_person_neighbor_frac": frac_same,
                   "conditions": "full; wear_perm(importance); wear_zero(presence x-check); "
                                 "wear_demo(EHR masked, demographics kept = deployment sufficiency); "
                                 "wear_only(EHR masked, demographics zeroed = pure-wearable floor)"},
        "heads": {},
    }
    hdr = (f"{'head':<20} {'n_pos':>5}  {'full':>7} {'perm_mu':>8} {'perm_sd':>7} "
           f"{'zero':>7} {'wear+demo':>9} {'wear_only':>9}  {'import(f-perm)':>14}")
    print("\n" + hdr, flush=True)
    for h in HEADS:
        m = msk[h]; y = lab[h][m]; n_pos = int(y.sum())
        a_full = auroc_masked(P["full"][h][m], y)
        a_perm_k = [auroc_masked(perm[h][k][m], y) for k in range(N_PERM)]
        a_zero = auroc_masked(P["wear_zero"][h][m], y)
        a_demo = auroc_masked(P["wear_demo"][h][m], y)
        a_only = auroc_masked(P["wear_only"][h][m], y)
        perm_mu = float(np.nanmean(a_perm_k)); perm_sd = float(np.nanstd(a_perm_k))
        imp = a_full - perm_mu
        print(f"{h:<20} {n_pos:>5}  {a_full:>7.4f} {perm_mu:>8.4f} {perm_sd:>7.4f} "
              f"{a_zero:>7.4f} {a_demo:>9.4f} {a_only:>9.4f}  {imp:>+14.4f}", flush=True)
        summary["heads"][h] = {
            "n_valid": int(m.sum()), "n_pos": n_pos,
            "auroc_full": a_full, "auroc_perm_mean": perm_mu, "auroc_perm_sd": perm_sd,
            "auroc_perm_per_k": a_perm_k, "auroc_wear_zero": a_zero,
            "auroc_wear_demo_sufficiency": a_demo, "auroc_wear_only_floor": a_only,
            "wearable_importance_perm": imp,
            "wearable_importance_zero": a_full - a_zero,
            "perm_vs_zero_gap": (a_full - perm_mu) - (a_full - a_zero),
        }

    # ---- Person-cluster bootstrap on the HEADLINE importance delta (perm_0)
    h = HEADLINE; m = msk[h]
    idx = np.where(m)[0]
    y_all = lab[h][idx]; pf = P["full"][h][idx]; pp0 = perm[h][0][idx]; pidm = pid_all[idx]
    order = np.argsort(pidm, kind="stable")
    uniq, starts = np.unique(pidm[order], return_index=True)
    groups = np.split(order, starts[1:])
    rng = np.random.RandomState(SEED)
    deltas = np.full(N_BOOTSTRAP, np.nan)
    for b in range(N_BOOTSTRAP):
        sel = np.concatenate([groups[j] for j in rng.randint(0, len(uniq), len(uniq))])
        yb = y_all[sel]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        deltas[b] = roc_auc_score(yb, pf[sel]) - roc_auc_score(yb, pp0[sel])
    n_bv = int(np.count_nonzero(~np.isnan(deltas)))
    lo = float(np.nanpercentile(deltas, (1 - CI) / 2 * 100)) if n_bv >= 100 else float("nan")
    hi = float(np.nanpercentile(deltas, (1 + CI) / 2 * 100)) if n_bv >= 100 else float("nan")
    summary["heads"][h]["importance_cluster_ci"] = {"lo": lo, "hi": hi, "n_bootstrap_valid": n_bv}
    summary["n_persons"] = len(uniq)

    s = summary["heads"][h]
    print(f"\n=== {HEADLINE} ===", flush=True)
    print(f"  IMPORTANCE (full - perm)        = {s['wearable_importance_perm']:+.4f}  "
          f"(perm SD {s['auroc_perm_sd']:.4f}; person-cluster 95% CI [{lo:+.4f}, {hi:+.4f}], n_boot={n_bv})  "
          f"{'sig' if lo > 0 else 'CI incl 0'}", flush=True)
    print(f"  SUFFICIENCY wear+demographics   = {s['auroc_wear_demo_sufficiency']:.4f}  "
          f"(EHR removed; deployment-relevant)", flush=True)
    print(f"  SUFFICIENCY wearables-only floor= {s['auroc_wear_only_floor']:.4f}  "
          f"(EHR + demographics removed)", flush=True)
    print(f"  full vs perm vs zero            = {s['auroc_full']:.4f} / {s['auroc_perm_mean']:.4f} / "
          f"{s['auroc_wear_zero']:.4f}  (perm-vs-zero gap {s['perm_vs_zero_gap']:+.4f})", flush=True)
    gap = abs(s["auroc_full"] - REFERENCE_AUROC_CV30D)
    print(f"  full AUROC (GPU fp32) {s['auroc_full']:.6f} vs CPU ref {REFERENCE_AUROC_CV30D:.6f}  "
          f"(|Δ|={gap:.2e}, {'OK' if gap < 5e-3 else 'WARN setup mismatch'})", flush=True)

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
