"""
10_ablation_streams.py — per-stream wearable ablation: decompose 09's whole-
wearable importance into STEPS vs HEART-RATE vs SLEEP, in AUROC terms, by
permutation at inference (no retraining).  Single GPU pass, fp32.  Faithful
extension of 09_ablation_wearables.py: SAME full context (EHR + confounders
intact), SAME permutation mechanism, SAME person-cluster bootstrap, SAME
reproduce-baseline guard — only the permuted columns differ.  Read 09 first.

Wearable layout (code-grounded; see 02_tokenizer_v3.py encode_window_v3 and
collate_v3).  Each wearable day token (positions 1..180, token_types==3) carries
an 11-dim feature vector wearable_feats[:, pos, :].  Columns map to streams:
    STEPS     -> [0]                 (daily total steps)
    HEART-RATE-> [1, 2, 3, 4]        (mean / resting(10pct) / max(95pct) / SDANN)
    SLEEP     -> [5, 6, 7, 8, 9, 10] (REM% / deep% / light% / duration / onset /
                                      7-day duration variability)
EHR positions (181..) carry all-zero wearable_feats, so column ops there are
no-ops; the model only consumes wearable_feats where token_types==3.

Conditions (cv_composite heads), all in the FULL context (EHR + confounders
kept — this is 09's importance context, just decomposed by stream):
  FULL      : inputs unchanged (shared baseline; reproduces the 0.8857 ref).
  PERM_ALL  : permute ALL 11 wearable columns across windows (K derangements).
              Reproduces 09's wear_perm; whole-wearable importance reference so
              the per-stream pieces are stated against the SAME number this
              pipeline computes (one-pipeline-one-figure), not 09's separately.
  PERM_S    : permute ONLY stream S's columns across windows (K derangements),
              EHR + confounders + the OTHER two wearable streams intact.
              IMPORTANCE_S = full - perm_S = the UNIQUE marginal AUROC value of
              stream S given everything else.  This is leave-one-stream-out.

WHY PERMUTATION, NOT ZEROING (load-bearing — see 09 review):
  Zeroing a column does NOT mean "stream absent": missing wearable days and all
  EHR/CLS/PAD positions carry zero wearable_feats during training, so after the
  BatchNorm the zero maps to a value the model has LEARNED to read as "no data
  today".  A zeroed-stream condition therefore measures the model's missing-data
  behaviour, not a clean counterfactual.  Permutation keeps every column on its
  real marginal distribution while destroying its association with the label —
  the defensible, in-distribution importance estimator.  This script is
  permutation-only by design.

INTERPRETATION CAVEATS (do not over-read the per-stream ranking):
  - Streams are mutually redundant (HR variability tracks sleep, steps track
    both) AND redundant with EHR.  So each IMPORTANCE_S is a LOWER BOUND (the
    retained channels partly reconstruct S), and the three do NOT sum to the
    whole-wearable importance.  A near-zero IMPORTANCE_S means "redundant given
    the rest", NOT "no signal".
  - Differential redundancy means cross-stream comparison is confounded: a
    stream that is less redundant with what's retained shows a larger leave-one-
    out delta even at equal raw signal.  Steps is 1 column; HR is 4; sleep is 6
    — after per-column BatchNorm, permuting more columns injects more
    perturbation, so the steps-vs-sleep delta also carries a column-COUNT
    confound.  We therefore report pairwise difference CIs but flag that any
    "stream X dominates" claim is a statement about UNIQUE contribution in this
    model, not about physiological signal magnitude.
  - These leave-one-out numbers are in the FULL (EHR-present) context.  The
    phone-deployable (EHR-masked) per-stream picture is a natural follow-up and
    will generally show larger per-stream values (no EHR to compensate).

Uncertainty (per stream, headline cv_composite_30d):
  - permutation SD across the K derangements, and
  - person-level CLUSTER bootstrap 95% CI on full - perm_S(k=0), plus paired
    pairwise-difference CIs between streams (shared resamples) so cross-stream
    contrasts are not mis-derived from independent marginal CIs.

Robustness (history: a prior run discarded a multi-hour forward pass to a post-
hoc bug):
  - raw per-window predictions are written to .npz IMMEDIATELY after the
    accumulation loop, BEFORE any AUROC/bootstrap, so the forward pass can never
    be lost to a downstream error;
  - the batch loop is wrapped so a mid-run failure dumps partial accumulators;
  - the FULL-vs-reference reproduction check is a HARD ASSERT before the JSON is
    written — a setup/precision mismatch aborts instead of writing wrong deltas.

In:  ~/workspace/phonefm-data/tokenized_v3/test_*.parquet, .../phonefm_v3/best.pt + config.json
Out: ~/workspace/phonefm-data/phonefm_v3/ablation_streams.json
     ~/workspace/phonefm-data/phonefm_v3/ablation_streams_rawpreds.npz (write-before-verify)
Run (GPU pod):  cd ~/repos/PhoneFM/workbench && python3 10_ablation_streams.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback
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
OUT_PATH = MODEL_DIR / "ablation_streams.json"
RAW_PATH = MODEL_DIR / "ablation_streams_rawpreds.npz"
PARTIAL_PATH = MODEL_DIR / "ablation_streams_rawpreds.PARTIAL.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 20260611

HEADS = ("cv_composite_30d", "cv_composite_180d", "cv_composite_365d")
HEADLINE = "cv_composite_30d"
N_WEARABLE_FEATURES = 11   # must match cfg.n_wearable_features; asserted below
N_PERM = 10
N_BOOTSTRAP = 2000
CI = 0.95
REFERENCE_AUROC_CV30D = 0.8856857329969465  # CPU fp32 baseline (same as 09)
REPRO_TOL = 5e-3

# Per-stream column indices into wearable_feats[..., :11] (see 02_tokenizer_v3.py).
STREAMS: dict[str, list[int]] = {
    "steps": [0],
    "hr":    [1, 2, 3, 4],
    "sleep": [5, 6, 7, 8, 9, 10],
}
# "all" = whole-wearable reference (reproduces 09's wear_perm).
PERM_KEYS = ["all"] + list(STREAMS.keys())
PERM_COLS: dict[str, list[int]] = {"all": list(range(N_WEARABLE_FEATURES)), **STREAMS}


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
        print("WARNING: CPU run does ~41 forward passes/batch — very slow; GPU strongly recommended.", flush=True)

    with open(MODEL_DIR / "config.json") as f:
        saved = json.load(f)
    cfg_d = saved["config"].copy()
    cfg_d["head_specs"] = tuple(tuple(s) for s in cfg_d["head_specs"])
    cfg = PhoneFMV3Config(**cfg_d)
    model = PhoneFMV3(cfg).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location=DEVICE), strict=True)
    model.eval()
    print(f"model loaded: {model.num_params() / 1e6:.1f}M params", flush=True)

    # Guard: the hard-coded column map must match the model's feature count and
    # partition every column exactly once.  A silent drift here would ablate the
    # wrong physiology, so fail loudly before any compute.
    assert cfg.n_wearable_features == N_WEARABLE_FEATURES, (
        f"cfg.n_wearable_features={cfg.n_wearable_features} != {N_WEARABLE_FEATURES}; "
        "stream column map is stale — refusing to run."
    )
    all_idx = sorted(c for cols in STREAMS.values() for c in cols)
    assert all_idx == list(range(N_WEARABLE_FEATURES)), (
        f"STREAMS columns {all_idx} do not partition 0..{N_WEARABLE_FEATURES - 1} exactly once."
    )

    torch.manual_seed(SEED)
    loader = make_loader_v3(
        str(DATA_DIR / "test_*.parquet"), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        shuffle=True, max_seq_len=cfg.max_seq_len, n_confounders=cfg.n_confounders,
    )
    print(f"loader: {len(loader)} batches, {len(loader.dataset)} windows", flush=True)
    g = torch.Generator(); g.manual_seed(SEED)

    col_t = {k: torch.tensor(PERM_COLS[k], dtype=torch.long, device=DEVICE) for k in PERM_KEYS}

    # accumulators
    preds_full = {h: [] for h in HEADS}
    # preds_perm[perm_key][h][k] -> list of per-batch arrays
    preds_perm = {pk: {h: [[] for _ in range(N_PERM)] for h in HEADS} for pk in PERM_KEYS}
    labels = {h: [] for h in HEADS}
    masks = {h: [] for h in HEADS}
    person_ids: list[np.ndarray] = []
    same_person_neighbor = 0
    total_pairs = 0
    n_batches_done = 0

    def dump_partial(reason: str) -> None:
        # Best-effort: persist whatever forward passes completed so an aborted
        # run is not a total loss.  Concatenate per-condition; tolerate empties.
        try:
            arrs: dict[str, np.ndarray] = {}
            if person_ids:
                arrs["person_id"] = np.concatenate(person_ids)
            for h in HEADS:
                if preds_full[h]:
                    arrs[f"full__{h}"] = np.concatenate(preds_full[h]).reshape(-1)
                if labels[h]:
                    arrs[f"label__{h}"] = np.concatenate(labels[h]).reshape(-1)
                    arrs[f"mask__{h}"] = np.concatenate(masks[h]).reshape(-1)
                for pk in PERM_KEYS:
                    for k in range(N_PERM):
                        if preds_perm[pk][h][k]:
                            arrs[f"perm_{pk}_k{k}__{h}"] = np.concatenate(preds_perm[pk][h][k]).reshape(-1)
            np.savez_compressed(PARTIAL_PATH, reason=reason, n_batches=n_batches_done, **arrs)
            print(f"  [partial dump] {reason}: wrote {PARTIAL_PATH} after {n_batches_done} batches", flush=True)
        except Exception as e:  # never let the dump mask the original error
            print(f"  [partial dump FAILED] {e}", flush=True)

    try:
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(DEVICE)
            token_types = batch["token_types"].to(DEVICE)
            day_index = batch["day_index"].to(DEVICE)
            wearable = batch["wearable_feats"].to(DEVICE)   # (B, L, 11)
            confounders = batch["confounders"].to(DEVICE)
            attn_mask = batch["attn_mask"].to(DEVICE)
            B = wearable.shape[0]
            pid = batch["person_id"].numpy()
            person_ids.append(pid)

            def fwd(wf):
                # EHR + confounders + attn_mask are always the originals here;
                # this script only perturbs wearable feature columns.
                logits = model(input_ids, token_types, day_index, wf, confounders, attn_mask)
                return {h: torch.sigmoid(logits[h].float()).cpu().numpy() for h in HEADS}

            out_full = fwd(wearable)
            for h in HEADS:
                preds_full[h].append(out_full[h])

            # one shared derangement list per batch, reused across ALL perm keys
            # so the person-mixing is identical for every stream (fair leave-one-
            # out comparison) and the neighbor diagnostic is counted once.
            derangements = []
            for k in range(N_PERM):
                d = random_derangement(B, g)
                derangements.append(d.to(DEVICE))
                same_person_neighbor += int((pid[d.numpy()] == pid).sum()); total_pairs += B

            for pk in PERM_KEYS:
                own = col_t[pk]
                for k in range(N_PERM):
                    wf_perm = wearable.clone()
                    wf_perm[:, :, own] = wearable[derangements[k]][:, :, own]
                    op = fwd(wf_perm)
                    for h in HEADS:
                        preds_perm[pk][h][k].append(op[h])

            for h in HEADS:
                labels[h].append(batch["labels"][h].numpy())
                masks[h].append(batch["masks"][h].numpy())
            n_batches_done = i + 1
            if i % 50 == 0:
                print(f"  batch {i}/{len(loader)}", flush=True)
    except BaseException as e:  # incl. KeyboardInterrupt / CUDA OOM
        print(f"\nABORTED at batch {n_batches_done}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        dump_partial(f"{type(e).__name__}: {e}")
        raise

    # ---- Concatenate
    pid_all = np.concatenate(person_ids)
    lab = {h: np.concatenate(labels[h]).astype(np.int64) for h in HEADS}
    msk = {h: np.concatenate(masks[h]).astype(bool) for h in HEADS}
    PF = {h: np.concatenate(preds_full[h]).astype(np.float64).reshape(-1) for h in HEADS}
    PP = {pk: {h: [np.concatenate(preds_perm[pk][h][k]).astype(np.float64).reshape(-1) for k in range(N_PERM)]
               for h in HEADS} for pk in PERM_KEYS}

    # ---- WRITE-BEFORE-VERIFY: persist raw preds before any AUROC/bootstrap so a
    # downstream bug can never discard the (expensive) forward pass.
    raw: dict[str, np.ndarray] = {"person_id": pid_all}
    for h in HEADS:
        raw[f"full__{h}"] = PF[h]
        raw[f"label__{h}"] = lab[h]
        raw[f"mask__{h}"] = msk[h]
        for pk in PERM_KEYS:
            for k in range(N_PERM):
                raw[f"perm_{pk}_k{k}__{h}"] = PP[pk][h][k]
    np.savez_compressed(RAW_PATH, **raw)
    print(f"\n[write-before-verify] saved raw preds {RAW_PATH}", flush=True)
    if PARTIAL_PATH.exists():
        try:
            PARTIAL_PATH.unlink()  # full run completed; drop any stale partial
        except OSError:
            pass

    frac_same = same_person_neighbor / max(total_pairs, 1)
    print(f"same-person neighbor fraction across permutations: {frac_same:.4f} (want ~0)", flush=True)

    # ---- HARD reproduction guard BEFORE computing/writing deltas.
    a_full_headline = auroc_masked(PF[HEADLINE][msk[HEADLINE]], lab[HEADLINE][msk[HEADLINE]])
    gap = abs(a_full_headline - REFERENCE_AUROC_CV30D)
    print(f"full AUROC (GPU fp32) {a_full_headline:.6f} vs CPU ref {REFERENCE_AUROC_CV30D:.6f} (|Δ|={gap:.2e})", flush=True)
    assert gap < REPRO_TOL, (
        f"FULL AUROC {a_full_headline:.6f} does not reproduce CPU ref {REFERENCE_AUROC_CV30D:.6f} "
        f"(|Δ|={gap:.2e} >= {REPRO_TOL}); refusing to write per-stream deltas computed against a "
        f"wrong baseline.  Raw preds are in {RAW_PATH} for offline inspection."
    )

    summary: dict = {
        "config": {"seed": SEED, "n_perm": N_PERM, "n_bootstrap": N_BOOTSTRAP, "ci": CI,
                   "device": DEVICE, "precision": "fp32", "context": "full (EHR + confounders kept)",
                   "bootstrap_unit": "person (cluster)",
                   "reference_auroc_cv30d_cpu_fp32": REFERENCE_AUROC_CV30D,
                   "same_person_neighbor_frac": frac_same,
                   "streams": {s: STREAMS[s] for s in STREAMS},
                   "estimator": "permutation leave-one-stream-out; full - perm_S = UNIQUE marginal "
                                "value of stream S given EHR + confounders + other wearable streams. "
                                "Lower bound (redundancy); per-stream importances do NOT sum to "
                                "whole-wearable (perm_all). Cross-stream ranking confounded by "
                                "differential redundancy and column count — see paired-diff CIs."},
        "heads": {},
    }

    stream_names = list(STREAMS.keys())
    hdr = (f"{'head':<20} {'key':<6} {'n_pos':>5}  {'full':>7} {'perm_mu':>8} {'perm_sd':>7}  "
           f"{'import(f-perm)':>14}")
    print("\n" + hdr, flush=True)
    full_auroc = {}
    for h in HEADS:
        m = msk[h]; y = lab[h][m]; n_pos = int(y.sum())
        a_full = auroc_masked(PF[h][m], y)
        full_auroc[h] = a_full
        summary["heads"][h] = {"n_valid": int(m.sum()), "n_pos": n_pos, "auroc_full": a_full, "perm": {}}
        for pk in PERM_KEYS:
            a_perm_k = [auroc_masked(PP[pk][h][k][m], y) for k in range(N_PERM)]
            perm_mu = float(np.nanmean(a_perm_k)); perm_sd = float(np.nanstd(a_perm_k))
            imp = a_full - perm_mu
            print(f"{h:<20} {pk:<6} {n_pos:>5}  {a_full:>7.4f} {perm_mu:>8.4f} {perm_sd:>7.4f}  {imp:>+14.4f}",
                  flush=True)
            summary["heads"][h]["perm"][pk] = {
                "auroc_perm_mean": perm_mu, "auroc_perm_sd": perm_sd, "auroc_perm_per_k": a_perm_k,
                "importance_perm": imp,
            }

    # ---- Person-cluster bootstrap on the HEADLINE: per-key importance (perm_0)
    # and PAIRED pairwise stream-difference CIs (shared resamples).
    h = HEADLINE; m = msk[h]
    idx = np.where(m)[0]
    y_all = lab[h][idx]; pf = PF[h][idx]; pidm = pid_all[idx]
    order = np.argsort(pidm, kind="stable")
    uniq, starts = np.unique(pidm[order], return_index=True)
    groups = np.split(order, starts[1:])
    rng = np.random.RandomState(SEED)
    sels = [np.concatenate([groups[j] for j in rng.randint(0, len(uniq), len(uniq))])
            for _ in range(N_BOOTSTRAP)]

    # per-key importance arrays across bootstrap (full - perm_key,0)
    pp0 = {pk: PP[pk][h][0][idx] for pk in PERM_KEYS}
    boot_imp = {pk: np.full(N_BOOTSTRAP, np.nan) for pk in PERM_KEYS}
    for b, sel in enumerate(sels):
        yb = y_all[sel]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        a_f = roc_auc_score(yb, pf[sel])
        for pk in PERM_KEYS:
            boot_imp[pk][b] = a_f - roc_auc_score(yb, pp0[pk][sel])

    def ci_of(arr: np.ndarray) -> dict:
        n_bv = int(np.count_nonzero(~np.isnan(arr)))
        lo = float(np.nanpercentile(arr, (1 - CI) / 2 * 100)) if n_bv >= 100 else float("nan")
        hi = float(np.nanpercentile(arr, (1 + CI) / 2 * 100)) if n_bv >= 100 else float("nan")
        return {"lo": lo, "hi": hi, "n_bootstrap_valid": n_bv}

    for pk in PERM_KEYS:
        summary["heads"][h]["perm"][pk]["importance_cluster_ci"] = ci_of(boot_imp[pk])

    # paired pairwise stream-difference CIs: importance_S1 - importance_S2
    # (note: full cancels, so this is perm_S2 - perm_S1 per resample).
    pair_ci = {}
    for a_i in range(len(stream_names)):
        for b_i in range(a_i + 1, len(stream_names)):
            s1, s2 = stream_names[a_i], stream_names[b_i]
            diff = boot_imp[s1] - boot_imp[s2]
            pair_ci[f"{s1}_minus_{s2}"] = ci_of(diff)
    summary["heads"][h]["pairwise_importance_diff_ci"] = pair_ci
    summary["n_persons"] = len(uniq)

    print(f"\n=== {HEADLINE} per-stream importance (full={full_auroc[h]:.4f}, "
          f"whole-wearable full-perm_all={summary['heads'][h]['perm']['all']['importance_perm']:+.4f}) ===",
          flush=True)
    for pk in PERM_KEYS:
        st = summary["heads"][h]["perm"][pk]; ci = st["importance_cluster_ci"]
        sig = "sig" if (not np.isnan(ci["lo"]) and ci["lo"] > 0) else "CI incl 0"
        tag = "WHOLE-WEARABLE" if pk == "all" else "stream"
        print(f"  {pk:<6} ({tag:<14}) import(full-perm) = {st['importance_perm']:+.4f}  "
              f"(perm SD {st['auroc_perm_sd']:.4f}; 95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}], "
              f"n_boot={ci['n_bootstrap_valid']})  {sig}", flush=True)
    print("  pairwise importance differences (paired person-cluster CI):", flush=True)
    for name, ci in pair_ci.items():
        sig = "sig" if (not np.isnan(ci["lo"]) and not np.isnan(ci["hi"]) and (ci["lo"] > 0 or ci["hi"] < 0)) else "CI incl 0"
        print(f"    {name:<16} 95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]  {sig}", flush=True)
    print("  CAVEAT: per-stream importances are UNIQUE (leave-one-out) contributions; they do NOT "
          "sum to whole-wearable and are confounded across streams by differential redundancy and "
          "column count (steps=1, hr=4, sleep=6 cols). Read as unique-given-the-rest, not signal rank.",
          flush=True)

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
