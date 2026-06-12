"""
11_ablation_streams_v2.py — follow-ups to 10_ablation_streams.py, answering two
questions the leave-one-out run left open, in a single GPU pass (fp32):

  Q1  ADD-ONE-IN (FULL context): is steps' ~0 leave-one-out importance because it
      is REDUNDANT with HR+sleep, or because it is UNINFORMATIVE?  For each stream
      S, permute the COMPLEMENT (the other two wearable streams) across windows so
      only S stays correctly paired, EHR + confounders intact.
        addin_S = AUROC(only_S) - AUROC(no-wearables)   [no-wearables = perm-all]
      A large addin_steps means steps DOES carry CV signal (just redundant);
      ~0 means genuinely uninformative.

  Q2  LEAVE-ONE-OUT in the PHONE-DEPLOYABLE (EHR-masked) context: which wearable
      stream does the on-device model (no hospital EHR, demographics kept) lean
      on?  EHR event tokens masked (attn_mask & token_types<4); confounders kept
      (= §15 wear_demo context, AUROC ~0.85).  Per stream S:
        loo_S_demo = AUROC(DEMO) - AUROC(DEMO + perm_S)
      Here redundancy is only across the 3 streams (no EHR to compensate), so
      per-stream values are larger and more interpretable than the FULL-context
      leave-one-out in 10.

Permutation-only, in-distribution (zeroing = the model's learned missing-day
code, not a clean counterfactual — see 10's review).  Confounders (demographics)
are kept in BOTH contexts; the only difference between FULL and DEMO is whether
EHR event tokens are attended.  Wearable column map (02_tokenizer_v3.py):
  steps [0]; heart-rate [1,2,3,4]; sleep [5,6,7,8,9,10].

Conditions per batch (cv_composite heads); K = N_PERM derangements, SHARED across
all perm keys per batch (fair, neighbor-frac counted once):
  POINT   full        : inputs unchanged                  (reproduces 0.8857)
  POINT   demo        : EHR masked, demographics kept      (reproduces ~0.85)
  PERM    full_perm_all   : permute all 11 wearable cols, FULL  (no-wear ref)
  PERM    full_only_steps : permute HR+sleep cols,        FULL  (steps alone)
  PERM    full_only_hr    : permute steps+sleep cols,     FULL  (HR alone)
  PERM    full_only_sleep : permute steps+HR cols,        FULL  (sleep alone)
  PERM    demo_perm_all   : permute all 11 cols,          DEMO  (no-wear ref)
  PERM    demo_perm_steps : permute steps col,            DEMO  (LOO steps)
  PERM    demo_perm_hr    : permute HR cols,              DEMO  (LOO HR)
  PERM    demo_perm_sleep : permute sleep cols,           DEMO  (LOO sleep)
=> 2 + 8*K forward passes/batch.

Derived (point estimates + person-cluster bootstrap 95% CI on perm_0 differences):
  whole-wear import (full) = full - full_perm_all   (sanity vs 10/§15 ~+0.0275)
  whole-wear import (demo) = demo - demo_perm_all   (new: wearables' value to the
                                                     phone model)
  addin_S (full)  = full_only_S - full_perm_all     [Q1]  S in steps/hr/sleep
  loo_S (demo)    = demo - demo_perm_S              [Q2]  S in steps/hr/sleep
Plus paired pairwise-difference CIs among the three addin_S and among the three
loo_S (shared resamples).

INTERPRETATION CAVEATS (carry forward from 10):
  - Column-count confound, and it runs in OPPOSITE directions for the two
    estimators.  For LEAVE-ONE-OUT (demo) more cols permuted => more perturbation
    (steps=1 < hr=4 < sleep=6).  For ADD-ONE-IN (full) only_S permutes the
    COMPLEMENT, so only_steps permutes 10 cols / only_sleep permutes 5 cols, and
    addin_S compares "permute complement" vs "permute all" — addin_sleep keeps 6
    cols correctly paired while addin_steps keeps only 1.  => addin magnitudes are
    NOT comparable across streams and the count effect can INVERT the steps-vs-
    sleep ordering.  HEADLINE Q1 only as a WITHIN-stream test: is addin_steps
    itself > 0 (its own CI vs no-wearables)?  Do NOT rank addin across streams.
  - addin_S measures S "alone among wearables, others decorrelated" — the other
    streams are permuted (in-distribution) not removed, so EHR + demographics +
    S's own signal drive the prediction.  perm_all keeps the SAME EHR+demographics
    and permutes all wearable cols, so the subtraction cancels the EHR/demographic
    baseline in expectation; addin_S is steps' marginal over no-wearable-signal.
  - The FULL and DEMO perm_all share the same per-batch derangement, so the
    cross-context whole-wear contrast (import_full vs import_demo) has correlated
    references — report the two with their own CIs, NOT a paired (full-demo) CI.

Robustness (same as 10): raw per-window preds written to .npz BEFORE any AUROC;
batch loop dumps partials on abort; FULL reproduction is a HARD ASSERT before the
JSON is written.

In:  ~/workspace/phonefm-data/tokenized_v3/test_*.parquet, .../phonefm_v3/best.pt + config.json
Out: ~/workspace/phonefm-data/phonefm_v3/ablation_streams_v2.json
     ~/workspace/phonefm-data/phonefm_v3/ablation_streams_v2_rawpreds.npz
Run (GPU pod):  cd ~/repos/PhoneFM/workbench && python3 -u 11_ablation_streams_v2.py
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
OUT_PATH = MODEL_DIR / "ablation_streams_v2.json"
RAW_PATH = MODEL_DIR / "ablation_streams_v2_rawpreds.npz"
PARTIAL_PATH = MODEL_DIR / "ablation_streams_v2_rawpreds.PARTIAL.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
NUM_WORKERS = 4
SEED = 20260611

HEADS = ("cv_composite_30d", "cv_composite_180d", "cv_composite_365d")
HEADLINE = "cv_composite_30d"
N_WEARABLE_FEATURES = 11      # must match cfg.n_wearable_features; asserted below
EHR_TYPE_MIN = 4              # token_types >= 4 are EHR event tokens (DX/MED/PX/LAB)
N_PERM = 5
N_BOOTSTRAP = 2000
CI = 0.95
REFERENCE_AUROC_CV30D = 0.8856857329969465  # CPU fp32 baseline (FULL); hard-asserted
REFERENCE_AUROC_DEMO_CV30D = 0.8500          # §15 wear_demo (EHR masked); soft-checked
REPRO_TOL = 5e-3
DEMO_TOL = 1.5e-2

# Per-stream column indices into wearable_feats[..., :11] (see 02_tokenizer_v3.py).
STREAMS: dict[str, list[int]] = {
    "steps": [0],
    "hr":    [1, 2, 3, 4],
    "sleep": [5, 6, 7, 8, 9, 10],
}
ALL_COLS = list(range(N_WEARABLE_FEATURES))


def _complement(cols: list[int]) -> list[int]:
    return [c for c in ALL_COLS if c not in cols]


# perm key -> (columns_to_permute, context)  context in {"full","demo"}
# FULL add-one-in: permute the COMPLEMENT of stream S (others decorrelated).
# DEMO leave-one-out: permute stream S itself (EHR masked).
PERM_SPECS: dict[str, tuple[list[int], str]] = {
    "full_perm_all":   (ALL_COLS,                       "full"),
    "full_only_steps": (_complement(STREAMS["steps"]),  "full"),
    "full_only_hr":    (_complement(STREAMS["hr"]),     "full"),
    "full_only_sleep": (_complement(STREAMS["sleep"]),  "full"),
    "demo_perm_all":   (ALL_COLS,                       "demo"),
    "demo_perm_steps": (STREAMS["steps"],               "demo"),
    "demo_perm_hr":    (STREAMS["hr"],                  "demo"),
    "demo_perm_sleep": (STREAMS["sleep"],               "demo"),
}
POINT_KEYS = ("full", "demo")
STREAM_NAMES = list(STREAMS.keys())


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
        print("WARNING: CPU run does ~42 forward passes/batch — very slow; GPU strongly recommended.", flush=True)

    with open(MODEL_DIR / "config.json") as f:
        saved = json.load(f)
    cfg_d = saved["config"].copy()
    cfg_d["head_specs"] = tuple(tuple(s) for s in cfg_d["head_specs"])
    cfg = PhoneFMV3Config(**cfg_d)
    model = PhoneFMV3(cfg).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location=DEVICE), strict=True)
    model.eval()
    print(f"model loaded: {model.num_params() / 1e6:.1f}M params", flush=True)

    # Guard: hard-coded column map must match the model's feature count and
    # partition every column exactly once.
    assert cfg.n_wearable_features == N_WEARABLE_FEATURES, (
        f"cfg.n_wearable_features={cfg.n_wearable_features} != {N_WEARABLE_FEATURES}; "
        "stream column map is stale — refusing to run."
    )
    part = sorted(c for cols in STREAMS.values() for c in cols)
    assert part == ALL_COLS, f"STREAMS columns {part} do not partition 0..{N_WEARABLE_FEATURES - 1}."

    torch.manual_seed(SEED)
    loader = make_loader_v3(
        str(DATA_DIR / "test_*.parquet"), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        shuffle=True, max_seq_len=cfg.max_seq_len, n_confounders=cfg.n_confounders,
    )
    print(f"loader: {len(loader)} batches, {len(loader.dataset)} windows", flush=True)
    g = torch.Generator(); g.manual_seed(SEED)

    col_t = {k: torch.tensor(cols, dtype=torch.long, device=DEVICE) for k, (cols, _) in PERM_SPECS.items()}

    # accumulators
    preds_point = {pk: {h: [] for h in HEADS} for pk in POINT_KEYS}
    preds_perm = {pk: {h: [[] for _ in range(N_PERM)] for h in HEADS} for pk in PERM_SPECS}
    labels = {h: [] for h in HEADS}
    masks = {h: [] for h in HEADS}
    person_ids: list[np.ndarray] = []
    same_person_neighbor = 0
    total_pairs = 0
    n_batches_done = 0

    def dump_partial(reason: str) -> None:
        # Best-effort: emit ONLY the first n_batches_done fully-completed batches for
        # every accumulator, so a mid-batch failure cannot produce a column-misaligned
        # PARTIAL (person_ids is appended before the forwards, labels/perms after).
        try:
            nb = n_batches_done
            arrs: dict[str, np.ndarray] = {}
            if nb > 0:
                arrs["person_id"] = np.concatenate(person_ids[:nb])
            for h in HEADS:
                if nb > 0:
                    arrs[f"label__{h}"] = np.concatenate(labels[h][:nb]).reshape(-1)
                    arrs[f"mask__{h}"] = np.concatenate(masks[h][:nb]).reshape(-1)
                    for pk in POINT_KEYS:
                        arrs[f"point_{pk}__{h}"] = np.concatenate(preds_point[pk][h][:nb]).reshape(-1)
                    for pk in PERM_SPECS:
                        for k in range(N_PERM):
                            arrs[f"perm_{pk}_k{k}__{h}"] = np.concatenate(preds_perm[pk][h][k][:nb]).reshape(-1)
            np.savez_compressed(PARTIAL_PATH, reason=np.asarray(reason), n_batches=np.asarray(nb), **arrs)
            print(f"  [partial dump] {reason}: wrote {PARTIAL_PATH} with {nb} aligned batches", flush=True)
        except Exception as e:
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

            attn_ctx = {
                "full": attn_mask,
                "demo": attn_mask & (token_types < EHR_TYPE_MIN),   # mask EHR event tokens
            }

            def fwd(wf, amask):
                # confounders (demographics) always kept; only wearable cols and the
                # attention context (full vs EHR-masked) vary.
                logits = model(input_ids, token_types, day_index, wf, confounders, amask)
                return {h: torch.sigmoid(logits[h].float()).cpu().numpy() for h in HEADS}

            for pk in POINT_KEYS:
                op = fwd(wearable, attn_ctx[pk])
                for h in HEADS:
                    preds_point[pk][h].append(op[h])

            # one shared derangement list per batch, reused across ALL perm keys.
            derangements = []
            for k in range(N_PERM):
                d = random_derangement(B, g)
                derangements.append(d.to(DEVICE))
                same_person_neighbor += int((pid[d.numpy()] == pid).sum()); total_pairs += B

            for pk, (_, ctx) in PERM_SPECS.items():
                own = col_t[pk]
                amask = attn_ctx[ctx]
                for k in range(N_PERM):
                    wf = wearable.clone()
                    wf[:, :, own] = wearable[derangements[k]][:, :, own]
                    op = fwd(wf, amask)
                    for h in HEADS:
                        preds_perm[pk][h][k].append(op[h])

            for h in HEADS:
                labels[h].append(batch["labels"][h].numpy())
                masks[h].append(batch["masks"][h].numpy())
            n_batches_done = i + 1
            if i % 25 == 0:
                print(f"  batch {i}/{len(loader)}", flush=True)
    except BaseException as e:
        print(f"\nABORTED at batch {n_batches_done}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        dump_partial(f"{type(e).__name__}: {e}")
        raise

    # ---- Concatenate
    pid_all = np.concatenate(person_ids)
    lab = {h: np.concatenate(labels[h]).astype(np.int64) for h in HEADS}
    msk = {h: np.concatenate(masks[h]).astype(bool) for h in HEADS}
    PT = {pk: {h: np.concatenate(preds_point[pk][h]).astype(np.float64).reshape(-1) for h in HEADS} for pk in POINT_KEYS}
    PP = {pk: {h: [np.concatenate(preds_perm[pk][h][k]).astype(np.float64).reshape(-1) for k in range(N_PERM)]
               for h in HEADS} for pk in PERM_SPECS}

    # ---- WRITE-BEFORE-VERIFY
    raw: dict[str, np.ndarray] = {"person_id": pid_all}
    for h in HEADS:
        raw[f"label__{h}"] = lab[h]
        raw[f"mask__{h}"] = msk[h]
        for pk in POINT_KEYS:
            raw[f"point_{pk}__{h}"] = PT[pk][h]
        for pk in PERM_SPECS:
            for k in range(N_PERM):
                raw[f"perm_{pk}_k{k}__{h}"] = PP[pk][h][k]
    np.savez_compressed(RAW_PATH, **raw)
    print(f"\n[write-before-verify] saved raw preds {RAW_PATH}", flush=True)
    if PARTIAL_PATH.exists():
        try:
            PARTIAL_PATH.unlink()
        except OSError:
            pass

    frac_same = same_person_neighbor / max(total_pairs, 1)
    print(f"same-person neighbor fraction across permutations: {frac_same:.4f} (want ~0)", flush=True)

    # ---- HARD reproduction guard (FULL) before computing/writing deltas.
    a_full_h = auroc_masked(PT["full"][HEADLINE][msk[HEADLINE]], lab[HEADLINE][msk[HEADLINE]])
    a_demo_h = auroc_masked(PT["demo"][HEADLINE][msk[HEADLINE]], lab[HEADLINE][msk[HEADLINE]])
    gap = abs(a_full_h - REFERENCE_AUROC_CV30D)
    dgap = abs(a_demo_h - REFERENCE_AUROC_DEMO_CV30D)
    print(f"FULL AUROC (GPU fp32) {a_full_h:.6f} vs CPU ref {REFERENCE_AUROC_CV30D:.6f} (|Δ|={gap:.2e})", flush=True)
    print(f"DEMO AUROC {a_demo_h:.4f} vs §15 wear_demo {REFERENCE_AUROC_DEMO_CV30D:.4f} "
          f"(|Δ|={dgap:.2e}, {'OK' if dgap < DEMO_TOL else 'CHECK'})", flush=True)
    assert gap < REPRO_TOL, (
        f"FULL AUROC {a_full_h:.6f} does not reproduce CPU ref {REFERENCE_AUROC_CV30D:.6f} "
        f"(|Δ|={gap:.2e} >= {REPRO_TOL}); refusing to write deltas. Raw preds in {RAW_PATH}."
    )
    # Q2 (the phone-deployable story) is entirely in the DEMO context, so gate the
    # whole run on the DEMO baseline reproducing §15's wear_demo too — a masking
    # drift here would silently poison every loo_demo number.
    assert dgap < DEMO_TOL, (
        f"DEMO AUROC {a_demo_h:.4f} does not reproduce §15 wear_demo {REFERENCE_AUROC_DEMO_CV30D:.4f} "
        f"(|Δ|={dgap:.2e} >= {DEMO_TOL}); EHR-masking context looks wrong — refusing to write Q2 deltas. "
        f"Raw preds in {RAW_PATH}."
    )

    summary: dict = {
        "config": {"seed": SEED, "n_perm": N_PERM, "n_bootstrap": N_BOOTSTRAP, "ci": CI,
                   "batch_size": BATCH_SIZE, "device": DEVICE, "precision": "fp32",
                   "bootstrap_unit": "person (cluster)",
                   "reference_auroc_cv30d_cpu_fp32": REFERENCE_AUROC_CV30D,
                   "reference_auroc_demo_cv30d": REFERENCE_AUROC_DEMO_CV30D,
                   "same_person_neighbor_frac": frac_same,
                   "streams": {s: STREAMS[s] for s in STREAMS},
                   "estimators": "FULL add-one-in: only_S (permute complement of S) - perm_all; "
                                 "DEMO leave-one-out: demo - demo_perm_S (EHR masked, demographics kept). "
                                 "Magnitudes confounded by column count (steps=1/hr=4/sleep=6) — read "
                                 "qualitatively, not as a signal ranking."},
        "heads": {},
    }

    # per-head point AUROCs and per-key perm means
    point_auroc = {pk: {} for pk in POINT_KEYS}
    perm_auroc = {pk: {} for pk in PERM_SPECS}   # perm_auroc[key][head] -> list of K
    for h in HEADS:
        m = msk[h]; y = lab[h][m]
        for pk in POINT_KEYS:
            point_auroc[pk][h] = auroc_masked(PT[pk][h][m], y)
        for pk in PERM_SPECS:
            perm_auroc[pk][h] = [auroc_masked(PP[pk][h][k][m], y) for k in range(N_PERM)]

    print(f"\n{'head':<20} {'full':>7} {'demo':>7} | "
          f"{'addin_steps':>11} {'addin_hr':>9} {'addin_sleep':>11} | "
          f"{'loo_steps':>9} {'loo_hr':>7} {'loo_sleep':>9}", flush=True)
    for h in HEADS:
        m = msk[h]
        a_full = point_auroc["full"][h]; a_demo = point_auroc["demo"][h]
        mu = {pk: float(np.nanmean(perm_auroc[pk][h])) for pk in PERM_SPECS}
        sd = {pk: float(np.nanstd(perm_auroc[pk][h])) for pk in PERM_SPECS}
        addin = {s: mu[f"full_only_{s}"] - mu["full_perm_all"] for s in STREAM_NAMES}
        loo = {s: a_demo - mu[f"demo_perm_{s}"] for s in STREAM_NAMES}
        print(f"{h:<20} {a_full:>7.4f} {a_demo:>7.4f} | "
              f"{addin['steps']:>+11.4f} {addin['hr']:>+9.4f} {addin['sleep']:>+11.4f} | "
              f"{loo['steps']:>+9.4f} {loo['hr']:>+7.4f} {loo['sleep']:>+9.4f}", flush=True)
        summary["heads"][h] = {
            "n_valid": int(m.sum()), "n_pos": int(lab[h][m].sum()),
            "auroc_full": a_full, "auroc_demo": a_demo,
            "whole_wear_import_full": a_full - mu["full_perm_all"],
            "whole_wear_import_demo": a_demo - mu["demo_perm_all"],
            "perm_mean": mu, "perm_sd": sd,
            "addin_full": addin,     # only_S - perm_all  (Q1)
            "loo_demo": loo,         # demo - demo_perm_S (Q2)
        }

    # ---- Person-cluster bootstrap on the HEADLINE (perm_0-based differences).
    h = HEADLINE; m = msk[h]
    idx = np.where(m)[0]
    y_all = lab[h][idx]; pidm = pid_all[idx]
    order = np.argsort(pidm, kind="stable")
    uniq, starts = np.unique(pidm[order], return_index=True)
    groups = np.split(order, starts[1:])
    rng = np.random.RandomState(SEED)
    sels = [np.concatenate([groups[j] for j in rng.randint(0, len(uniq), len(uniq))])
            for _ in range(N_BOOTSTRAP)]

    pf = PT["full"][h][idx]; pd_ = PT["demo"][h][idx]
    # all K derangements per key, so the CI averages over K to MATCH the K-mean
    # point estimate (effects here are near-zero, so a single perm_0 CI would be
    # centered on a different number than the headline — see adversarial review).
    ppk = {pk: [PP[pk][h][k][idx] for k in range(N_PERM)] for pk in PERM_SPECS}

    boot_addin = {s: np.full(N_BOOTSTRAP, np.nan) for s in STREAM_NAMES}
    boot_loo = {s: np.full(N_BOOTSTRAP, np.nan) for s in STREAM_NAMES}
    boot_wwf = np.full(N_BOOTSTRAP, np.nan)   # whole-wear import full
    boot_wwd = np.full(N_BOOTSTRAP, np.nan)   # whole-wear import demo
    for b, sel in enumerate(sels):
        yb = y_all[sel]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue

        def kmean(pk: str) -> float:
            return float(np.mean([roc_auc_score(yb, ppk[pk][k][sel]) for k in range(N_PERM)]))

        a_full = roc_auc_score(yb, pf[sel])
        a_demo = roc_auc_score(yb, pd_[sel])
        a_pall_f = kmean("full_perm_all")
        a_pall_d = kmean("demo_perm_all")
        boot_wwf[b] = a_full - a_pall_f
        boot_wwd[b] = a_demo - a_pall_d
        for s in STREAM_NAMES:
            boot_addin[s][b] = kmean(f"full_only_{s}") - a_pall_f
            boot_loo[s][b] = a_demo - kmean(f"demo_perm_{s}")

    def ci_of(arr: np.ndarray) -> dict:
        n_bv = int(np.count_nonzero(~np.isnan(arr)))
        lo = float(np.nanpercentile(arr, (1 - CI) / 2 * 100)) if n_bv >= 100 else float("nan")
        hi = float(np.nanpercentile(arr, (1 + CI) / 2 * 100)) if n_bv >= 100 else float("nan")
        return {"lo": lo, "hi": hi, "n_bootstrap_valid": n_bv}

    hb = summary["heads"][h]
    hb["whole_wear_import_full_ci"] = ci_of(boot_wwf)
    hb["whole_wear_import_demo_ci"] = ci_of(boot_wwd)
    hb["addin_full_ci"] = {s: ci_of(boot_addin[s]) for s in STREAM_NAMES}
    hb["loo_demo_ci"] = {s: ci_of(boot_loo[s]) for s in STREAM_NAMES}
    # paired pairwise-difference CIs (shared resamples)
    def pair_ci(boot: dict) -> dict:
        out = {}
        for a_i in range(len(STREAM_NAMES)):
            for b_i in range(a_i + 1, len(STREAM_NAMES)):
                s1, s2 = STREAM_NAMES[a_i], STREAM_NAMES[b_i]
                out[f"{s1}_minus_{s2}"] = ci_of(boot[s1] - boot[s2])
        return out
    hb["addin_pairwise_ci"] = pair_ci(boot_addin)
    hb["loo_pairwise_ci"] = pair_ci(boot_loo)
    summary["n_persons"] = len(uniq)

    def _fmt(name, val, ci):
        sig = "sig" if (not np.isnan(ci["lo"]) and ci["lo"] > 0) else "CI incl 0"
        return f"  {name:<22} {val:>+8.4f}  95% CI [{ci['lo']:>+.4f}, {ci['hi']:>+.4f}]  {sig}"

    print(f"\n=== {HEADLINE} (full={point_auroc['full'][h]:.4f}, demo={point_auroc['demo'][h]:.4f}) ===", flush=True)
    print("  Q1 ADD-ONE-IN (FULL context: stream alone among wearables, over no-wear):", flush=True)
    print(_fmt("whole-wearable", hb["whole_wear_import_full"], hb["whole_wear_import_full_ci"]), flush=True)
    for s in STREAM_NAMES:
        print(_fmt(f"addin_{s}", hb["addin_full"][s], hb["addin_full_ci"][s]), flush=True)
    print("  Q2 LEAVE-ONE-OUT (DEMO/EHR-masked: phone-deployable context):", flush=True)
    print(_fmt("whole-wearable(demo)", hb["whole_wear_import_demo"], hb["whole_wear_import_demo_ci"]), flush=True)
    for s in STREAM_NAMES:
        print(_fmt(f"loo_{s}(demo)", hb["loo_demo"][s], hb["loo_demo_ci"][s]), flush=True)
    print("  CAVEAT: ADD-ONE-IN — headline only WITHIN-stream (is addin_steps>0 vs its own CI?); do NOT", flush=True)
    print("    rank addin across streams (column count runs OPPOSITE here: only_steps perturbs 10 cols,", flush=True)
    print("    only_sleep 5). LEAVE-ONE-OUT(demo) ranking is column-count-confounded same as 10. The", flush=True)
    print("    full-vs-demo whole-wear references share a derangement — don't form a paired (full-demo) CI.", flush=True)

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
