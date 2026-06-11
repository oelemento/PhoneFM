"""
08_trajectory_analysis.py — time-to-event aligned cardiovascular risk
trajectories from the per-window predictions written by 07_save_predictions.py.

The question: *does the model's predicted CV risk separate event cases from
event-free controls in the windows BEFORE the 30-day label horizon turns on?*
That is the only version of "early warning" that is not circular — see below.

==============================================================================
Why the naive version is circular (and how this script avoids it)
==============================================================================
We do not have exact event dates.  We infer each case's event date from the
cv30 LABEL: the windows with cv30_label==1 bracket the event in [D_event-30,
D_event].  But the cv30 head was TRAINED to fire exactly on that label.  So if
we plotted cv30_pred vs distance-to-the-label-derived-anchor, the rise into
x=0 would be guaranteed by construction, not evidence of foresight.  Likewise
the cv180/cv365 heads have label horizons that cover the ENTIRE [-180,0]
plotted range, so their high predictions there are tautological end-to-end —
those panels are NOT shown as primary evidence.

The non-circular claim this script tests:
  Does the cv30 head — which only ever saw a 30-day label — predict ELEVATED
  risk for cases vs controls at x < -30 (i.e., 30-180 days before the event,
  OUTSIDE its training horizon)?  If yes, the model generalizes beyond the
  horizon it was trained on, which is genuine early warning.

Design choices that make the claim defensible (each was a code-review finding):
  1. FIRST-event anchoring (min cv30-positive end_date), not last-event — so
     x<0 is unambiguously pre-first-event and recurrent AFib does not fold
     inter-event intervals into the early bins.
  2. Controls anchored at a RANDOM masked window (seeded), not their last
     observation — removes the case=real-event vs control=low-risk-endpoint
     asymmetry that could manufacture the gap.
  3. Primary statistic = bootstrap CI on the per-bin (case_mean - control_mean)
     DIFFERENCE, declared "elevated" only where the difference CI excludes 0.
     (Comparing two separate CI bands by eye is the wrong test.)
  4. The [-30, 0) bin is SHADED as "label-on (tautological)"; the early-warning
     claim is read only from bins left of -30.
  5. A balanced-panel sensitivity curve (persons present in ALL bins) guards
     against the survivorship confound where far-negative bins are a healthier,
     longer-history subset.
  6. n-per-bin is annotated on the figure and bins below MIN_N_PER_BIN are
     greyed; CIs on thin bins are not trustworthy.
  7. Lead time = left edge of the earliest CONTIGUOUS run of
     difference-significant bins that reaches the -30 boundary, gated on MIN_N.
     No arbitrary absolute-probability threshold.

==============================================================================
Inputs / Outputs
==============================================================================
In:  ~/workspace/phonefm-data/phonefm_v3/test_predictions.parquet
Out: ~/workspace/phonefm-data/phonefm_v3/trajectory_aligned.png   (main figure)
     ~/workspace/phonefm-data/phonefm_v3/trajectory_examples.png  (individual)
     ~/workspace/phonefm-data/phonefm_v3/trajectory_summary.json  (all stats)

Run (fast, pure numpy/pandas/matplotlib):
    cd ~/repos/PhoneFM/workbench && python3 08_trajectory_analysis.py

Figures are written on the pod.  Aggregate trajectories are summary stats; the
example panels are de-identified ("Case A".."Case E"); view via the Jupyter
file browser and do not export off-perimeter without the usual review.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless pod
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===========================================================================
# Config
# ===========================================================================

MODEL_DIR = Path("/home/jupyter/workspace/phonefm-data/phonefm_v3")
PRED_PATH = MODEL_DIR / "test_predictions.parquet"
FIG_AGG = MODEL_DIR / "trajectory_aligned.png"
FIG_EX = MODEL_DIR / "trajectory_examples.png"
SUMMARY = MODEL_DIR / "trajectory_summary.json"

# cv30 is the ONLY non-circular early-warning probe (trained on a 30d label;
# we test elevation OUTSIDE that horizon).  cv180/cv365 are computed into the
# JSON for completeness but flagged tautological-across-range and not plotted
# as primary evidence.
PRIMARY = "cv30"
TAUTOLOGICAL_FROM = -30  # cv30 label is "on" for x in [-30, 0]; claim lives left of this

LOOKBACK_DAYS = 180     # plot window: x in [-LOOKBACK_DAYS, 0)
BIN_DAYS = 30           # time-bin width
N_BOOTSTRAP = 1000      # person-level resamples
CI = 0.95
MIN_N_PER_BIN = 20      # bins below this are greyed; CIs untrustworthy
N_EXAMPLES = 5
RNG_SEED = 20260611     # deterministic (no Date.now/random at import)


# ===========================================================================
# Binning helpers
# ===========================================================================

def bin_edges() -> np.ndarray:
    return np.arange(-LOOKBACK_DAYS, 1, BIN_DAYS)  # [-180,...,-30,0] → 6 bins


def bin_centers(edges: np.ndarray) -> np.ndarray:
    return (edges[:-1] + edges[1:]) / 2.0


def assign_bin(x_days: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Left-closed bins; x>=0 (event window), x<-LOOKBACK, and non-finite
    (NaN/NaT-derived) all excluded (idx=-1).  The NaN guard is load-bearing:
    a control whose random anchor is unmapped for a non-primary horizon yields
    NaN x_days, which without this guard digitizes to an out-of-range bin and
    crashes person_bin_matrix with an IndexError."""
    x = np.asarray(x_days, dtype=np.float64)
    idx = np.digitize(x, edges) - 1
    idx[(x < edges[0]) | (x >= edges[-1]) | ~np.isfinite(x)] = -1
    return idx


def person_bin_matrix(
    df: pd.DataFrame, pred_col: str, edges: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """One mean predicted-risk value per (person, bin).  Returns (matrix, pids)
    where matrix is (n_persons, n_bins) with NaN for empty (person,bin)."""
    n_bins = len(edges) - 1
    d = df.copy()
    d["bin"] = assign_bin(d["x_days"].to_numpy(), edges)
    d = d[d["bin"] >= 0]
    pids = sorted(d["person_id"].unique().tolist())
    pos = {p: i for i, p in enumerate(pids)}
    mat = np.full((len(pids), n_bins), np.nan, dtype=np.float64)
    bm = d.groupby(["person_id", "bin"])[pred_col].mean()
    for (pid, b), v in bm.items():
        mat[pos[int(pid)], int(b)] = v
    return mat, pids


def _nanmean_cols(sample: np.ndarray) -> np.ndarray:
    """nan-aware column mean, NaN where a column is all-NaN (no warning)."""
    n_bins = sample.shape[1]
    out = np.full(n_bins, np.nan)
    ok = np.sum(~np.isnan(sample), axis=0) > 0
    if ok.any():
        with np.errstate(invalid="ignore"):
            out[ok] = np.nanmean(sample[:, ok], axis=0)
    return out


def bootstrap_difference(
    case_mat: np.ndarray, ctrl_mat: np.ndarray, rng: np.random.RandomState
) -> dict:
    """Person-level cluster bootstrap of case mean, control mean, and the
    case-minus-control DIFFERENCE, per bin.  The difference CI is the primary
    statistic (CI excluding 0 → significant separation)."""
    n_case, n_bins = case_mat.shape
    n_ctrl = ctrl_mat.shape[0]

    case_pt = _nanmean_cols(case_mat)
    ctrl_pt = _nanmean_cols(ctrl_mat)
    diff_pt = case_pt - ctrl_pt
    case_n = np.sum(~np.isnan(case_mat), axis=0)
    ctrl_n = np.sum(~np.isnan(ctrl_mat), axis=0)

    case_boot = np.full((N_BOOTSTRAP, n_bins), np.nan)
    ctrl_boot = np.full((N_BOOTSTRAP, n_bins), np.nan)
    diff_boot = np.full((N_BOOTSTRAP, n_bins), np.nan)
    for b in range(N_BOOTSTRAP):
        cm = _nanmean_cols(case_mat[rng.randint(0, n_case, n_case), :])
        km = _nanmean_cols(ctrl_mat[rng.randint(0, n_ctrl, n_ctrl), :])
        case_boot[b] = cm
        ctrl_boot[b] = km
        diff_boot[b] = cm - km

    lo_q, hi_q = (1 - CI) / 2 * 100, (1 + CI) / 2 * 100
    with np.errstate(invalid="ignore"):
        out = {
            "case": {"point": case_pt, "lo": np.nanpercentile(case_boot, lo_q, axis=0),
                     "hi": np.nanpercentile(case_boot, hi_q, axis=0), "n": case_n},
            "control": {"point": ctrl_pt, "lo": np.nanpercentile(ctrl_boot, lo_q, axis=0),
                        "hi": np.nanpercentile(ctrl_boot, hi_q, axis=0), "n": ctrl_n},
            "diff": {"point": diff_pt, "lo": np.nanpercentile(diff_boot, lo_q, axis=0),
                     "hi": np.nanpercentile(diff_boot, hi_q, axis=0)},
        }
    return out


def _jsonify(d: dict) -> dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else
                _jsonify(v) if isinstance(v, dict) else v) for k, v in d.items()}


def _sanitize(o):
    """Replace NaN floats with None so the JSON is strict-parser-safe (jq etc.).
    Empty bins legitimately produce NaN point/lo/hi values."""
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize(x) for x in o]
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    if not PRED_PATH.exists():
        raise SystemExit(f"predictions not found: {PRED_PATH}\nRun 07_save_predictions.py first.")

    df = pd.read_parquet(PRED_PATH)
    df["end_date"] = pd.to_datetime(df["end_date"])
    print(f"loaded {len(df)} windows, {df['person_id'].nunique()} persons", flush=True)

    horizons = [c[:-5] for c in df.columns if c.endswith("_pred")]  # e.g. cv30, cv180, cv365
    a_label, a_mask = f"{PRIMARY}_label", f"{PRIMARY}_mask"

    # ---- Cohorts -----------------------------------------------------------
    # case   : >=1 PRIMARY-positive masked window (event date estimable)
    # control: NO event at ANY horizon across masked windows
    any_event = np.zeros(len(df), dtype=bool)
    for h in horizons:
        any_event |= (df[f"{h}_mask"] == 1) & (df[f"{h}_label"] == 1)
    df["_any_event"] = any_event

    case_pos = (df[a_mask] == 1) & (df[a_label] == 1)
    case_pids = set(df.loc[case_pos, "person_id"].unique())
    has_event = df.groupby("person_id")["_any_event"].any()
    control_pids = set(has_event[~has_event].index)
    neither = df["person_id"].nunique() - len(case_pids) - len(control_pids)
    print(f"cases (anchorable via {PRIMARY}): {len(case_pids)}", flush=True)
    print(f"controls (event-free all horizons): {len(control_pids)}", flush=True)
    print(f"neither (only cv180/cv365 event, excluded): {neither}", flush=True)

    # ---- FIRST-event date per case (min PRIMARY-positive end_date) ----------
    d_firstevent = df[case_pos].groupby("person_id")["end_date"].min()

    # Recurrence proxy: cases whose PRIMARY-positive windows span > 30 days
    span = (df[case_pos].groupby("person_id")["end_date"].max()
            - df[case_pos].groupby("person_id")["end_date"].min()).dt.days
    n_recurrent = int((span > 30).sum())
    print(f"cases with PRIMARY-positive span >30d (recurrence proxy): "
          f"{n_recurrent}/{len(case_pids)}", flush=True)

    edges = bin_edges()
    centers = bin_centers(edges)
    rng = np.random.RandomState(RNG_SEED)

    summary: dict = {
        "config": {"lookback_days": LOOKBACK_DAYS, "bin_days": BIN_DAYS,
                   "n_bootstrap": N_BOOTSTRAP, "ci": CI, "primary": PRIMARY,
                   "tautological_from": TAUTOLOGICAL_FROM, "min_n_per_bin": MIN_N_PER_BIN,
                   "anchor": "first-event (cases) / random-window (controls)",
                   "rng_seed": RNG_SEED},
        "n_cases": len(case_pids), "n_controls": len(control_pids),
        "n_neither_excluded": neither, "n_recurrent_cases": n_recurrent,
        "bin_centers": centers.tolist(), "horizons": {},
    }

    # ---- Control anchor: one RANDOM masked window per control (seeded) ------
    #  Pre-pick a random anchor index per control from its PRIMARY-masked
    #  windows, so x=0 is a random observation rather than the endpoint.
    ctrl_anchor: dict[int, pd.Timestamp] = {}
    ctrl_mask_df = df[df["person_id"].isin(control_pids) & (df[a_mask] == 1)]
    for pid, g in ctrl_mask_df.groupby("person_id"):
        ds = g["end_date"].to_numpy()
        ctrl_anchor[int(pid)] = pd.Timestamp(ds[rng.randint(0, len(ds))])

    # ---- Build per-horizon aligned matrices & bootstrap --------------------
    bands: dict[str, dict] = {}
    for h in horizons:
        pred_c, mask_c = f"{h}_pred", f"{h}_mask"

        cdf = df[df["person_id"].isin(case_pids) & (df[mask_c] == 1)].copy()
        cdf = cdf.join(d_firstevent.rename("anchor"), on="person_id")
        cdf["x_days"] = (cdf["end_date"] - cdf["anchor"]).dt.days
        case_mat, _ = person_bin_matrix(cdf, pred_c, edges)

        kdf = df[df["person_id"].isin(control_pids) & (df[mask_c] == 1)].copy()
        kdf["anchor"] = kdf["person_id"].map(ctrl_anchor)
        # A control masked for this horizon but with no cv30-masked window has
        # no anchor (anchors are picked from cv30-masked windows).  Drop those
        # rows so they don't become NaN x_days.  assign_bin also NaN-guards.
        kdf = kdf[kdf["anchor"].notna()]
        kdf["x_days"] = (kdf["end_date"] - kdf["anchor"]).dt.days
        ctrl_mat, _ = person_bin_matrix(kdf, pred_c, edges)

        bands[h] = bootstrap_difference(case_mat, ctrl_mat, rng)
        bands[h]["_case_mat"] = case_mat  # kept for balanced-panel sensitivity
        summary["horizons"][h] = _jsonify({k: v for k, v in bands[h].items() if not k.startswith("_")})

    # ---- Balanced-panel sensitivity (PRIMARY only): persons in ALL bins ----
    case_mat_p = bands[PRIMARY]["_case_mat"]
    full_rows = ~np.isnan(case_mat_p).any(axis=1)
    if full_rows.sum() >= MIN_N_PER_BIN:
        bal_pt = _nanmean_cols(case_mat_p[full_rows])
    else:
        bal_pt = np.full(len(centers), np.nan)
    summary["balanced_panel_n"] = int(full_rows.sum())
    summary["horizons"][PRIMARY]["case_balanced_point"] = bal_pt.tolist()

    # ---- Lead time: earliest bin of the contiguous diff-significant run that
    #  reaches the -30 boundary, gated on MIN_N (primary horizon, x<0) --------
    p = bands[PRIMARY]
    diff_lo = p["diff"]["lo"]
    enough = (p["case"]["n"] >= MIN_N_PER_BIN) & (p["control"]["n"] >= MIN_N_PER_BIN)
    sig = (diff_lo > 0) & enough            # difference CI excludes 0 AND bin has enough persons
    # claim region: bins strictly left of the tautological boundary
    claim = centers < TAUTOLOGICAL_FROM
    last_claim_bin = int(np.max(np.where(claim)[0]))  # the bin just left of -30
    lead = None
    if sig[last_claim_bin]:
        i = last_claim_bin
        while i - 1 >= 0 and sig[i - 1]:
            i -= 1
        lead = float(abs(edges[i]))  # left edge of the contiguous significant run
    summary["lead_time_days_est"] = lead
    summary["significant_bins_x_lt_-30"] = [float(centers[j]) for j in np.where(sig & claim)[0]]
    if lead is not None:
        print(f"lead time: cases significantly elevated vs controls from "
              f"~{lead:.0f} days before event (contiguous, CI-supported, n>={MIN_N_PER_BIN})", flush=True)
    else:
        print("lead time: no contiguous CI-supported elevation left of -30 (no early-warning claim)", flush=True)

    # ---- Figure: PRIMARY panel (cases vs controls + difference) ------------
    # gridspec_kw (not the height_ratios kwarg) works on matplotlib <3.6 too.
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.5, 7.5),
                                   gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
    cN, kN = p["case"]["n"], p["control"]["n"]
    thin = (cN < MIN_N_PER_BIN) | (kN < MIN_N_PER_BIN)

    axA.axvspan(TAUTOLOGICAL_FROM, 0, color="gold", alpha=0.12)
    axA.text(TAUTOLOGICAL_FROM / 2, 0.02, "label-on\n(tautological)", ha="center",
             va="bottom", fontsize=7, color="#8a6d00")
    axA.plot(centers, p["case"]["point"], "-o", color="#c0392b", lw=2, label=f"cases (n={len(case_pids)})")
    axA.fill_between(centers, p["case"]["lo"], p["case"]["hi"], color="#c0392b", alpha=0.18)
    axA.plot(centers, p["control"]["point"], "-s", color="#2c3e50", lw=2, label=f"controls (n={len(control_pids)})")
    axA.fill_between(centers, p["control"]["lo"], p["control"]["hi"], color="#2c3e50", alpha=0.15)
    axA.plot(centers, bal_pt, ":", color="#c0392b", lw=1.6, label=f"cases, balanced panel (n={int(full_rows.sum())})")
    for j, c in enumerate(centers):  # annotate n + grey thin bins
        axA.annotate(f"{int(cN[j])}", (c, p["case"]["point"][j]), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=6, color="gray")
        if thin[j]:
            axA.axvspan(c - BIN_DAYS / 2, c + BIN_DAYS / 2, color="gray", alpha=0.08)
    axA.set_ylabel(f"predicted {PRIMARY} CV risk")
    axA.set_title("CV risk: cases (first-event anchored) vs controls (random-window anchored)")
    axA.legend(fontsize=8, loc="upper left")
    axA.grid(alpha=0.25)

    axB.axhline(0, color="black", lw=0.8)
    axB.axvspan(TAUTOLOGICAL_FROM, 0, color="gold", alpha=0.12)
    axB.plot(centers, p["diff"]["point"], "-D", color="#6c3483", lw=2)
    axB.fill_between(centers, p["diff"]["lo"], p["diff"]["hi"], color="#6c3483", alpha=0.2)
    for j in np.where(sig)[0]:  # mark CI-supported, adequately-powered bins
        axB.plot(centers[j], p["diff"]["point"][j], "*", color="green", ms=13, zorder=5)
    axB.set_ylabel("case − control\n(95% CI)")
    axB.set_xlabel("days before event (cases) / before random anchor (controls)")
    axB.grid(alpha=0.25)
    if lead is not None:
        axB.axvline(-lead, ls="--", color="green", lw=1)
        axB.text(-lead, axB.get_ylim()[1] * 0.8, f"lead ~{lead:.0f}d", color="green", fontsize=8, ha="right")

    fig.suptitle("Time-to-event aligned CV risk (early-warning claim lives left of the gold band)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_AGG, dpi=150, bbox_inches="tight")
    print(f"saved {FIG_AGG}", flush=True)

    # ---- Individual example panels (slope computed on x<-30 only) ----------
    pred_c, mask_c = f"{PRIMARY}_pred", f"{PRIMARY}_mask"
    cdf = df[df["person_id"].isin(case_pids) & (df[mask_c] == 1)].copy()
    cdf = cdf.join(d_firstevent.rename("anchor"), on="person_id")
    cdf["x_days"] = (cdf["end_date"] - cdf["anchor"]).dt.days
    cdf = cdf[(cdf["x_days"] >= -LOOKBACK_DAYS) & (cdf["x_days"] < 0)]

    slopes = {}
    for pid, g in cdf.groupby("person_id"):
        pre = g[g["x_days"] < TAUTOLOGICAL_FROM]  # non-tautological region only
        if len(pre) >= 3 and np.ptp(pre["x_days"].to_numpy()) > 0:
            x = pre["x_days"].to_numpy(float)
            y = pre[pred_c].to_numpy(float)
            slopes[int(pid)] = float(np.polyfit(x, y, 1)[0])
    if slopes:
        ordered = sorted(slopes.items(), key=lambda kv: kv[1])  # ascending: A=falling, E=rising
        n = len(ordered)
        idx = sorted(set(int(round(q * (n - 1))) for q in (0.0, 0.25, 0.5, 0.75, 1.0)))
        ex = [ordered[i][0] for i in idx][:N_EXAMPLES]
        ncol = len(ex)
        figx, axx = plt.subplots(1, ncol, figsize=(3.2 * ncol, 3.4), sharey=True)
        if ncol == 1:
            axx = [axx]
        for k, (ax, pid) in enumerate(zip(axx, ex)):
            g = cdf[cdf["person_id"] == pid].sort_values("x_days")
            ax.axvspan(TAUTOLOGICAL_FROM, 0, color="gold", alpha=0.12)
            ax.plot(g["x_days"], g[pred_c], "-o", color="#c0392b", lw=2)
            sl = slopes[pid] * 30
            tag = "model MISS" if sl < 0 else "rising"
            ax.set_title(f"Case {chr(65+k)}: {tag}\n(slope {sl:+.3f}/mo, x<-30)", fontsize=8)
            ax.set_xlabel("days before event")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            if k == 0:
                ax.set_ylabel(f"predicted {PRIMARY} CV risk")
        figx.suptitle("Representative individual trajectories — spread across slope distribution (NOT typical cases)", y=1.04, fontsize=10)
        figx.tight_layout()
        figx.savefig(FIG_EX, dpi=150, bbox_inches="tight")
        print(f"saved {FIG_EX}", flush=True)
        summary["example_cases"] = {chr(65 + k): {"slope_per_month_xlt30": slopes[pid] * 30} for k, pid in enumerate(ex)}
    else:
        print("WARNING: no cases with >=3 windows left of -30 — skipping example panels", flush=True)

    with open(SUMMARY, "w") as f:
        json.dump(_sanitize(summary), f, indent=2)
    print(f"saved {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
