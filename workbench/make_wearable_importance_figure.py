"""
make_wearable_importance_figure.py — publication figure summarising the wearable
importance ablations (test set, cv_composite_30d). Numbers are the aggregate,
de-identified ΔAUROC point estimates + 95% person-cluster bootstrap CIs already
transcribed into docs/v3_test_analysis_2026-06-11.md (§15-§18). No raw data.

Panel A — SUFFICIENCY: how much of the full model do wearables recover?
Panel B — PER-STREAM importance, triangulated across three estimators.

Out: figures/wearable_importance.png
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
OUT = HERE.parent / "figures" / "wearable_importance.png"
OUT.parent.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
})

# ---------------------------------------------------------------------------
# Data (cv_composite_30d; full-model AUROC = 0.8857). From §15-§17.
# ---------------------------------------------------------------------------
# Panel A: AUROC under progressive removal (sufficiency ladder).
suff_labels = [
    "Full model\n(EHR + wearables\n+ demographics)",
    "Phone model\n(wearables +\ndemographics)",
    "Wearables\nonly",
]
suff_vals = [0.8857, 0.8500, 0.6789]
suff_colors = ["#2f4b7c", "#ef8a43", "#b8bcc4"]

# Panel B: per-stream importance (ΔAUROC) under three estimators.
streams = ["Steps", "Heart-rate", "Sleep"]
estimators = [
    "Unique – full model (leave-one-out)",
    "Alone – full model (add-one-in)",
    "Unique – phone model, no EHR (leave-one-out)",
]
# point, lo, hi for [steps, hr, sleep] x [LOO_full, addin, LOO_phone]
pt = {
    "Steps":      [0.0005, 0.0034, -0.0001],
    "Heart-rate": [0.0147, 0.0136, 0.0308],
    "Sleep":      [0.0157, 0.0105, 0.0275],
}
lo = {
    "Steps":      [-0.0035, -0.0001, -0.0056],
    "Heart-rate": [0.0060, 0.0011, 0.0139],
    "Sleep":      [-0.0004, -0.0014, 0.0014],
}
hi = {
    "Steps":      [0.0078, 0.0078, 0.0058],
    "Heart-rate": [0.0322, 0.0292, 0.0480],
    "Sleep":      [0.0292, 0.0228, 0.0511],
}
est_colors = ["#4c72b0", "#55a868", "#c44e52"]

# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(11.2, 4.7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.7], wspace=0.32,
                      left=0.07, right=0.985, top=0.86, bottom=0.20)

# ---- Panel A: sufficiency ladder -----------------------------------------
axA = fig.add_subplot(gs[0, 0])
xA = np.arange(len(suff_vals))
barsA = axA.bar(xA, suff_vals, width=0.62, color=suff_colors,
                edgecolor="black", linewidth=0.7, zorder=3)
axA.axhline(0.5, ls=(0, (4, 3)), lw=1.0, color="#888", zorder=1)
axA.text(len(suff_vals) - 0.5, 0.5 + 0.006, "chance", ha="right", va="bottom",
         fontsize=8.5, color="#777")
for x, v in zip(xA, suff_vals):
    axA.text(x, v + 0.006, f"{v:.3f}", ha="center", va="bottom",
             fontsize=10, fontweight="bold")
# arrow annotation: phone vs full gap
axA.annotate("", xy=(1, 0.8500), xytext=(0, 0.8857),
             arrowprops=dict(arrowstyle="-", lw=0.8, color="#444", ls=":"))
axA.text(0.5, 0.905, "−0.036 AUROC\n(EHR removed)", ha="center", va="bottom",
         fontsize=8.6, color="#333")
axA.set_xticks(xA)
axA.set_xticklabels(suff_labels, fontsize=9)
axA.set_ylim(0.48, 0.95)
axA.set_ylabel("Test AUROC (cv 30-day)")
axA.set_title("a  Wearables are sufficient on a phone", loc="left",
              fontweight="bold", fontsize=12)
axA.set_yticks(np.arange(0.5, 0.95, 0.1))

# ---- Panel B: per-stream importance --------------------------------------
axB = fig.add_subplot(gs[0, 1])
nE = len(estimators)
group_w = 0.78
bw = group_w / nE
xB = np.arange(len(streams))
for j in range(nE):
    vals = [pt[s][j] for s in streams]
    los = [pt[s][j] - lo[s][j] for s in streams]      # lower error length
    his = [hi[s][j] - pt[s][j] for s in streams]      # upper error length
    sig = [(lo[s][j] > 0 or hi[s][j] < 0) for s in streams]
    offs = (j - (nE - 1) / 2) * bw
    xpos = xB + offs
    # solid for significant, hatched/translucent for non-significant
    colors = [est_colors[j] if sig[k] else est_colors[j] for k in range(len(streams))]
    alphas = [1.0 if sig[k] else 0.42 for k in range(len(streams))]
    for k in range(len(streams)):
        axB.bar(xpos[k], vals[k], width=bw * 0.92, color=est_colors[j],
                alpha=alphas[k], edgecolor="black", linewidth=0.6, zorder=3,
                hatch="" if sig[k] else "///")
    axB.errorbar(xpos, vals, yerr=[los, his], fmt="none", ecolor="#222",
                 elinewidth=1.0, capsize=2.6, zorder=4)
    # significance star above the CI
    for k in range(len(streams)):
        if sig[k]:
            axB.text(xpos[k], hi[streams[k]][j] + 0.0012, "*", ha="center",
                     va="bottom", fontsize=13, color="#222", zorder=5)

axB.axhline(0, lw=1.0, color="black", zorder=2)
axB.set_xticks(xB)
axB.set_xticklabels(streams, fontsize=11, fontweight="bold")
axB.set_ylabel("Wearable importance  (ΔAUROC)")
axB.set_ylim(-0.012, 0.058)
axB.set_title("b  The signal is in heart-rate and sleep — not step count", loc="left",
              fontweight="bold", fontsize=12)
legend_handles = [Patch(facecolor=est_colors[j], edgecolor="black", linewidth=0.6,
                        label=estimators[j]) for j in range(nE)]
legend_handles.append(Patch(facecolor="white", edgecolor="#222", hatch="///",
                            label="CI includes 0 (n.s.)"))
axB.legend(handles=legend_handles, fontsize=8.4, loc="upper left",
           frameon=False, handlelength=1.4, labelspacing=0.35)
axB.text(0.99, 0.97, "* 95% CI excludes 0", transform=axB.transAxes,
         ha="right", va="top", fontsize=8.5, color="#444")

fig.suptitle(
    "Wearable streams in the phone-resident cardiovascular model "
    "(All of Us test set, 13.3M-parameter model)",
    fontsize=12.5, fontweight="bold", y=0.985)

fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"saved {OUT}")
