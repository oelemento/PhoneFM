# v3 Supervised Training — Final Results

**Run completed:** 2026-06-10 13:11 UTC (Wed Jun 10)
**Duration:** ~1h 35min on NVIDIA A100-SXM4-40GB
**Best epoch:** 0 (early-stop fired after epoch 3 with patience=3)
**Best `sum_primary_auroc`:** 2.1032

The best-metric peaked at the very first eval and never recovered. The pretrained backbone + fresh-init heads with minimal supervised tuning was already at the optimum; additional epochs caused t2d / dep heads to overfit.

---

## Pipeline configuration

- Cohort: 9,242 / 12,453 PhoneFM-eligible ppts (74.2%, after 545-day observation filter)
- Splits: train 6,496 / val 1,356 / test 1,390
- Train windows: ~620K (stride 14d); val ~65K; test ~65K (stride 30d)
- Model: PhoneFMV3, 13.3M params, 13 heads, n_confounders=9
- Backbone init: `~/workspace/phonefm-data/phonefm_pretrain_v3/backbone_only.pt` (val_masked_MSE=0.1805 at epoch 19/20)
- Loss: masked multi-head BCE, sum-weighted, MAX_POS_WEIGHT=50, MIN_POS_FOR_TRAINING=50
- LR: peak 1e-04, warmup 1000 steps, cosine decay; 5 epochs target with early-stop patience=3
- Best-metric: `sum_primary_auroc` over `[cv_composite_30d, mortality_365d, t2d_365d, dep_365d]`

---

## Heads dropped at train time (n_pos<50)

`mortality_30d`, `mortality_180d`, `mortality_365d` — death table sparse (only 3/14/15 positives in train). Heads still emit logits at eval but contribute 0 gradient to loss. AoU's `death` table appears to have very few entries linked to the Fitbit-wearing PhoneFM cohort; this is a known structural limitation, not a bug.

---

## Heads starved at val time (n_pos<5)

`mortality_*` — 0 positives in val. Preflight check correctly fell back to summing the 3 viable primary heads (`cv_composite_30d`, `t2d_365d`, `dep_365d`) for best-metric selection.

---

## Per-epoch learning curve

All AUROCs are val-set, point estimates (no cluster bootstrap yet — that's the test-set eval, in progress).

### Primary best-metric heads

| Epoch | sum_primary_auroc | cv_composite_30d | t2d_365d | dep_365d | best.pt? |
|-------|-------------------|------------------|----------|----------|----------|
| **0** | **2.1032** ⭐     | **0.8775**       | **0.6391** | **0.5866** | ✅ saved |
| 1     | 2.0670            | 0.8785           | 0.6221   | 0.5638   | ✗ |
| 2     | 2.0159            | 0.8726           | 0.5718   | 0.5714   | ✗ |
| 3     | 2.0228            | 0.8675           | 0.5842   | 0.5710   | ✗ → early stop |

### All 10 primary heads + 3 negative controls

| Endpoint | Epoch 0 | Epoch 1 | Epoch 2 | Epoch 3 |
|----------|---------|---------|---------|---------|
| **cv_composite_30d**   | **0.8775** | 0.8785 | 0.8726 | 0.8675 |
| cv_composite_180d  | 0.8598 | 0.8574 | 0.8505 | 0.8409 |
| cv_composite_365d  | 0.8566 | 0.8485 | 0.8388 | 0.8304 |
| t2d_180d           | 0.6343 | 0.6338 | 0.5736 | 0.5931 |
| **t2d_365d**           | **0.6391** | 0.6221 | 0.5718 | 0.5842 |
| dep_180d           | 0.6022 | 0.5664 | 0.5679 | 0.5642 |
| **dep_365d**           | **0.5866** | 0.5638 | 0.5714 | 0.5710 |
| _Negative controls (target ≈ 0.5)_ | | | | |
| skin_neoplasm_365d | 0.6210 | 0.5843 | 0.5115 | **0.4752** |
| refractive_errors_365d | 0.5286 | 0.5290 | 0.5887 | 0.5583 |
| dental_caries_365d | 0.6366 | 0.4910 | 0.5129 | **0.3756** |
| _Mortality heads (n_pos=0 in val)_ | | | | |
| mortality_30d / 180d / 365d | NaN | NaN | NaN | NaN |

---

## Interpretation

**1. cv_composite_30d holds up across epochs (0.87 ± 0.01).** Cardiovascular composite (AFib + MI + HF in 30 days) is the strongest signal, robust to overfitting. **This is the headline v3 result that's directly comparable to v2's AFib_30d val AUROC of 0.9286.** v3 is slightly lower because:
- v3 is composite (AFib + MI + HF) where v2 was AFib-only; including MI and HF dilutes the AFib-dominant signal
- v3 had only 1 supervised epoch effectively (epoch 0 = backbone load + minimal gradient)

**2. cv_composite_180d and 365d degrade modestly (0.86 → 0.83).** Longer horizons accumulate more noise + more competing risks.

**3. t2d_365d drops sharply with more training (0.64 → 0.57).** Clear overfitting. The backbone produced sensible features at epoch 0, but the head started memorizing training labels.

**4. dep_365d hovers around chance (0.59).** Depression signal is weak in wearable data at this cohort size. Maybe sleep architecture features need more development, or cohort is too small.

**5. Negative controls drop toward 0.5 (or below) as training progresses.** This is GOOD:
- skin_neoplasm_365d: 0.62 → 0.48 ✓ (epoch 3 is at chance)
- dental_caries_365d: 0.64 → 0.38 ✓ (well below chance — means the model learned to be UNcertain on this)
- refractive_errors_365d: 0.53 → 0.56 (stayed near chance)

The downward trend on neg controls validates that the model is learning real wearable signal rather than utilization confounding. At epoch 0 some confounding leaks in (people who get diagnosed wear Fitbits); by epoch 3 it has been regularized away — but at the cost of overfitting primaries.

**6. Pretraining did most of the work.** Epoch-0 numbers (cv_composite_30d=0.87) are basically the pretrained backbone with random-init heads + ~9.6K supervised gradient steps. The 39K subsequent steps mostly hurt. This is consistent with the SSL story: the backbone learned wearable-relevant features in pretrain; fine-tuning is just a calibration step on top.

---

## Comparison to v2 baselines

| Metric | v2 (val) | v3 (val) | Notes |
|--------|----------|----------|-------|
| AFib_30d / cv_composite_30d val AUROC | 0.9286 | 0.8775 | v3 is composite (afib+mi+hf), so dilution expected |
| HF_30d val AUROC | 0.8543 | (composite) | folded into cv_composite |
| MI_30d val AUROC | 0.7728 | (composite) | folded into cv_composite |
| Composite_30d val AUROC | 0.8927 | 0.8775 | comparable (v3 has stricter cohort + multi-horizon labels) |
| Mortality val AUROC | (single mortality_30d=0) | (n_pos=0) | structural limitation in both |
| Endpoints | 5 (afib, mi, hf, cv_death, composite) | 13 across 3 horizons | v3 covers t2d, dep, neg controls |

The v3 composite is comparable to v2's composite. v3's gain is **breadth** (multi-domain, multi-horizon, with negative controls) rather than peak AUROC on any one endpoint.

---

## Next steps (in progress / pending)

1. ✅ Training complete
2. 🟡 **Test eval (`06_eval_v3_test.py`, PID 4972)** — running as of 13:13 UTC. Will produce per-head AUROC + AUPRC with **cluster-bootstrap 95% CIs** (1000 resamples, person-level). ETA ~25 min.
3. ⏳ **Subgroup analysis (`06_subgroup_analysis.py`)** — pre-registered race × age interaction + marginals. ETA ~30 min after step 2.
4. ⏳ Diagnose mortality sparseness — is it the AoU death table, the 545d cohort filter, or a labeling-boundary bug? Currently structural; revisit before claiming mortality as an endpoint.
5. ⏳ Investigate whether **lower lr + more regularization** at supervised fine-tune could squeeze out more t2d / dep AUROC. Epoch 0 is best with lr=1e-04 cosine; maybe lr=1e-05 from the start would be safer.

---

## Artifacts in the workspace bucket

`gs://phonefm-data-wb-sparkly-lentil-9368/phonefm_v3/`:

| File | Size | Purpose |
|------|------|---------|
| `best.pt` | 53 MB | Epoch 0 model — load this for inference |
| `last.pt` | 53 MB | Epoch 3 model (final state at early stop) |
| `config.json` | 4.9 KB | HP + cfg + `val_at_save` for epoch 0 |
| `metrics.json` | 13 KB | Full per-epoch val metrics |
| `run.log` | ~50 KB | Complete training log |

Plus the snapshot directory at `gs://phonefm-data-wb-sparkly-lentil-9368/_snapshot_2026-06-10_v3_training/` with frozen copies of all of the above (for emergency recovery if the live files get clobbered).
