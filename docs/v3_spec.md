# PhoneFM v3 — Multi-Domain, Multi-Horizon Risk Foundation Model

**Status:** draft
**Author:** Olivier Elemento (with Claude)
**Date:** 2026-06-09
**Builds on:** v2 (commit `c62a31b` on `v2-dev`), val AFib AUROC 0.9229 after epoch 0

---

## 1. Why v3

v2 demonstrated that the wearable + EHR mixed-token architecture works for cardiovascular endpoints. To make this clinically meaningful and publishable beyond "another AUROC paper" we need to:

1. **Move past cardiovascular only.** Wearable signals (HR, steps, sleep) carry information for metabolic, mental health, sleep, and mortality endpoints — most of which have stronger phone-deployment stories than AFib (where Apple Watch already has FDA clearance).
2. **Predict at multiple time horizons.** A single 30-day horizon misses chronic disease onset (T2D, depression, OSA) where the signal builds over months or years.
3. **Establish baseline-conditional prediction.** Many endpoints are only meaningful in people who don't already have the condition (T2D in non-diabetics, depression in never-depressed). v2 predicts for everyone, which adds noise.

The v3 design lifts these three constraints without redesigning the backbone.

---

## 2. Endpoint panel

Five domains × three horizons = 15 (endpoint, horizon) prediction heads.

| Domain | Endpoint | ICD-10 / source | Baseline exclusion | Horizons |
|---|---|---|---|---|
| Mortality | All-cause death | `death` table | None | 30d, 180d, 365d |
| Metabolic | T2D incident | E11.\* | E10/E11 ever before window | 180d, 365d |
| Mental health | Major depression incident | F32.\*, F33.\* | F32/F33 ever before window | 180d, 365d |
| Cardiovascular | Composite (AFib + MI + HF) | I48.\*, I21–I22, I50.\* | None (kept for continuity with v2) | 30d, 180d, 365d |
| Sleep | OSA incident | G47.33 | G47.33 ever before | 365d only |

**Total heads:** 4 + 2 + 2 + 3 + 1 = **12 active heads** (cv_death dropped per v2; we may add back if mortality covers it).

**Why these five:**
- **Mortality:** zero-cost extension (death table already pulled); gold-standard endpoint that reviewers respect; gives the paper a survival-analysis story.
- **T2D incident:** Master 2022 *Nat Med* established daily steps → T2D in AoU. Direct precedent. Multi-task model that matches Master plus delivers AFib lets you cite both works as baselines you beat.
- **Depression incident:** lifetime prevalence ~17%, strong sleep-architecture signal, opens "mental health from phone" as a clinical story. High novelty for cardiovascular reviewers.
- **CV composite:** keeps continuity with v2 and v1 results, allows narrative "we extend the cardio model to broader risk."
- **OSA:** direct fit with the sleep stage features. Lower priority — drop if cohort size is insufficient after baseline exclusion.

---

## 3. Multi-horizon labels

For each (endpoint, horizon) pair, label = `1` if event occurs in `(end_date, end_date + horizon]`, else `0`. Labels are independent — a person who dies at day 200 has `mortality_30d=0`, `mortality_180d=0`, `mortality_365d=1`.

Implementation: `encode_window` returns a labels dict keyed by `f"{endpoint}_{horizon}d"`. The dataset shards store one int8 column per label. The model has one BCE head per (endpoint, horizon).

**Cost of longer horizons:** every additional horizon requires more right-censored data (we can only generate 365d labels for windows where `end_date + 365d ≤ fb_end`). Expected ~30% loss of windows per added horizon. v3 cohort filter becomes `observation_days ≥ 180 + 365 = 545` (was 210 in v2).

---

## 4. Cohort filtering per endpoint

For endpoints with baseline exclusions (T2D, depression, OSA), we generate per-endpoint label masks rather than dropping the window entirely. This preserves training signal for other endpoints from the same window.

Implementation: `encode_window` returns both `label_{ep}_{h}d` (0/1) and `mask_{ep}_{h}d` (1=use this sample for this endpoint, 0=skip). The loss respects the mask:

```python
loss_ep = (bce(logit, label) * mask).sum() / mask.sum().clamp(min=1)
```

For mortality and CV composite there is no mask (always 1).

---

## 5. Data extraction additions

### 5.1 `endpoint_concept_ids.json` expansion

New keys needed (ICD-10 source concept IDs, same format as existing afib/mi/hf):

- `t2d_incident`: ICD-10 E11.\* family
- `t2d_baseline_exclusion`: E10.\* + E11.\* (any diabetes)
- `dep_incident`: F32.\* + F33.\*
- `dep_baseline_exclusion`: F32.\* + F33.\* + F34.1 (dysthymia)
- `osa_incident`: G47.33 only
- `osa_baseline_exclusion`: G47.33

(BQ lookup: `SELECT concept_id, concept_name FROM concept WHERE vocabulary_id='ICD10CM' AND concept_code LIKE 'E11%'` then write JSON.)

**Effort:** ~1 hour. New file: `data_prep/build_endpoint_concept_ids.py`.

### 5.2 Tokenizer changes (`02_tokenizer_v3.py`)

Fork from `02_tokenizer_v2.py`. Changes:

1. Cohort filter to `observation_days ≥ 545` (180 day input + 365 day max horizon).
2. `fetch_cardio_events` becomes `fetch_endpoint_events` — pulls a union of all endpoint condition codes (incl. baseline-exclusion codes) and returns one row per (person, date, endpoint_tag).
3. `precompute_confounders` extended to include baseline T2D / depression / OSA flags (just additional `_fetch_condition_ever_before` calls).
4. `encode_window` builds 12 labels + 6 masks per window (4 endpoints with masks, mortality and CV without).
5. Labels for incident endpoints respect mask: if baseline flag is 1, mask=0 and label=0.

### 5.3 Dataset (`phonefm_dataset_v3.py`)

Fork from `phonefm_dataset_v2.py`. Add label_mask passthrough in `__getitem__` and `collate_v3`. Endpoint constant list expands from 5 to 12.

---

## 6. Architecture changes

**Minimal**, by design — the v2 backbone works.

- `PhoneFMV3Config.n_endpoint_heads = 12` (was 4 + composite)
- `PhoneFMV3Config.head_names`: explicit list of `(endpoint, horizon)` tuples
- Heads stay as independent `nn.Linear(d_model, 1)` — no horizon-conditioning needed since each head is dedicated
- Backbone identical: d_model=384, n_layers=6, n_heads=6, BatchNorm input norm preserved

**Optional:** add a horizon embedding to the input projection (single learnable vector per horizon, added to CLS). Theoretically helps when the same backbone serves multiple horizons. Pragmatic call: skip in v3 first run, add in v3.1 if needed.

---

## 7. Training plan

**Loss:**
```python
total = 0
for (ep, h) in head_names:
    pred = logits[(ep, h)]
    lbl  = labels[f"{ep}_{h}d"]
    msk  = masks.get(f"{ep}_{h}d", torch.ones_like(lbl))
    pw   = pos_weights[(ep, h)]
    bce  = F.binary_cross_entropy_with_logits(pred, lbl, pos_weight=pw, reduction='none')
    total = total + head_weights[(ep, h)] * (bce * msk).sum() / msk.sum().clamp(min=1)
```

**Per-head pos_weight:** keep the MAX_POS_WEIGHT=50 cap from v2.

**Head weights:** uniform 1.0 initially. After first epoch, downweight any head whose AUROC stays at chance.

**Optimizer:** AdamW unchanged; lr=1e-4, warmup=1000, cosine decay.

**Best metric:** sum of AUROCs across all "primary" heads (afib_30d + mortality_365d + t2d_365d + dep_365d). Avoids one-endpoint best-fitting.

**Epochs:** 10 (v2 settled by epoch 5; v3 has 3× more heads so allocate same total compute = ~3.5h on A100).

---

## 8. Evaluation plan

Per (endpoint, horizon) on val and test:
- AUROC, AUPRC
- Calibration (Brier score, reliability diagram)
- Decision curve analysis (clinical utility)

Subgroup analyses on test set (AoU has the demographics): race × age × sex × SES. This is the AoU strength most papers don't leverage.

Comparisons to clinical scores:
- AFib_30d vs. CHA2DS2-VASc
- CV composite vs. ASCVD risk equation
- T2D 365d vs. FINDRISC
- Depression vs. PHQ-2 if available

**External validation (post-v3):** UK Biobank Fitbit subset (n≈100K) — separate project after v3 numbers land.

---

## 9. Estimated timeline & cost

| Step | Time | A100 cost |
|---|---|---|
| Build `endpoint_concept_ids.json` (BQ + JSON) | 0.5 day | $0 |
| Fork tokenizer/dataset/model/train to v3 | 1 day | $0 |
| Re-tokenize on n1-highmem-2 (longer cohort = more chunks) | 3-4h | $0.50 |
| Train 10 epochs on A100 | ~3.5h | ~$13 |
| Generate val/test metrics + subgroup analyses | 0.5 day | $0 |
| **Total** | **~3 days work + ~$14 compute** | |

---

## 10. Open questions

1. **OSA inclusion.** AoU baseline OSA prevalence may swallow most of the cohort (sleep apnea is common). Run a quick `SELECT COUNT(DISTINCT person_id) FROM condition_occurrence WHERE condition_source_concept_id IN (G47.33 codes)` against the cohort before committing.
2. **Mortality horizon = 365d implies cohort filter ≥ 545 observation days.** That drops us from 12,100 to maybe 8,000-9,000 participants. Decide if that's acceptable, or shorten the max horizon to 180d (≥ 360 observation days).
3. **External concept ID list.** AoU sometimes ships pre-built phenotype concept sets via the workspace data. Worth a 30-min check before building ourselves.
4. **Head weight tuning.** Should we weight clinical-priority heads (mortality, AFib) higher than auxiliary (OSA)? Recommend uniform first, then re-weight based on epoch-1 AUROC.

---

## 11. Decision points before commit

- [ ] Pick the final endpoint list (this draft proposes 5 domains)
- [ ] Confirm max horizon (365d or 180d)
- [ ] Confirm baseline-exclusion strategy (per-endpoint mask vs. cohort drop)
- [ ] Pick best_metric formula for `best.pt` selection
- [ ] Decide whether to run v3 immediately or first validate v2 on test set
