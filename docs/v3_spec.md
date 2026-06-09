# PhoneFM v3 — Multi-Domain, Multi-Horizon Risk Foundation Model

**Status:** draft (v0.2 — OSA dropped per empirical check)
**Author:** Olivier Elemento (with Claude)
**Date:** 2026-06-09
**Builds on:** v2 (commit `c62a31b` on `v2-dev`), val AFib AUROC 0.9229 after epoch 0

**Changelog:**
- v0.5 (2026-06-09): finalize §13 decisions — empirical cohort check confirms 76% retention at 545d filter → keep 365d max horizon; per-endpoint masking adopted; v2 test eval mandated before v3 training; subgroup definitions JSON pre-registered.
- v0.4 (2026-06-09): incorporate v2 training lessons — drop epochs 10→5 with early stopping (v2 peaked at epoch 2), inherit numerical stability defaults that took v2 5 mid-training restarts to discover, add pre-launch sanity assertions, bootstrap CI eval, sum-of-primary-AUROCs as best metric, larger-backbone ablation. See §14.
- v0.3 (2026-06-09): adopt four enhancements per strategy review — self-supervised pre-training (stub committed at `workbench/04_pretrain_v3.py`), negative-control endpoints, discrete-time hazard framing, pre-registered subgroup analyses. PRS integration and iPhone deployment moved to future-work section.
- v0.2 (2026-06-09): drop OSA as a primary outcome (AoU prevalence 2.5% vs. true ~25% → detection bias dominates); move OSA to baseline confounder. Document empirical OSA prevalence query results. Address Critical feasibility-reviewer findings on cohort size empirical check, SNOMED descendant traversal, and multi-source phenotyping.
- v0.1 (2026-06-09): initial draft with 5 domains × 3 horizons.

---

## 1. Why v3

v2 demonstrated that the wearable + EHR mixed-token architecture works for cardiovascular endpoints. To make this clinically meaningful and publishable beyond "another AUROC paper" we need to:

1. **Move past cardiovascular only.** Wearable signals (HR, steps, sleep) carry information for metabolic, mental health, sleep, and mortality endpoints — most of which have stronger phone-deployment stories than AFib (where Apple Watch already has FDA clearance).
2. **Predict at multiple time horizons.** A single 30-day horizon misses chronic disease onset (T2D, depression, OSA) where the signal builds over months or years.
3. **Establish baseline-conditional prediction.** Many endpoints are only meaningful in people who don't already have the condition (T2D in non-diabetics, depression in never-depressed). v2 predicts for everyone, which adds noise.

The v3 design lifts these three constraints without redesigning the backbone.

---

## 2. Endpoint panel

Four primary domains × multiple horizons = **11 active heads**.

| Domain | Endpoint | Phenotype source | Baseline mask | Horizons |
|---|---|---|---|---|
| Mortality | All-cause death | `death` table (any death) | None | 30d, 180d, 365d |
| Metabolic | T2D first-recorded | Multi-source (ICD E11 + A1c ≥ 6.5 + RxNorm anti-diabetic) | Same multi-source ever before | 180d, 365d |
| Mental health | Depression first-recorded | Multi-source (ICD F32/F33 + antidepressant RxNorm) | Same multi-source ever before | 180d, 365d |
| Cardiovascular | Composite (AFib + MI + HF) | SNOMED descendant traversal (concept_ancestor) of root concepts | None (continuity with v2) | 30d, 180d, 365d |

**Total active heads:** 3 + 2 + 2 + 3 = **10 supervised heads with backprop** (cv_death dropped; OSA dropped — see §2.1) **+ 3 negative-control heads** (see §11.2) = **13 total heads at training time**.

**Why these four (post-OSA drop):**
- **Mortality:** zero-cost extension (death table already pulled); gold-standard endpoint reviewers respect; gives the paper a survival-analysis story.
- **T2D first-recorded:** Master 2022 *Nat Med* established daily steps → T2D in AoU. Direct precedent. Renamed from "incident" to "first-recorded" to be honest about AoU's right-truncation; multi-source phenotype matches eMERGE/PheKB standard.
- **Depression first-recorded:** lifetime prevalence ~17%, strong sleep-architecture signal, opens "mental health from phone" as a clinical story. Antidepressant evidence captures undertreated cases.
- **CV composite:** keeps continuity with v2 and v1 results, allows narrative "we extend the cardio model to broader risk." SNOMED descendant traversal restored to fix v2's source-concept-only undercount.

### 2.1 OSA: dropped as primary outcome, added as confounder

**Empirical query (2026-06-09 against `wb-silky-artichoke-2408.C2024Q3R8`):**

| Cohort | OSA diagnosed (G47.33 + SNOMED) | N | Rate |
|---|---|---|---|
| AoU Controlled Tier (whole CDR) | 16,082 | 633,547 | **2.54%** |
| PhoneFM cohort (first 5,000 pids) | 219 | 5,000 | **4.38%** |

True adult OSA prevalence per polysomnography studies is **~25–30%**. AoU's recorded prevalence is **~10%** of the true rate, consistent with the ~80–90% undiagnosed figure in the OSA literature. PhoneFM cohort enrichment (1.7× over CDR) reflects that Fitbit wearers skew healthcare-engaged, not that they have more disease.

**Decision:** Drop OSA from `head_names`. A "baseline-OSA-excluded → predict incident OSA in 365d" head would actually train the model to predict **who gets referred for and completes a sleep study**, not who has the disease. Detection bias would dominate the wearable signal because:
1. Undiagnosed OSA at baseline is silently 80% of the cohort.
2. Wearable signal (sleep fragmentation, HR variability) carries actual OSA information, *not* future-referral information.
3. The model would learn to predict referral, which is a healthcare-access proxy rather than a clinical outcome.

**Instead, OSA becomes a confounder:**
- Add `baseline_osa` (G47.33 + SNOMED descendants ever before window) to the 8-dim confounder vector (`precompute_confounders`).
- The confounder vector grows from 8 → 9 dims, requiring `PhoneFMV3Config.n_confounders = 9`.
- This stops the wearable signal from fighting against existing OSA in the other heads.

**Future option (not in v3):** if/when we have a polysomnography-confirmed subcohort, reframe OSA as "positive sleep study within 365d" using CPT 95810/95811 from `procedure_occurrence`. The model would predict who *should be* referred — a more clinically actionable phone-driven endpoint that avoids the detection-bias trap.

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

### 5.1 `endpoint_concept_ids.json` expansion (with SNOMED descendant traversal)

v2 used ICD-10 prefix matching (e.g., `LIKE 'E11%'`) on `condition_source_concept_id`. That works for AFib (AoU sites mostly submit ICD), but undercounts T2D and depression by 20–40% at SNOMED-shop sites (e.g., academic medical centers feeding AoU via Epic with SNOMED Reference Sets).

**Correct approach** uses `concept_ancestor` to traverse from a SNOMED root through all descendants, then unions with ICD-10 leaf codes:

```sql
-- T2D: SNOMED root 201826 (Type 2 diabetes mellitus) + ICD10CM E11.*
SELECT DISTINCT ca.descendant_concept_id AS cid
FROM `{CDR}.concept_ancestor` ca
JOIN `{CDR}.concept` c ON ca.ancestor_concept_id = c.concept_id
WHERE c.concept_id = 201826
   OR (c.vocabulary_id = 'ICD10CM' AND c.concept_code LIKE 'E11%')
```

Apply this pattern for all five primary endpoints + their baseline-exclusion variants. The label query then filters on `condition_concept_id IN UNNEST(...)` (the standard concept), not `condition_source_concept_id`. v2 used `_source_` because AFib is overwhelmingly ICD-coded; that choice ports poorly to T2D / depression and is corrected in v3.

**Codeset definitions (multi-source phenotype):**

| Endpoint | Diagnosis evidence | Lab evidence (`measurement`) | Drug evidence (`drug_exposure`) |
|---|---|---|---|
| `mortality` | `death.death_date IS NOT NULL` | – | – |
| `t2d_first` | E11.\* + SNOMED 201826 descendants (≥2 occurrences, ≥30 d apart) | HbA1c (LOINC 4548-4) ≥ 6.5% **OR** fasting glucose (LOINC 1558-6) ≥ 126 **OR** random glucose ≥ 200 | RxNorm class "Antidiabetic Agents" via `concept_ancestor` from 21600712 |
| `dep_first` | F32.\* + F33.\* + SNOMED 192080 descendants (≥1 occurrence) | PHQ-9 ≥ 10 (if available in `observation`) | RxNorm class "Antidepressants" via `concept_ancestor` from 21604686 |
| `cv_composite` | AFib (SNOMED 313217) + MI (SNOMED 4329847) + HF (SNOMED 316139) descendants | – (kept binary for v2 continuity) | – |
| `baseline_osa` (confounder) | G47.33 + SNOMED 4154290 descendants | – | – |

**Phenotype rule:** A subject is **first-recorded** at the earliest date of *any* of the three evidence types (Dx, Lab, Drug). Baseline-exclusion uses the same multi-source rule on the lookback period.

**Effort:** ~4 hours (was 1 hour). New file: `data_prep/build_endpoint_concept_ids.py` writes JSON with separate code lists per evidence type per endpoint. Includes a self-test that counts hits per endpoint per evidence source on the full cohort and warns if any source contributes <5%.

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

## 9. Estimated timeline & cost (v0.3, with enhancements)

| Step | Time | A100 cost |
|---|---|---|
| Build `endpoint_concept_ids.json` (BQ + JSON, multi-source phenotypes) | 0.5 day | $0 |
| Build `subgroup_definitions.json` (pre-registration) | 0.25 day | $0 |
| Fork tokenizer/dataset/model/train to v3 (incl. negative-control heads) | 1 day | $0 |
| Re-tokenize on n1-highmem-2 (longer cohort, new endpoint codesets) | 3-4h | $0.50 |
| **Pre-train backbone on wearable history (§11.1)** | **~1.5h** | **~$5** |
| Fine-tune 13-head supervised model from pretrained backbone (fewer epochs needed) | ~2h | ~$8 |
| Generate val/test metrics + subgroup analyses (§11.4) + DCA + Cox sensitivity (§11.3) | 0.5 day | $0 |
| **Total** | **~3 days work + ~$14 compute** | |

Pretraining + faster fine-tune nets out to roughly the same total cost as the v0.1 plan but delivers the transferable backbone + foundation-model framing as bonus.

---

## 10. Open questions

1. **OSA inclusion.** AoU baseline OSA prevalence may swallow most of the cohort (sleep apnea is common). Run a quick `SELECT COUNT(DISTINCT person_id) FROM condition_occurrence WHERE condition_source_concept_id IN (G47.33 codes)` against the cohort before committing.
2. **Mortality horizon = 365d implies cohort filter ≥ 545 observation days.** That drops us from 12,100 to maybe 8,000-9,000 participants. Decide if that's acceptable, or shorten the max horizon to 180d (≥ 360 observation days).
3. **External concept ID list.** AoU sometimes ships pre-built phenotype concept sets via the workspace data. Worth a 30-min check before building ourselves.
4. **Head weight tuning.** Should we weight clinical-priority heads (mortality, AFib) higher than auxiliary (OSA)? Recommend uniform first, then re-weight based on epoch-1 AUROC.

---

## 11. Enhancements adopted (in scope for v3)

Four strategy-review additions accepted into v3 scope; near-zero engineering cost on top of the §2–9 plan but materially strengthens the publication story.

### 11.1 Self-supervised pre-training

**Pretext task:** masked daily-vector reconstruction on 90-day wearable subwindows. 15% of valid daily-vector positions are zeroed and their `input_ids` replaced with `MASK_ID=4`; a regression head predicts the missing 11-dim vectors via MSE loss.

**Status:** stub committed at `workbench/04_pretrain_v3.py` on branch `v3-spec`. Smoke test at `workbench/04_pretrain_v3_smoke.py` verifies masked-MSE math, shape contracts, and gradient flow on synthetic shards.

**Backbone reuse:** `PhoneFMPretrainV3` uses identical layer names as `PhoneFMV2`, so the post-pretrain `backbone_only.pt` (without the `pretrain_head.*` keys) loads into the v3 supervised model via `state_dict(strict=False)`. Supervised heads initialize randomly on top of pretrained backbone.

**Why pretrain:**
- Backbone learns wearable distribution structure (circadian rhythm, sleep cycles, illness perturbations) from unlabeled data before being asked to predict labels.
- Strengthens "foundation model" framing — reviewer-recognized SSL → fine-tune pattern from BERT/GPT/ChronosFM.
- Fine-tuning converges faster than from-scratch (typically 1/3 the steps), so total pipeline cost is roughly neutral.
- Reusable backbone for v3.1, v4, and any future PhoneFM project — pays back across the program, not just this paper.

**Budget:** 1-2h on A100 for ~100K-200K steps, ~$4-6.

### 11.2 Negative control endpoints

Add three heads for conditions that *should not* be predictable from wearable signal. Same multi-source phenotype machinery as the primary endpoints, but the prior on AUROC is 0.5.

| Negative control | ICD-10 / SNOMED | Why this is "negative" |
|---|---|---|
| Skin malignant neoplasm | C43.\* + C44.\* + SNOMED 4112752 descendants | Skin cancer onset is not driven by wearable-detectable physiology. |
| Refractive errors of the eye | H52.\* + SNOMED 4218554 descendants | Vision changes are uncorrelated with HR/sleep/activity. |
| Dental caries | K02.\* + SNOMED 4210708 descendants | Oral health depends on diet and hygiene, not movement or HR. |

**Why this matters:** if the model predicts skin cancer at AUROC 0.7+, the wearable signal is functioning as a *healthcare-engagement proxy* (people who get diagnosed are people who see doctors, who tend to be Fitbit-wearing/EHR-engaged), not as actual disease information. Reviewers will ask this in any AoU paper. Pre-built negative controls preempt the critique.

**Expected outcome:** AUROC ≈ 0.5 ± 0.03 (95% CI) on each negative control, demonstrating the wearable signal carries real cardiovascular/metabolic/sleep information rather than utilization confounding.

**Implementation:** add three more entries to `endpoint_concept_ids.json`; three more `Linear(d_model, 1)` heads; loss participates in backprop at the same head_weight as primary endpoints (to test for spurious predictability). Zero new training time.

### 11.3 Discrete-time hazard framing

Frame the multi-horizon BCE structure as a **discrete-time survival model** rather than "binary classification at multiple horizons." For each (endpoint, horizon h), the head predicts:

$$\hat{\lambda}_{ep,h}(x) = P(\text{event occurs in horizon } h \mid \text{survived to end\_date}, x)$$

i.e., a discrete hazard. The right-censoring mask in §4 already implements the survival-analysis cohort logic correctly: `mask=0` for participants whose follow-up doesn't reach the horizon. Under independent censoring, this is the **Brown 1975 / Kvamme & Borgan 2019** discrete-time equivalent of a Cox proportional-hazards model and inherits its statistical properties.

**Why this matters:** "multi-horizon BCE classifier" sounds ad-hoc to clinical-statistics reviewers; "discrete-time hazard model with right-censoring" is a recognized survival framework with decades of theory behind it. Same math, different framing, much better paper.

**Implementation:** zero code change. One paragraph in the methods section + one pre-registered sensitivity analysis:
- Refit each `(endpoint, 365d)` head as a Cox PH model on the test set features, compare C-index to the discrete-time AUROC.
- If they agree within 0.01, the discrete-time choice is vindicated and the paper claims discrete-time hazard modeling as a methodological contribution.

### 11.4 Pre-registered subgroup analyses

AoU's demographic diversity is the strongest scientific asset most wearable papers fail to leverage. v3 pre-registers the following analyses *before* training starts so the paper cannot be accused of p-hacking subgroups post hoc.

**Marginal subgroups** (computed independently on test set):
- Race / ethnicity: Black, Hispanic/Latino, Asian, White, Other
- Age at end_date: <55, 55–65, >65
- Sex at birth: Male, Female
- SES proxy (AoU income quartile when available)

**One pairwise interaction** (highest policy relevance, lowest multiple-testing burden):
- Race × age stratum

**Per-subgroup metrics:**
- AUROC, AUPRC with bootstrap 95% CI (1,000 resamples)
- Brier calibration score
- Net benefit at clinical decision thresholds (DCA — see §8)

**Pre-registration mechanism:** the subgroup definition file `data_prep/subgroup_definitions.json` is committed BEFORE the test set is touched. Diff against this file is checked in the eval script. Required for *Nature Medicine* methodological rigor.

**Implementation:** one new eval script `06_subgroup_analysis.py`. Runs on the saved `best.pt` after training completes — separate from training loop, no impact on training time.

---

## 12. Future work (post-v3)

Two enhancements reviewed and deferred to v3.1+ to keep v3 scope tractable.

### 12.1 Polygenic risk score integration

AoU has whole-genome data for ~250K participants, with overlap into the PhoneFM cohort. Published-and-validated PRS exist for AFib, T2D, depression, and CAD (PGS Catalog IDs PGS000727, PGS000014, PGS000731, PGS000018, etc.).

**Plan:** add PRS as additional confounder dims (n_confounders 9 → 13+). Two analyses become possible:
- Show wearable signal adds AUROC *beyond* genetic risk (the strong reviewer ask)
- Mendelian-randomization-style causal analysis on T2D and depression using AoU genetics — could elevate v3.1 from prediction to causal inference

Cost: ~3 days work + zero new compute (PRS are precomputed). Target: v3.1.

### 12.2 iPhone deployment with Core ML

Convert `best.pt` to Core ML format. Benchmark on iPhone hardware:
- Inference latency (target <100ms on A17)
- Memory footprint (target <50MB)
- Battery drain over 24h of continuous prediction

Two days of swift + CoreML work, but this is the **differentiator** that separates this from every other academic wearable foundation model paper. No big-tech academic group is publishing actual on-device deployment numbers.

Target: write-up phase, after v3 trains and validates.

### 12.3 External validation cohorts

- **UK Biobank wrist accelerometer** subset (~100K) for EHR-conditional path only (UKB uses Axivity AX3, not Fitbit, so wearable features don't transfer)
- **Sage Bionetworks Fitbit cohort** (My Data Helps) for true Fitbit external validation if data access can be arranged

Both are post-v3 paper-prep work.

---

## 13. Decision points — ALL RESOLVED (2026-06-09)

- [x] **Endpoint list = 4 primary domains + 3 negative controls = 13 heads** (§2 + §11.2). Final list:
  - mortality_30d, mortality_180d, mortality_365d
  - t2d_180d, t2d_365d (first-recorded, multi-source phenotype)
  - dep_180d, dep_365d (first-recorded, multi-source phenotype)
  - cv_composite_30d, cv_composite_180d, cv_composite_365d (AFib + MI + HF descendants via concept_ancestor)
  - skin_neoplasm_365d, refractive_errors_365d, dental_caries_365d (negative controls)

- [x] **Max horizon = 365d** (§13.1 empirical evidence below)

- [x] **Baseline-exclusion strategy = per-endpoint mask** (§4 already specifies this). A participant with baseline T2D contributes mask=0 to T2D heads but mask=1 to all other heads, preserving training signal across non-affected endpoints. Cohort-level drop was rejected because it would lose ~30% of windows to a single endpoint's exclusion.

- [x] **best_metric = sum of primary endpoint AUROCs** (§14.10)

- [x] **Pre-v3: validate v2 on test set FIRST** (§13.2 below)

- [x] **Subgroup definitions JSON pre-registered now** (§13.3 below — `data_prep/subgroup_definitions.json` to be committed before any test access)

### 13.1 Empirical cohort-size results (driving the 365d max horizon decision)

Query against `wb-silky-artichoke-2408.C2024Q3R8.heart_rate_minute_level`, GROUP BY person_id, filtered to PhoneFM cohort (12,453 pids):

| Observation-span filter | Cohort retained | Retention | Notes |
|---|---|---|---|
| ≥ 210d (v2 baseline) | 12,100 | 100% | Current v2 cohort |
| ≥ 360d | 10,381 | 86% | If max horizon = 180d |
| ≥ 450d | 9,759 | 81% | If max horizon = 270d |
| **≥ 545d** (v3 baseline) | **9,242** | **76%** | **Adopted: 180d input + 365d max horizon** |
| ≥ 730d | 7,966 | 66% | Rejected: 24% loss too aggressive |

Distribution stats: median span 1,009 days (~2.8 years), p25 = 529d, p75 = 1,821d. Median actual days-with-data within span = 690 (~68% data density). The PhoneFM cohort is unusually long-lived because we already require ≥180d as a v2 cohort filter.

**Resolution:** 365d max horizon retains 9,242 participants. After 80/10/10 split: ~7,400 train / ~920 val / ~920 test. Sufficient for multi-task training across 10 active heads + 3 negative controls.

### 13.2 Run order: validate v2 on test, THEN start v3

Rationale:
- v2 test eval is ~30 min vs. v3 full pipeline ~5-6h. Cheap to do first.
- v2's val AFib AUROC 0.9285 is the headline claim of any current paper. If it doesn't hold on test, we have a generalization problem v3 won't fix and the work should pause for diagnosis.
- v2 test numbers are the BASELINE that v3 must beat. Paper requires both.
- The v2 best.pt checkpoint is locked at epoch 2 (afib_auroc=0.9285, saved 2026-06-09 18:19). Stable for downstream eval.

**Sequence:**
1. v2 training finishes (~30 min from this writing)
2. Run `06_eval_v2_test.py` on the v2 test shards using the saved `best.pt`. Report per-endpoint AUROC + bootstrap 95% CI (per §14.8). Expected runtime <30 min on A100.
3. If v2 test AFib AUROC ≥ 0.85 (some generalization gap is normal): proceed to v3. If below 0.85: pause, investigate generalization gap before any v3 work.
4. Pretrain backbone via `04_pretrain_v3.py` (~1.5h)
5. Re-tokenize for v3 endpoints + new cohort filter (~3h on n1-highmem-2)
6. Train v3 supervised (~3h, 5 epochs)
7. Eval v3 on test with subgroup + DCA + Cox sensitivity analyses

Total time to v3 results: ~3 days work + ~12h compute, $15-25.

### 13.3 Pre-registered subgroup definitions

To be saved as `data_prep/subgroup_definitions.json` BEFORE any test-set evaluation script is written. SHA1 of this file is checked in the eval script; if it has changed after first commit, the eval is considered invalid.

```json
{
  "race": {
    "Black or African American": [8516],
    "Hispanic or Latino": [38003563, 38003564],
    "Asian": [8515],
    "White": [8527],
    "Other": [8557, 8657, 0]
  },
  "age_at_end_date": {
    "<55": [0, 54],
    "55-65": [55, 65],
    ">65": [66, 120]
  },
  "sex_at_birth": {
    "Female": [45878463],
    "Male": [45880669]
  },
  "ses_income_quartile": {
    "Q1 (lowest)": [1, 1],
    "Q2": [2, 2],
    "Q3": [3, 3],
    "Q4 (highest)": [4, 4]
  },
  "interactions": [
    "race × age_at_end_date"
  ],
  "metrics_per_subgroup": [
    "AUROC", "AUPRC", "Brier", "calibration_intercept", "calibration_slope"
  ],
  "bootstrap_n": 1000,
  "ci_level": 0.95
}
```

Race concept IDs are AoU's standard `race_concept_id` values. Age is computed as `(end_date - person.birth_datetime).years` at window encoding time. Sex from `person.sex_at_birth_concept_id`. SES from AoU's `income_concept_id` mapped to quartiles per AoU PPI documentation.

Per-subgroup reporting threshold: any cell with N < 30 is reported as "N/A (insufficient sample)" rather than producing low-confidence AUROC point estimates.

---

## 14. v2 lessons learned (must apply in v3)

v2 trained successfully (AFib val AUROC 0.9285 at epoch 2, composite 0.8966 at epoch 0) but required 5 mid-training restarts and several failed runs to converge on stable hyperparameters and architectural defenses. The lessons below are mandatory inputs for v3 — not preferences.

### 14.1 Reduce epochs 10 → 5 with early stopping

v2 epoch-by-epoch AFib val AUROC: 0.9229 → 0.9202 → **0.9285** → 0.9105 → 0.9153 → 0.9248 → 0.9190 → 0.9248 (still in epoch 7 at write time). Peak at epoch 2; epochs 3-7 wasted ~$8 of A100 time learning nothing on the primary metric. Composite peaked at epoch 0. HF peaked at epoch 0.

**v3 change:** `HP['epochs'] = 5`. Add early stopping: stop if best metric hasn't improved for 3 consecutive epochs. Saves ~$6 and 90 minutes per training run. With pre-training (§11.1) initializing the backbone, even 3 epochs may suffice.

### 14.2 Inherit ALL v2 numerical stability defaults

v2 NaN-trained 1100+ steps before the run was diagnosed and killed. Fixes that turned a NaN-spewing run into a converging one:

```python
HP = dict(
    lr=1e-4,                  # was 3e-4 in initial v2; 3e-4 + multi-head BCE diverged
    warmup_steps=1000,        # was 500; gentler ramp
    weight_decay=0.15,        # was 0.1; v2 overfit HF/composite by epoch 3
    dropout=0.15,             # was 0.1; same reason
    grad_clip=1.0,
    balance_on=None,          # DO NOT enable WeightedRandomSampler; pos_weight in BCE
                              # alone handles class imbalance. v2 had both = double
                              # balancing → NaN at step 50.
    MAX_POS_WEIGHT=50,        # cap pos_weight to prevent BCE explosion when n_pos
                              # is tiny (v2 cv_death had n_pos=3, raw pw=212,797)
    MIN_POS_FOR_TRAINING=50,  # drop any head with fewer than 50 train positives
                              # from the loss entirely (head_weight=0)
)
```

Plus model-side defenses inherited from v2 `phonefm_model_v2.py`:
- `BatchNorm1d` on wearable input (raw `steps` range 0-100k otherwise overflows bf16)
- `BatchNorm1d` on confounder input (raw age/BMI/SBP same problem)
- `nan_to_num` defensive zero in `PhoneFMV2Dataset.__getitem__` for any wearable_feats or confounders that slip through with NaN

### 14.3 Empirically validated stability training recipe

bf16 autocast with `cuda_autocast` from `torch.cuda.amp` (NOT `torch.amp.autocast` — doesn't exist on AoU image's torch 2.0.1). Loss is internally upcast to fp32 by `binary_cross_entropy_with_logits`. AdamW with `(0.9, 0.95)` betas. Cosine schedule.

### 14.4 Test a larger backbone as v3.1-A vs v3.1-B comparison

v2's 13.3M-param backbone plateaued at AFib AUROC ~0.93 by epoch 2 and never improved. This is consistent with *capacity* being the binding constraint, not training duration. v3 with 13 heads vs v2's 4 heads demands more representation per parameter.

**Sub-ablation in v3.1:** train both `d_model=384, n_layers=6` (v2 default, 13M params) and `d_model=512, n_layers=12` (v1 scale, ~52M params) with the same pre-trained backbone + supervised heads. If the larger backbone gains ≥0.01 AUROC on AFib or mortality, it's worth the ~2× training cost.

### 14.5 Add wearable_feats nonzero assertion at tokenizer start

v2's "all-zero wearable_feats" failure (cohort timestamp midnight mismatch — see v3 spec changelog) wasted 1h45m + ~$5 of compute. The defense costs five lines:

```python
# In 02_tokenizer_v3.py, after writing the first shard:
first_shard = OUT_DIR / f"train_0000.parquet"
df = pd.read_parquet(first_shard)
sample_feats = [np.frombuffer(df.iloc[i]['wearable_feats'], dtype=np.float32)
                for i in range(min(10, len(df)))]
max_val = max(arr.max() for arr in sample_feats)
assert max_val > 100, f"wearable_feats all near-zero (max={max_val}). " \
                      f"Tokenizer is producing degenerate output — abort."
print(f"sanity OK: wearable_feats max = {max_val:.1f}")
```

This catches any future "tokenizer silently broke" class of bug in the first shard, not 6 hours into the full run.

### 14.6 NaN guard at training step level

```python
# In the training loop, after loss.backward():
if not torch.isfinite(loss).item():
    print(f"NaN detected at step {step}; halting", flush=True)
    raise RuntimeError("NaN loss")
```

v2 produced NaN loss for 1,100+ training steps before we noticed in the log. Hard halt prevents an entire wasted run.

### 14.7 Environment setup script

v2 cost 30 min on `numpy>=2.0` (broken vs torch 2.0.1) → `numpy<2` (broken vs pyarrow shipped against numpy 2). `setup_env.sh` to be sourced at the top of every training script:

```bash
#!/bin/bash
set -e
pip install --quiet 'numpy<2'
pip install --quiet --force-reinstall --no-deps pyarrow
python3 -c "import torch, numpy, pyarrow; \
  print('numpy', numpy.__version__, 'torch', torch.__version__, \
        'pyarrow', pyarrow.__version__); \
  assert torch.from_numpy(numpy.zeros(3)).sum().item() == 0, 'torch<->numpy broken'; \
  import pandas; pandas.read_parquet('/dev/null') if False else None"
```

Idempotent: costs nothing on a clean env, fixes the v2 compat issues once.

### 14.8 Bootstrap CIs for low-positive endpoints

v2 MI val AUROC bounced 0.7259 → 0.7728 → 0.7657 → 0.7984 → 0.7796 → 0.8306 across epochs purely from sampling variance (643 positives in val). v3 will have endpoints with substantially fewer positives:
- mortality_30d: estimated ~50 positives (5% of v2 cohort with 30d horizon → ~600, but excluding short-followup people → ~50)
- depression_180d: estimated ~200 positives
- negative controls: by construction ~baseline rate (~1-5%)

Raw point-estimate AUROC is meaningless without uncertainty quantification:

```python
from sklearn.utils import resample
def bootstrap_auc(y_true, y_pred, n_boot=1000, ci=0.95):
    aucs = []
    for _ in range(n_boot):
        idx = resample(np.arange(len(y_true)))
        if y_true[idx].sum() < 5: continue   # skip degenerate resamples
        aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    lo, hi = np.quantile(aucs, [(1-ci)/2, 1-(1-ci)/2])
    return np.median(aucs), lo, hi
```

Report `AUROC (95% CI)` for every (endpoint, horizon) pair in the val/test log.

### 14.9 Pre-launch checklist

v2 took 5 mid-training restarts to find stable hyperparams. v3 must verify the following BEFORE the 5-hour training run starts:

- [ ] Smoke test on tokenizer's first shard (wearable_feats nonzero, all label columns present, masks make sense)
- [ ] 100-step dry run with `batch_size=4, max_steps=100, num_workers=0` and verbose logging
- [ ] Code review on every new script ≥50 lines (the lesson from this very session — v3 already costs reviewer subagent + code reviewer subagent runs)
- [ ] All four critical reviewer findings on the pretrain code (C1, C2, M3, M4) verified fixed via re-run of `04_pretrain_v3_smoke.py`
- [ ] Tokenizer-level wearable_feats assertion passes
- [ ] NaN-step guard installed in training loop

### 14.10 best_metric = sum of primary endpoint AUROCs

v2 used `best_metric=afib_auroc` because AFib was the cleanest signal and one of two primary endpoints. v3 has 4 primary-domain (endpoint, horizon) pairs that we care about equally:

```python
PRIMARY_HEADS = ['afib_30d', 'mortality_365d', 't2d_365d', 'dep_365d']
HP['best_metric_formula'] = 'sum_primary_auroc'
# In the eval block:
best_score = sum(val_m[f'{h}_auroc'] for h in PRIMARY_HEADS
                 if not math.isnan(val_m[f'{h}_auroc']))
```

This prevents the `best.pt` selection from chasing one endpoint at the expense of others. Sum-of-AUROCs is a stable, monotone metric across all four heads.

### 14.11 Training time + cost expectations from v2

v2 actuals (vs the prediction in v2 spec):
- Predicted: ~17h on A100
- Actual: ~3.4h (5x faster — seq_len=512 vs predicted 4096; smaller backbone)

v3 actuals will track v2's ~3-4h despite adding 13 heads (pretrain initializes backbone faster + we'll cut epochs 10→5). Budget **~$15-20** of A100 time, not $50+. Stop the run early if metrics plateau before epoch 5.
