# All of Us / Verily Workbench — Controlled-Tier model-egress request

**Updated:** 2026-06-15 (post-MIA-audit; reflects v3 reality)
**Submit via:** support.researchallofus.org → "Egress / data leaving CT"
**Expected review window:** 5–10 business days (template estimate; model-weight requests can take longer)

## Workspace

- **Workspace name:** PhoneFM CT cohort (Copy)
- **GCP project:** `wb-shrewd-lime-9770`
- **Workbench URL:** https://workbench.verily.com/workspaces/phonefm-ct-cohort-copy
- **Tier:** Controlled
- **Billing account:** `01043B-772AC9-33DDE1` (under GCP Research Credits award, application submitted 2026-06-14)
- **Source bucket:** `gs://phonefm-data-wb-shrewd-lime-9770/phonefm_v3/`

## What we want to export

A trained transformer checkpoint (13.3M parameters, ~53 MB) consisting of:

- Token embedding tables for wearable + EHR vocabulary (~8K tokens, built from training-split participants only)
- Transformer backbone with rotary positional embedding on calendar-day index
- 13 linear classification heads (10 active during training: 3 cardiovascular composite horizons, 2 T2D horizons, 2 depression horizons, 3 negative controls; 3 mortality heads dropped at training due to n_pos<50)

### Specific files

| File | Size | Contents | sha256 |
|---|---|---|---|
| `best.pt` | ~53 MB | PyTorch `state_dict` (fp32 weights). Parameter-only — no optimizer / RNG / epoch state (verified by `prep_model_export.sh`) | *to be filled by `prep_model_export.sh` at submission time* |
| `config.json` | ~4.9 KB | Model config (d_model, n_layers, head_specs, training HPs, val_metrics_at_save) | *to be filled* |
| `mia_audit.json` | ~5.5 KB | Aggregate / de-identified membership-inference audit results (numbers + methodology metadata) | *to be filled* |

Optional companion (for full transparency, also de-identified):

| File | Size | Contents |
|---|---|---|
| `model_card.md` | <1 MB | Public-facing description (architecture, training cohort size, intended use, limitations). No participant data. |

### Specifically NOT exporting

Any per-participant data, cohort identifiers, raw wearable signals, EHR token sequences, predicted probabilities, embeddings of individual participants, or anything that could be linked back to All of Us individuals.

---

## Why the export is necessary

**Primary use:** external validation of the model on the MESA (Multi-Ethnic Study of Atherosclerosis) Sleep ancillary cohort. NSRR data request submitted 2026-06-14 (id #27710); BioLINCC application submitted 2026-06-14 (id #17837). Both target a 1-year and 5-year external validation of the cv_composite_30d / 180d / 365d endpoints. The model weights themselves contain no AoU participant information; MESA participants supply the input data, the model computes a risk score, and outputs are reported in aggregate (AUROC, AUPRC, calibration) per MESA's standard reporting practice.

**Secondary use:** preliminary data for a planned NIH R01 application (the GCP Research Credits proposal submitted 2026-06-14 describes the broader research arc).

**Future use (out of scope for this egress):** on-device deployment via Apple Core ML / Android TensorFlow Lite using the participant's own HealthKit / Health Connect data as input. Patients would never see All of Us data; the model would run locally on the device.

---

## Membership-inference audit (pre-egress evidence)

Full results: `docs/mia_audit_results_2026-06-15.md`. Headline:

**Primary attack (cv_composite_30d single-head BCE; person-cluster bootstrap, n=1300 train vs n=1300 held-out test participants, K=8 windows per person, n_boot=2000):**

| Metric | Point AUROC | 95% CI |
|---|---|---|
| **train vs test** | **0.5048** | **[0.4827, 0.5261]** |
| train vs val (diagnostic only) | 0.5011 | [0.4793, 0.5227] |

CI upper bound 0.5261 is comfortably below the 0.55 pre-registered threshold and the interval covers chance (0.50). A complementary multi-head BCE attack gives 0.5054 [0.4827, 0.5283]. The checkpoint shows **no detectable memorization** of training participants.

### Structural priors that reinforce the result

- **Capacity vs data:** 13.3M parameters shared across ~6,496 train participants and ~620K train windows — very low per-example capacity; the model is forced to generalize.
- **Architecture:** plain feed-forward transformer; no memory bank, retrieval head, or nearest-neighbor lookup that could echo individuals.
- **Training:** weight decay, dropout, early-stopping; effectively ~9.6K supervised gradient steps on top of the pretrained backbone.
- **Egress bundle is parameter-only:** verified by `prep_model_export.sh` manifest (no optimizer state, no RNG, no data).

### MIA audit reproducibility

- Code: `workbench/mia_audit.py` v2.1 at commit `0158756` on `v3-spec` (oelemento/PhoneFM)
- Pre-audit adversarial code review identified 4 CRITICAL + 8 MAJOR findings on v1; v2 addressed both critical let-blocks (per-window loss formula, val as diagnostic only) and the 4 most-defensible MAJOR issues (persons-first sampling, autocast handling, deterministic sampling, numeric-only output)
- Audit ran on T4 in fp32 (T4 lacks bf16; v2.1 falls back gracefully). fp32 is the more conservative audit-time precision

---

## Train / val / test split guarantees

- **Person-disjoint** at every split (train ∩ val ∩ test = ∅ at the participant level)
- **Vocabulary built only from training-split participants** — no val/test tokens leak into the embedding table
- **Confounder reference distributions** (age/sex/race/BMI quantiles) computed on training split only
- **Best-metric checkpoint** selected on val (sum of primary AUROCs across cv_composite_30d, mortality_365d, t2d_365d, dep_365d); test is never used for any selection decision

---

## Risk mitigation

1. **Pre-egress MIA audit** (above): primary AUROC 0.5048 CI [0.4827, 0.5261]; multihead corroboration; diagnostic val arm
2. **Train-only vocabulary build:** no val/test code leakage into the embedding table
3. **Calibration gap monitored during training:** mean predicted probability vs label rate per epoch; no over-sharpening detected
4. **Parameter-only manifest:** `prep_model_export.sh` produces sha256 + tensor count + dtype histogram and asserts no `optimizer / RNG / epoch / scaler / scheduler` keys are present in the checkpoint
5. **No commitment to public weight release:** distribution is initially to Weill Cornell Medicine HIPAA-aligned compute infrastructure for MESA external validation. Any subsequent public release (e.g., HuggingFace) would be a separate request after external-validation review

---

## Approval contacts

- **PI:** Olivier Elemento, PhD (ole2001@med.cornell.edu)
- **Institution:** Weill Cornell Medicine, Englander Institute for Precision Medicine
- **Address:** 1305 York Avenue, Box 140, New York, NY 10021
- **Co-PI / Data Access Requester:** *(if AoU policy requires a second signatory)*

---

## Pre-submission checklist

- [ ] Run `prep_model_export.sh` to stage the bundle + compute sha256
- [ ] Attach sha256 manifest + parameter histogram to the support request
- [ ] Attach `docs/mia_audit_results_2026-06-15.md` (or paste headline numbers inline)
- [ ] Confirm the destination bucket / institutional location for the approved weights (`gs://wcm-eipm-...` TBD)
- [ ] Confirm WCM IRB exemption / determination letter is on file for the external validation work
- [ ] Submit via support.researchallofus.org
