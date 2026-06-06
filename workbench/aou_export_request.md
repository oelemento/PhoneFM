# All of Us model-export request template

Submit Day 1 of Week 2 (not Week 4). Review takes 5–10 business days.

## Workspace
- **Workspace name:** PhoneFM cardio prototype (Olivier Elemento, Cornell)
- **Tier:** Controlled
- **Application ID:** [fill from Workbench]

## What we want to export

A trained transformer checkpoint (~50M parameters, ~200 MB raw / ~25 MB after fp16 quantization) consisting of:
- Embedding tables for wearable + EHR tokens (vocabulary size ~8K, built from training-split participants only)
- 12-layer transformer backbone with rotary positional embedding on calendar-day index
- Linear classification head for 30-day cardiovascular event prediction

We are requesting export in two interchangeable forms:
1. **PyTorch state_dict** (`cardio_fm_v1.pt`) — for archival / re-conversion
2. **Core ML mlpackage** (`cardio_fm_v1.mlpackage`) — for direct iOS deployment

The two artifacts are bit-equivalent under fp16 (parity report attached).

**Specifically NOT exporting:** any per-participant data, cohort identifiers, raw signals, aggregates, embeddings of individual participants, or anything that could be linked back to All of Us individuals.

## Why the export is necessary

The model will be deployed on iPhones (Core ML format) for on-device inference. Patients will use their OWN HealthKit data as input; the model weights themselves contain no participant information.

## Verification steps we have taken

1. The model is trained on randomly-mixed-batch SGD with class-balanced sampling; no per-participant fine-tuning, no participant-level overfitting signal
2. **Membership-inference audit** on the trained checkpoint: train members vs held-out non-members AUROC < 0.55 (chance = 0.50), 95th-percentile per-participant loss difference < 0.02. Full report at `mia_audit.json`
3. **Train/val/test splits are participant-disjoint.** Vocabulary, decile reference distributions, and EHR feature spaces are built ONLY from training-split participants; no information from val/test participants enters the model
4. **Core ML parity verified:** `coreml_parity_report.json` documents max |Δlogit| < 0.05 across 20 random probes vs the PyTorch reference, confirming the exported `.mlpackage` faithfully represents the audited weights
5. We will NOT publish the model to a public repository; distribution is via App Store binary (compiled Core ML model embedded in app, not user-extractable)
6. Per All of Us PIA policy, no participant-level data will be referenced in the model card or documentation

## Risk mitigation

- Pre-export membership-inference audit (per-participant loss AUROC against held-out participants)
- Train-only vocabulary build (no val/test code leakage into the embedding table)
- Calibration gap check during training (mean predicted probability vs label rate per epoch) — confirms the model doesn't sharpen on memorized training participants
- Public model card describes architecture + training cohort SIZE only, not identities

## Output file inventory (for reviewer)

| File | Size | Contents |
|---|---|---|
| `cardio_fm_v1.pt` | ~200 MB | PyTorch state_dict (fp32 weights) |
| `cardio_fm_v1.mlpackage/` | ~25 MB | Core ML mlprogram (fp16 weights, iOS 18 target). Directory, not file. |
| `config.json` | <1 KB | Model config (d_model, n_layers, vocab_size, training hyperparams) |
| `model_card.md` | <1 MB | Public-facing description (no participant data) |
| `mia_audit.json` | <1 KB | Pre-export membership-inference audit results |
| `coreml_parity_report.json` | <1 KB | PyTorch vs Core ML logit-equivalence verification |

## Approval contact

Olivier Elemento (ole2001@med.cornell.edu) + co-PI as needed per All of Us policy.
