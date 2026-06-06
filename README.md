# PhoneFM

Phone-resident cardio foundation model — trained on All of Us Researcher Workbench (Fitbit + EHR cohort), deployed on iPhone via HealthKit + Core ML for on-device inference.

A 4-week prototype within the LLM-in-a-Box venture. The patient-facing arm: every iPhone runs its own small foundation model on its own HealthKit data, surfaces a weekly cardio-risk delta in plain language, and never sends raw signals off the device.

## Status

Prototype scaffolding. Training has not yet run on the Workbench — pipeline scripts are reviewed and ready to paste into a Jupyter kernel.

## Repo layout

```
docs/
  PLAN.md                  4-week prototype plan
workbench/                 All of Us Researcher Workbench notebooks (Python 3.10)
  README.md                run order
  00_setup.py              pinned pip install + capability check
  01_cohort_extraction.py  BigQuery cohort + train/val/test split
  02_tokenizer.py          wearable + EHR token vocabulary
  03_dataset.py            PyTorch Dataset over parquet shards
  04_model.py              ~50M-param transformer
  05_train.py              training loop (A100, ~12 h)
  06_eval_mia.py           membership-inference audit
  07_coreml_convert.py     PyTorch → Core ML mlpackage
  aou_export_request.md    model-export approval template
ios/                       iPhone app (SwiftUI + HealthKit + Core ML)
  README.md                Xcode setup
  PhoneFMApp/              app source
```

## Constraints

- Participant data never leaves the All of Us Workbench.
- Only the trained model weights leave, after individual review (5–10 business days).
- Pre-export membership-inference audit must pass before submission.
- iOS app target: iPhone running iOS 18+ (Apple Foundation Models framework for the on-device explanation layer).

## Not a medical device

Research preview. Does not diagnose, treat, or replace care from a clinician.
