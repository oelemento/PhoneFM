# PhoneFM Workbench training pipeline

End-to-end scripts to train the cardio foundation model inside All of Us Researcher Workbench, audit it for membership-inference leakage, and request export.

**Order of operations (all in one Jupyter kernel, Python 3.10):**

| Step | Script | Output | Notes |
|---|---|---|---|
| 0 | `00_setup.py` | (pip installs) | Pin torch≥2.2, coremltools≥7.2, sklearn. **Restart the kernel after this finishes.** |
| 1 | `01_cohort_extraction.py` | `/tmp/cohort_base.parquet`, `/tmp/splits.json`, `/tmp/endpoint_concept_ids.json` | BigQuery cohort + 70/15/15 participant-level split. ~15-20K eligible participants. |
| 2 | `02_tokenizer.py` | `/tmp/vocab.json`, `/tmp/tokenized/*.parquet` | Train-only vocab build; 30-day windows with 14-day stride. |
| 3 | `03_dataset.py` | (module) | PyTorch `Dataset` over the parquet shards. Save as `phonefm_dataset.py` so `05_train.py` can import it. |
| 4 | `04_model.py` | (module) | GPT-style transformer + classification head. Save as `phonefm_model.py`. |
| 5 | `05_train.py` | `/tmp/phonefm_v1/best.pt` + bucket sync | ~12h on a single A100, default hyperparams target ~50M params. |
| 6 | `06_eval_mia.py` | `/tmp/phonefm_v1/mia_audit.json` | Pre-export membership-inference audit. Must pass before export request. |
| 7 | `07_coreml_convert.py` | `/tmp/phonefm_v1/cardio_fm_v1.mlpackage` + parity report | PyTorch → Core ML (fp16, iOS 18 target, ANE-friendly attention). ~10 min on CPU. |

**File-name caveat:** Python module names can't start with a digit. Save `03_dataset.py` and `04_model.py` in Jupyter as `phonefm_dataset.py` and `phonefm_model.py` respectively before running `05_train.py` (which imports them by name). The first two notebooks are run as cells, not imported, so they keep their numeric prefix.

**Cell ordering inside a single notebook is fine** — each numbered script can be a single Jupyter notebook with the cells in order. Cross-notebook globals (`endpoint_concept_ids`, `SPLITS`, `COHORT`, `VOCAB`) are kernel-resident; restart the kernel and you re-run from scratch.

**Compute target:** Workbench Standard GPU node (NVIDIA A100 40 GB). Budget ~$80–120 for training. The MIA audit is CPU + GPU light, ~30 min.

**After 06 passes:** submit the export request (template at `aou_export_request.md`) with `mia_audit.json` attached. Approval takes 5–10 business days. Once approved, convert `best.pt` → Core ML and drop into the iOS app's `Resources/` folder.
