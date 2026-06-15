# NSRR — MESA Sleep Data Request (draft)

**Submit at:** https://sleepdata.org/data/requests/mesa/start
**Dataset:** MESA Sleep (Exam 5 ancillary; ~2,261 participants; 7-day wrist actigraphy + in-home PSG + sleep questionnaires; collected 2010–2013)

---

## Investigator information

- **PI:** Olivier Elemento, PhD
- **Title:** Professor of Physiology and Biophysics; Director, Englander Institute for Precision Medicine
- **Department / Institution:** Department of Systems and Computational Biomedicine, Englander Institute for Precision Medicine, Weill Cornell Medicine
- **Address:** 1305 York Avenue, Box 140, New York, NY 10021
- **Email:** ole2001@med.cornell.edu

## Project title

External validation of a wearable + electronic-health-record foundation model for short-horizon cardiovascular event prediction.

## Research use statement (~250 words)

Cardiovascular disease remains the leading global cause of death, and existing clinical risk scores require clinic visits and a small set of static inputs. Continuous physiological signals collected by consumer wearables — step counts, heart rate, sleep architecture — are now ubiquitous, but no validated on-device foundation model translates these into actionable cardiovascular risk.

We have developed a 13.3-million-parameter transformer foundation model (working name "PhoneFM") pretrained via self-supervised masked daily-vector reconstruction on 920,000 wearable + electronic-health-record windows from 12,453 All of Us Research Program participants with Fitbit data. In supervised fine-tuning, the model achieves an AUROC of 0.886 (95% CI 0.850–0.916) on a 30-day cardiovascular composite endpoint (atrial fibrillation + myocardial infarction + heart failure). Three pre-registered negative-control endpoints (skin neoplasm, refractive errors, dental caries) track near 0.5, supporting that the model captures wearable physiology rather than healthcare-utilization confounding.

We propose to externally validate this model on the MESA Sleep ancillary substudy. MESA Sleep provides the closest publicly available analog to our training inputs: 7-day wrist actigraphy, in-home polysomnography-derived sleep architecture, and demographic / clinical covariates in a well-characterized multi-ethnic cohort with adjudicated cardiovascular outcomes. We will map MESA Actiwatch and PSG-derived features to the model's daily-vector input format and report discrimination (AUROC, AUPRC) and calibration (Brier, calibration slope) on 1-year and 5-year incident cardiovascular endpoints, with person-cluster bootstrap confidence intervals and pre-registered subgroup analyses by sex, age, and race/ethnicity.

## Specific data requested

- **Wrist actigraphy raw signals (Actiwatch Spectrum)** — all available participants, 7-day windows, raw activity counts and light data
- **In-home polysomnography raw signals (EEG / EOG / EMG / ECG / respiratory channels)** — for an EHR-masked sensitivity analysis comparing PSG-derived sleep architecture against Fitbit-derived sleep stages
- **NSRR harmonized sleep covariates** — derived variables for sleep duration, efficiency, fragmentation, sleep stages, AHI, apnea-hypopnea events, arousals
- **Demographics file** — age, sex, race/ethnicity, BMI, smoking, comorbidities, medications available at Exam 5 (these populate the model's 9 demographic confounders)

CV outcome event files (incident MI, CHD, stroke, AFib, HF, mortality) will be requested in parallel from BioLINCC and the MESA Coordinating Center.

## Anticipated products

- One primary peer-reviewed publication (target: *Nature Medicine* or *npj Digital Medicine*) reporting external validation of the foundation model
- Pre-registered analysis protocol deposited prior to data access
- Open-source release of mapping code (NSRR Actiwatch → model input format) for reproducibility

## Data security

Data will be stored on Weill Cornell Medicine's HIPAA-aligned compute infrastructure. No re-identification will be attempted. All analyses will be reproducible from logged code; no participant-level data will leave the secure environment.

## Prior NSRR / BioLINCC data use

[TODO — list any prior MESA / SHHS / NSRR projects led by PI Elemento]

## Funding and conflicts of interest

[TODO — paste from `/Users/ole2001/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Administration/Funding and COI statements.md`]
