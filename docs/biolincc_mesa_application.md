# BioLINCC — MESA Application (draft)

**Submit at:** https://biolincc.nhlbi.nih.gov/studies/mesa/ → "Request Data"
**Account:** account creation required at biolincc.nhlbi.nih.gov
**Dataset:** MESA (parent study, Exams 1–6, events updated through CY2020 in the public release; ancillary substudies handled separately — see MESA CC email)

---

## Investigator information

- **PI:** Olivier Elemento, PhD
- **Title:** Professor of Physiology and Biophysics; Director, Englander Institute for Precision Medicine
- **Department / Institution:** Department of Systems and Computational Biomedicine, Englander Institute for Precision Medicine, Weill Cornell Medicine
- **Address:** 1305 York Avenue, Box 140, New York, NY 10021
- **Email:** ole2001@med.cornell.edu

## Project title

External validation of a wearable + electronic-health-record foundation model for short-horizon cardiovascular event prediction using the MESA cohort.

## Research proposal (~500 words)

**Background.** Existing cardiovascular risk tools (ASCVD Pooled Cohort Equations, CHA₂DS₂-VASc) require clinic visits and rely on a small set of static inputs. Consumer wearables now provide continuous physiological data — step counts, heart rate, and sleep architecture — to hundreds of millions of users, but no validated on-device foundation model has been shown to translate these signals into actionable cardiovascular risk estimates with adjudicated outcomes.

**Preliminary work.** We have developed a 13.3-million-parameter transformer foundation model pretrained via self-supervised masked daily-vector reconstruction on 920,000 wearable + electronic-health-record windows from 12,453 All of Us Research Program participants with concurrent Fitbit data. After supervised fine-tuning the model achieves an AUROC of 0.886 (95% CI 0.850–0.916) on a 30-day cardiovascular composite endpoint (incident atrial fibrillation, myocardial infarction, or heart failure), with horizon decay to 0.849 at 365 days. Three pre-registered negative-control endpoints (skin neoplasm, refractive errors, dental caries) track near 0.5, supporting the interpretation that the model captures wearable physiology rather than healthcare-utilization confounding. Per-stream ablation localizes the wearable signal to heart-rate and sleep architecture (each contributing roughly +0.03 AUROC in the phone-deployable, electronic-health-record-masked context), with step counts adding essentially nothing once heart rate and sleep are present.

**Aims for this application.**

1. **External discrimination and calibration.** Apply the trained model to the MESA cohort and report AUROC, AUPRC, Brier score, and calibration slope for incident cardiovascular endpoints over 1-year and 5-year horizons (matched to MESA's published follow-up windows), with person-cluster bootstrap 95% confidence intervals.

2. **Subgroup performance.** Pre-registered subgroup analyses by sex, age band (<55 / 55–65 / >65), and race/ethnicity (per MESA's four self-reported groups: White, Black, Hispanic, Chinese-American), with explicit reporting of cell sizes and minimum-event-count thresholds.

3. **Benchmarking against existing MESA-derived clinical risk scores.** Side-by-side reporting against the MESA CHD Risk Score (Polonsky et al., JAMA 2010) and mC2HEST (Chen et al., JACC Advances 2024) for the appropriate endpoint subsets.

**Specific data requested.**

- MESA cohort core variables (demographics, race/ethnicity, BMI, smoking, lipids, blood pressure, diabetes, prior CVD)
- MESA medication file (Exams 1–5)
- MESA laboratory file (chemistry panels, lipids, HbA1c)
- MESA adjudicated CV event files (MI, definite/probable angina with revascularization, stroke, CHD death, other CVD death, all-cause mortality)
- MESA atrial fibrillation events file (ascertained via ICD codes, study ECGs, and Medicare claims linkage) — request via ancillary if not in standard release
- MESA heart failure events file — request via ancillary if not in standard release
- Linkage variables to enable join with the MESA Sleep substudy (requested separately via NSRR at sleepdata.org)

**Analytic plan and data security.** Analyses will be conducted on Weill Cornell Medicine's HIPAA-aligned compute infrastructure under institutional data use agreement. No re-identification will be attempted. All analysis code will be version-controlled and made publicly available upon publication; participant-level data will not leave the secure environment.

## Anticipated publications

One peer-reviewed publication (target: *Nature Medicine* or *npj Digital Medicine*) reporting external validation of the foundation model on the MESA cohort, with the MESA-Sleep-anchored wearable analysis as a paired companion or supplementary analysis.

## Prior BioLINCC data use

[TODO — list any prior BioLINCC projects led by PI Elemento]

## IRB and DUA

The Weill Cornell Medicine institutional review board has determined that external validation of a pre-trained model on a public, de-identified cohort qualifies as non-human-subjects research; an exemption determination letter will be submitted with the executed data use agreement.

## Funding and conflicts of interest

[TODO — paste from `/Users/ole2001/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Administration/Funding and COI statements.md`]
