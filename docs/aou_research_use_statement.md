# Research Use Statement — PhoneFM duplicate workspace

Answers for the All of Us workspace duplication form. Public-facing: these appear in the Research Hub Directory, so they're written for a non-technical audience while staying scientifically accurate.

---

## Scientific question

Can a deep-learning foundation model trained jointly on continuous wearable data (Fitbit heart rate, steps, sleep) and electronic health records predict major cardiovascular events — atrial fibrillation, myocardial infarction, and heart failure — within 30 to 365 days, and can such a model be deployed on a consumer smartphone so that risk monitoring becomes accessible without a clinic visit?

---

## Scientific approach

We are developing PhoneFM, a transformer-based foundation model that:

1. **Pretrains** in a self-supervised way on 90-day daily-wearable subwindows using masked daily-vector reconstruction (a BERT-style objective adapted to continuous physiological features), with no labels required, on approximately 9,000 All of Us participants who have at least 545 days of Fitbit data.
2. **Fine-tunes** the pretrained backbone on 13 endpoints across three time horizons (30, 180, 365 days): a cardiovascular composite (AFib + MI + HF), all-cause mortality, type-2 diabetes first-recorded, depression first-recorded, and three pre-registered negative-control conditions (skin malignant neoplasm, refractive errors, dental caries) used to test whether the model captures real wearable physiology rather than healthcare-utilization signals.
3. **Evaluates** with person-level cluster bootstrap 95% confidence intervals, comparing against established clinical risk scores (CHA₂DS₂-VASc, ASCVD, FINDRISC), and runs pre-registered subgroup analyses across race/ethnicity, sex, age, and socioeconomic status to surface any performance disparities.
4. **Deploys** the trained model to consumer mobile devices via Apple Core ML for on-device inference, allowing risk estimates to be produced without sending personal health data to any server.

---

## Anticipated findings

- Quantitative performance estimates (AUROC, AUPRC, calibration) for each (endpoint, horizon) head, with cluster-bootstrap confidence intervals.
- A pretrained transformer backbone that can be reused for future wearable-based health endpoints without re-running self-supervised pretraining.
- Evidence about whether wearable + EHR data, used together, improve on either source alone for short- and medium-term cardiovascular risk.
- Quantification of how negative-control endpoint AUROCs deviate from 0.5, separating genuine wearable signal from healthcare-utilization confounding.
- Subgroup-stratified performance metrics that document fairness gaps in model accuracy across race, age, sex, and socioeconomic strata.
- Latency, memory, and battery measurements for on-device deployment on commodity smartphone hardware.

---

## How findings benefit the community

- **Accessible risk monitoring.** A model that runs on a smartphone people already own removes the cost and access barriers of clinic-based cardiovascular risk assessment, with particular benefit to rural, underserved, and resource-limited populations that All of Us was designed to engage.
- **On-device privacy preservation.** All inference runs locally; sensitive wearable and health data never leaves the participant's phone, addressing community concerns about cloud-based health AI.
- **Open scientific resource.** The trained backbone, training code, and evaluation pipeline will be released openly (subject to required export-review), so other researchers and clinical teams can fine-tune for their own endpoints without rebuilding the pretraining infrastructure.
- **Transparent fairness reporting.** Pre-registered subgroup analyses are published as part of the primary results — not optional supplementary material — so any disparities are visible and can drive subsequent equity-focused modeling work.
- **Pathway to clinical translation.** Findings inform NIH funding directions and ultimately a SaMD (Software as a Medical Device) pre-submission to FDA for cardiovascular early-warning.

---

## Demographic differences in disease

Yes — this study explicitly tests for differential model performance across demographic strata. We have pre-registered (before any test-set access) subgroup analyses across:

- **Race / ethnicity:** Black or African American, Hispanic or Latino, Asian, White, Other
- **Age at end of input window:** under 55, 55-65, over 65
- **Sex at birth:** Female, Male
- **Socioeconomic status:** income quartile when reported
- **One pairwise interaction:** race × age stratum

Per-subgroup metrics include AUROC, AUPRC, Brier calibration score, calibration intercept, and calibration slope — each reported with person-level cluster-bootstrap 95% confidence intervals. Cells with fewer than 30 participants are reported as insufficient sample rather than producing low-confidence estimates. All thresholds and bin edges are fixed in `subgroup_definitions.json` (SHA-1 logged at evaluation time) before any test-set access, so the analyses cannot be retroactively tuned.

The goal is to surface, not hide, any model performance gaps across the diverse populations that the All of Us program is uniquely positioned to study — and to ensure that consumer-device cardiovascular risk monitoring, when it reaches clinical deployment, does not amplify existing health disparities.

---

## Other relevant information (optional)

This workspace is a duplicate of an existing All of Us workspace (`phonefm-ct-cohort`) under the same researcher and the same approved data access tier. The duplication is being created because the original workspace's initial-credit period has expired; the new workspace will run on self-funded Weill Cornell Medicine billing so that the project can continue without interruption. No new data access or scope expansion is requested beyond what was approved for the original workspace.

---

## Tips before submitting

1. **Trim each answer to ~150-250 words** if the form has a character limit. The drafts above are slightly over.
2. **"Anticipated findings" should be the longest answer** — reviewers read this most closely.
3. **The "How findings benefit the community" answer is the most-read field by participants** browsing the Research Hub Directory. Lead with accessibility and privacy preservation — those are themes the All of Us community responds to.
4. **Don't add "iPhone" or "Apple" branding** unnecessarily; "consumer smartphone" and "on-device inference" read as neutral and avoid implying we're a commercial collaboration with Apple.
5. **The "Other" box** is a good place to explain the workspace-duplication context so the AoU reviewers immediately understand why a duplicate exists.
