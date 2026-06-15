# All of Us egress submission — verbatim log

**Date:** 2026-06-15
**Workspace:** PhoneFM CT cohort (Copy)
**Workspace UUID:** `d56338c6-28f2-4461-9d4d-7e2c3aad6d3f`
**GCP project:** `wb-shrewd-lime-9770`
**Workbench URL:** https://workbench.verily.com/workspaces/phonefm-ct-cohort-copy
**Artifact:** PyTorch checkpoint of PhoneFM v3 (13.3M-parameter transformer) trained on AoU Fitbit + EHR data.

This document captures the EXACT content submitted (or pre-filled and pending submission) for every AoU touch-point in the model-egress workflow, so the request can be reconstructed from scratch if AoU bounces it or asks for a resubmission.

---

## 0. Pre-egress evidence pushed to repo (commit `91dbc5e` on `v3-spec`)

- `docs/mia_audit_results_2026-06-15.md` — full MIA audit log (methodology, caveats, structural priors, ready-to-paste paragraph). Headline: **person-level MIA AUROC 0.5048, 95% CI [0.4827, 0.5261]**, multihead corroboration 0.5054 [0.4827, 0.5283]. CI upper bound below the 0.55 pre-registered threshold; interval covers chance. No detectable memorization.
- `workbench/aou_export_request.md` — v3-current egress request template (mirrors the Zendesk body below).
- `workbench/mia_audit.py` — audit code @ commit `0158756` (v2.1; bf16-fallback for T4).
- `workbench/prep_model_export.sh` — staging script.

---

## 1. Staged bundle (verified on the pod via `bash workbench/prep_model_export.sh`)

Computed 2026-06-15 ~16:49 UTC. Files staged to `/home/jupyter/export/phonefm_v3/` on the pod (inside CT; will egress to a destination TBD with AoU support):

| File | Size (bytes) | sha256 |
|---|---:|---|
| `best.pt` | 53,063,971 | `ac0af3bf2a1360a9d50308881d2408cc5c6c5c2004b96edc4d6e9728834728be` |
| `config.json` | 4,975 | `a16e1b0d981728446c3a4d56c30c690593399a92146d66601af6ae5957833d35` |

Parameter manifest:
- 120 tensor entries; 13,257,311 parameters (13.26M)
- dtypes: `{torch.float32: 118, torch.int64: 2}`
- non-tensor keys: none
- Verdict: parameter tensors only — no optimizer state, no RNG, no data

---

## 2. Heads-up email to AoU support (SENT 2026-06-15 ~12:37 EDT)

**To:** support@researchallofus.org
**From:** ole2001@med.cornell.edu (Exchange)
**Subject:** Heads-up: AI/ML model-weight egress request — PhoneFM CT cohort

**Body (verbatim):**

```
Dear All of Us support team,

I am preparing to file a Large Download Exemption request for a model-weight egress (not summary statistics).

Workspace: PhoneFM CT cohort (Copy), UUID d56338c6-28f2-4461-9d4d-7e2c3aad6d3f, GCP project wb-shrewd-lime-9770.

Artifact: a 53MB PyTorch checkpoint (best.pt + config.json), a 13.3M-parameter transformer trained on Fitbit + EHR data from ~12,500 AoU participants. Intended use: external validation on the MESA Sleep cohort (NSRR request #27710 and BioLINCC request #17837 submitted 2026-06-14).

Looking further ahead, I would also like to explore whether the model could be deployed on a phone to provide health insights, in the context of an IRB-approved research study.

A person-level membership-inference audit on the checkpoint returned AUROC = 0.5048 (95% CI [0.4827, 0.5261]), comfortably below the 0.55 pre-registered threshold and statistically indistinguishable from chance. Full audit report in the project repo (oelemento/PhoneFM) at docs/mia_audit_results_2026-06-15.md.

Three quick questions before I file:

1. Is there anything beyond the Large Download Exemption form + Zendesk ticket I should prepare for an AI/ML model-weight request?

2. Should I provision a Weill Cornell Medicine GCS bucket in advance for the approved release, or does AoU provide a release endpoint?

3. Is the standard 5-7 business day turnaround applicable for model-weight requests, or does the AI/ML pathway have a different timeline?

I'm happy to share more detail on request.

Best regards,
Olivier Elemento, PhD
Professor of Physiology and Biophysics
Director, Englander Institute for Precision Medicine
Weill Cornell Medicine
1305 York Avenue, Box 140, New York, NY 10021
ole2001@med.cornell.edu
```

---

## 3. REDCap "Large Download Exemption Intake Form" (SUBMITTED 2026-06-15)

**Form URL:** https://redcap.pmi-ops.org/surveys/?s=YRXMJFJ97J3WMWLE
(After submission the REDCap survey reissues a unique-session URL; the version that was actually submitted is what counts.)

**Answers (verbatim):**

| Question | Answer |
|---|---|
| First Name | Olivier |
| Last Name | Elemento |
| Institutional Email Address | ole2001@med.cornell.edu |
| Researcher Workbench User Name | *(user-provided, e.g. `oelemento@researchallofus.org` — fill before resubmit)* |
| Will the file be downloaded using the Legacy Researcher Workbench or Researcher Workbench 2.0? | Researcher Workbench 2.0 |
| Workspace UUID (Workbench 2.0 only) | `d56338c6-28f2-4461-9d4d-7e2c3aad6d3f` |
| Workspace URL | https://workbench.verily.com/workspaces/phonefm-ct-cohort-copy |
| Workspace title | PhoneFM CT cohort (Copy) |
| Are you completing this download request form after an egress event? | No |
| Is this request related to working on a large data extraction or query? | No |
| Is your request related to AI/ML or a specific command, package, or tool that uses an API call? | **Yes** |
| Please confirm you have reviewed and will comply with the AI/ML policies | Yes |
| AI/ML description (command/package/tool, inbound/outbound files) | *PyTorch checkpoint (best.pt, ~53 MB) for a 13.3-million-parameter transformer foundation model trained on Fitbit + EHR data from ~12,500 AoU participants, plus companion config.json (~5 KB) with the model configuration. The 'tool' is the standard PyTorch model.load_state_dict() API; no external API calls or services are invoked. Outbound files: best.pt + config.json (parameter-only, no participant-level data, verified by workbench/prep_model_export.sh manifest). No inbound files. Egress is a one-time transfer to a Weill Cornell Medicine GCS bucket for external validation on the MESA Sleep cohort.* |
| Is your request specific to a genomic file download or use? | No |
| Is your download request related to the All by All dataset? | **No** *(corrected from initial Yes — All by All is a specific AoU GWAS-style summary dataset we do NOT use)* |
| What file type are you attempting to download? | `.pt (PyTorch state_dict checkpoint) and .json (model configuration)` |
| Anticipated file size | `53 MB total (best.pt = 53,063,971 bytes; config.json = 4,975 bytes)` |
| Is person_id and/or sample_id included as a field in file download? | No |
| Detailed description of data + use | *PyTorch checkpoint best.pt (53,063,971 B; sha256 ac0af3bf...28be) plus config.json (4,975 B; sha256 a16e1b0d...3d35) for a 13.3-million-parameter transformer foundation model trained on Fitbit + EHR data from the AoU Controlled Tier CDR. Training cohort: ~12,500 participants with paired Fitbit and EHR data; 920K training windows of 180 days each. Architecture: 12-layer transformer with rotary positional embedding on calendar-day index; 13 prediction heads (10 active during training: cardiovascular composite at 30/180/365 days, T2D at 180/365 days, depression at 180/365 days, 3 pre-registered negative controls; 3 mortality heads dropped due to n_pos<50). Parameter manifest verified by workbench/prep_model_export.sh: 120 tensor entries, 13,257,311 parameters, dtypes {torch.float32: 118, torch.int64: 2}, no optimizer state, no RNG, no data. Use: load the weights on Weill Cornell Medicine HIPAA-aligned compute, run inference on the MESA Sleep cohort participants (NSRR + BioLINCC requests #27710 and #17837 submitted 2026-06-14), report aggregate discrimination (AUROC, AUPRC) and calibration (Brier, calibration slope) at 1-year and 5-year horizons with person-cluster bootstrap 95% CIs. Pre-registered subgroup analyses by sex, age band, and self-reported race/ethnicity. Pre-egress membership-inference audit on the checkpoint returned AUROC 0.5048, 95% CI [0.4827, 0.5261], comfortably below the 0.55 pre-registered threshold (full report at docs/mia_audit_results_2026-06-15.md in the oelemento/PhoneFM repository, commit 91dbc5e).* |
| Justification (why download is impactful, why can't be completed without egress) | *External validation of the foundation model on the independent MESA Sleep cohort (Multi-Ethnic Study of Atherosclerosis; NSRR request #27710 and BioLINCC application #17837 submitted 2026-06-14) requires the trained model weights to leave the All of Us Controlled Tier. The MESA Sleep cohort cannot be brought into AoU CT because it is governed by separate Data Use Agreements (NHLBI BioLINCC and NSRR) that prohibit re-hosting; the only path to external validation is therefore egress of the trained model. External validation is a standard requirement for any foundation-model publication (Nature Medicine, npj Digital Medicine, Lancet Digital Health reviewer expectations) and is the primary preliminary-data deliverable for a planned NIH R01 application. The egress is one-time and parameter-only (no participant data, no aggregate row-level outputs); the bundle has been staged and verified by workbench/prep_model_export.sh (sha256 hashes above). A person-level membership-inference audit on the checkpoint shows no detectable memorization of training participants (AUROC 0.5048, 95% CI [0.4827, 0.5261], CI upper bound below the 0.55 threshold). Without the egress relaxation, the project cannot proceed beyond the in-CT training stage and the planned external validation cannot be completed.* |
| Preferred start date for the 2-day relaxation window | *(blank — accept default Mon-Tue post-approval)* |
| Are you interested in having the VUMC IDASC team assist with this download? | Yes |
| Confirm no row-level data is being downloaded | Yes, I confirm no row level data is being downloaded |
| Reviewed and will comply with the Data and Statistics Dissemination Policy | Yes |
| Confirm contents/description are complete and accurate; will only download listed files | Yes |

---

## 4. Zendesk support ticket (PRE-FILLED, awaiting user sign-in + Submit)

**Portal:** https://support.researchallofus.org/hc/en-us/requests/new?ticket_form_id=16530933079444
**Issue category:** Request Large Download
**Email:** ole2001@med.cornell.edu
**REDCap-completion checkbox:** ☑ (legitimately, after REDCap submission above)
**Subject:** `Controlled-Tier AI/ML model-weight egress — PhoneFM (13.3M-parameter checkpoint)`

**Body (verbatim):**

```
Hello,

I am filing a Controlled-Tier model-weight egress request via the Large Download Exemption pathway, AI/ML route. A heads-up email was sent to support@researchallofus.org earlier today; this Zendesk ticket is the formal request.

WORKSPACE
- Name: PhoneFM CT cohort (Copy)
- UUID: d56338c6-28f2-4461-9d4d-7e2c3aad6d3f
- GCP project: wb-shrewd-lime-9770

ARTIFACTS TO EXPORT (aggregate; no participant-level data)
- best.pt: 53,063,971 B, sha256 ac0af3bf2a1360a9d50308881d2408cc5c6c5c2004b96edc4d6e9728834728be
- config.json: 4,975 B, sha256 a16e1b0d981728446c3a4d56c30c690593399a92146d66601af6ae5957833d35

PARAMETER MANIFEST (verified by workbench/prep_model_export.sh on the pod)
- 120 tensor entries; 13,257,311 parameters (13.26M)
- dtypes: torch.float32 x 118, torch.int64 x 2
- non-tensor keys: none
- Verdict: parameter tensors only, no optimizer state, no RNG, no data

INTENDED USE
Primary: external validation on the MESA Sleep cohort (NSRR request #27710 and BioLINCC request #17837 submitted 2026-06-14). Longer term, I would also like to explore whether the model could be deployed on a phone to provide health insights, in the context of an IRB-approved research study.

MEMBERSHIP-INFERENCE AUDIT (pre-egress evidence)
Person-level loss-threshold attack with person-cluster bootstrap (n=1300 train vs n=1300 held-out test participants, K=8 windows per person, n_boot=2000):
- Primary metric (cv_composite_30d single-head BCE): AUROC 0.5048, 95% CI [0.4827, 0.5261]. CI upper bound below the 0.55 pre-registered threshold; interval covers chance.
- Multi-head BCE corroboration: AUROC 0.5054, 95% CI [0.4827, 0.5283].

Combined with the 13.3M parameters / 6,496 train participants ratio and the absence of any retrieval or memory-bank component, the checkpoint shows no detectable memorization of training participants.

Full audit report: docs/mia_audit_results_2026-06-15.md in the oelemento/PhoneFM repository (commit 91dbc5e).
Full request rationale: workbench/aou_export_request.md in the same repository.

DESTINATION
TBD pending guidance from the AoU support team (asked in the heads-up email). I am happy to provision a Weill Cornell Medicine GCS bucket if AoU does not provide a release endpoint.

Thanks,
Olivier Elemento, PhD
Professor of Physiology and Biophysics
Director, Englander Institute for Precision Medicine
Weill Cornell Medicine
ole2001@med.cornell.edu
```

**Attachments:** none yet (could optionally attach `mia_audit.json` and `SHA256SUMS.txt` from the pod, but those would need their own egress; deferred — reviewer can access via the repo + the linked docs).

---

## 5. Outstanding actions

1. Olivier signs into Zendesk (top-right "Sign-in using @researchallofus.org account") and clicks Submit on the support ticket.
2. AoU support reviews (5–7 business days standard; AI/ML pathway timeline unconfirmed — asked in heads-up email).
3. On approval: provision destination bucket (probably `gs://wcm-eipm-phonefm-export/` or per AoU guidance), `gsutil cp ~/export/phonefm_v3/*` from pod to destination during the 2-day Mon-Tue relaxation window.

---

## 6. To reproduce / resubmit

1. **Stage bundle (~30 sec on pod):** `cd ~/repos/PhoneFM && bash workbench/prep_model_export.sh`
   - Re-compute sha256 — must match section 1 above; if they don't, the checkpoint changed and the audit/request docs may need to be re-run.
2. **REDCap form:** open https://redcap.pmi-ops.org/surveys/?s=YRXMJFJ97J3WMWLE (form ID `YRXMJFJ97J3WMWLE`); paste in the answers from section 3 above.
3. **Zendesk ticket:** open https://support.researchallofus.org/hc/en-us/requests/new?ticket_form_id=16530933079444; pick "Request Large Download" issue; paste subject + body from section 4.
4. **Heads-up email (optional):** if support has already responded to the prior heads-up, skip; otherwise re-send the body in section 2.

Related files in repo (commit 91dbc5e or later):
- `docs/mia_audit_results_2026-06-15.md`
- `workbench/aou_export_request.md`
- `workbench/mia_audit.py`
- `workbench/prep_model_export.sh`
