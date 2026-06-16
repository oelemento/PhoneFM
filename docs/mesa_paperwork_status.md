# MESA external-validation paperwork — status

Tracks every paperwork track in motion for the MESA external-validation arc of PhoneFM. Sibling to `docs/aou_egress_submission_2026-06-15.md` (which covers the AoU side).

---

## Summary table (as of 2026-06-16)

| Track | ID / contact | State | Blocked on |
|---|---|---|---|
| NSRR MESA Sleep data request | #27710 | Submitted 2026-06-14 | NSRR review (2-week SLA) |
| BioLINCC MESA data request | #17837 | "Pending Requestor Documentation" | IRB letter + CV from us |
| MESA Coordinating Center | chsccweb@u.washington.edu | Drafted email; status of actual send unclear | — |
| WCM IRB NHSR determination | irb@med.cornell.edu | Email sent 2026-06-16 | IRB Navigation Team response |
| Olivier's CV for BioLINCC | local PDF | Identified file | User to attach when BioLINCC reopens |

---

## 1. NSRR MESA Sleep (request #27710) — SUBMITTED

- URL: https://sleepdata.org/data/requests/27710
- Submitted: 2026-06-14
- Type: DAUA Standard (Individual) v2.0.0, e-signed
- Dataset selected: MESA only
- Posting permission: name + summary public
- Expected turnaround: 2 weeks (per NSRR confirmation page)
- Executed agreement: NOT on disk locally; viewable via `https://sleepdata.org/data/requests/27710/print` (NSRR's "PDF View Currently Unavailable" — use browser Print → Save as PDF if needed)

## 2. BioLINCC MESA (request #17837) — PENDING REQUESTOR DOCUMENTATION

- URL: https://biolincc.nhlbi.nih.gov/my/submitted/request/
- Submitted: 2026-06-14
- BioLINCC reviewer: Megan Savage (posted 2026-06-15 at 8:20 AM)
- Status quote: *"In order to proceed with your request, NHLBI requires the following documentation: Documentation of Institutional Review Board (IRB) approval or exemption. Your Curriculum Vitae as the data access requester. Please provide the documentation in PDF format."*

### What we owe BioLINCC

- **IRB documentation** — see §4 below (NHSR determination requested 2026-06-16)
- **CV** — `/Users/ole2001/Desktop/ADMIN/Applications/CPRIT 2025/vjc 11.22.24 CV OElemento - Faculty (Oct. 2024)CD[16].pdf` is the most recent on disk (Oct 2024 content, modified 2025-03-17). Suggested rename when attaching: `Elemento_CV_2024-10.pdf`. Alternative source: ask Victoria for the freshest version on Box.

## 3. MESA Coordinating Center inquiry — DRAFTED

- To: chsccweb@u.washington.edu
- Subject: External validation of a wearable cardiovascular foundation model on MESA — ancillary event-file availability
- Three questions: AFib ancillary event-file availability path; HF ancillary event-file availability path; latest CV event-file follow-up cutoff (beyond CY2020?)
- Status: drafted at `docs/mesa_cc_inquiry_email.md`; opened in Apple Mail compose window on 2026-06-14; user's actual Send not confirmed in chat. Re-check Outlook Sent folder; resend if needed.

## 4. WCM IRB NHSR determination request — SENT 2026-06-16

- To: irb@med.cornell.edu
- Subject: NHSR determination question - external validation of pre-trained model on de-identified MESA data
- Attached: 2-page `/tmp/PhoneFM_IRB_Project_Description.pdf` (in SAP template format)
- Email body and PDF mirror the prior successful pattern (the Speech Accessibility Project NHSR request)
- Key framing choices:
  - Style mirrors prior SAP NHSR consultation email — "Hello," + 4 short paragraphs
  - PDF mirrors `SAP_NHSR_description.pdf` template structure (Project title / PI / Purpose / Data source / Basis for NHSR / Procedures / Data security / Funding)
  - Drops "DUA" / "Data Use Agreement" language throughout — at WCM only OSRA can sign institutional DUAs; what's in flight at NSRR is a personal "Standard (Individual)" form and BioLINCC's RMDA hasn't been issued yet
  - Includes the model-weights-aren't-human-data argument with MIA AUROC 0.5048 (CI [0.4827, 0.5261]) inline
  - Names the BioLINCC #17837 operational dependency to nudge a fast turnaround
- Expected turnaround: typically 1-2 weeks at WCM IRB Navigation Team for an NHSR determination

### Backup plan if WCM IRB pushes back on NHSR

If the IRB Navigation Team declines to issue NHSR and asks us to file a Secondary Analysis IRA instead, the next step is the longer WRG-HS workflow (see vault note `Knowledge Base/WCM IRB Submission (WRG-HS).md`). Critical reminder from the EHR FM precedent (Protocol #26-06030378, bounced 2026-06-15 for missing PRMC approval): **General PRMC review is required for secondary analysis at WCM**, not just for clinical trials. The Secondary Analysis IRA goes through OnCore/ePRMS PRMC before the IRB will actually review.

## 5. Critical-path

The MESA external validation cannot proceed until:

1. **IRB determination letter** (WCM IRB → us → BioLINCC)
2. **CV attached** to BioLINCC #17837
3. **BioLINCC RMDA issued** (BioLINCC, after #1+#2 satisfied)
4. **WCM OSRA countersigns the BioLINCC RMDA** (institutional signature; only WCM can sign on WCM's behalf)
5. **Data released by BioLINCC** to a WCM destination
6. **NSRR data release** (independent of BioLINCC; #27710 in 2-week review)
7. **AoU egress approved** (parallel; tracked in `docs/aou_egress_submission_2026-06-15.md`)
8. **Model weights egressed** to WCM destination
9. **External validation analyses run**

Estimated critical-path: 4–6 weeks if every step proceeds on its normal SLA.

---

## Reproduction notes

The full set of paperwork drafts is in this repo:

- `docs/nsrr_mesa_data_request.md` — NSRR application content
- `docs/biolincc_mesa_application.md` — BioLINCC application content
- `docs/mesa_cc_inquiry_email.md` — MESA CC inquiry email draft
- `docs/wcm_irb_nhsr_request.md` — original WCM IRB request draft (the actual sent version was tighter; PDF at `/tmp/PhoneFM_IRB_Project_Description.pdf`)
- `docs/aou_egress_submission_2026-06-15.md` — AoU egress submission log
- `docs/mia_audit_results_2026-06-15.md` — MIA audit (egress evidence)
