# PhoneFM — phone-resident cardio FM, 4-week prototype

**Venture:** LLM-in-a-Box (Olivier × Shahram). PhoneFM is the patient-facing arm.

**Goal of weeks 1-4:** TestFlight iPhone app that reads HealthKit (incl. Apple Health Records FHIR), runs an on-device cardio risk model trained on All of Us Fitbit + EHR cohort, surfaces weekly cardio-risk delta with plain-language explanation.

**Out of scope for v1:** FDA-cleared diagnosis, medication titration, alerting. Stay in "risk surface + topic-for-your-doctor" territory.

## Timeline

| Week | Workstream A (Workbench, ML) | Workstream B (iOS) |
|---|---|---|
| 1 | Cohort extraction, endpoint definition, tokenizer spec | Xcode project, HealthKit entitlement, permission flow |
| 2 | Train 50–100M base model. **Submit model-export request Day 1 of Week 2.** | HealthKit data assembly, Health Records consent, sample data UI |
| 3 | Iterate on training; finalize export package | Core ML conversion, on-device inference, weekly risk gauge |
| 4 | Hand off final model | Apple Foundation Models explanation layer, TestFlight beta to 5–10 users |

**Decision gate at end of Week 4:** is output clinically meaningful? Path forward = federated continuous learning + WCM opt-in expansion + SaMD pre-sub.

## Cohort definition (locked at start of Week 1)

- All of Us participants with linked Fitbit data ≥ 180 days
- Linked EHR (condition_occurrence has ≥ 1 record)
- Age ≥ 40 at enrollment (focus cardio risk window)
- Held-out event window: 30 days after last training-data timestamp

**Endpoints (composite):**
- AFib detection (ICD-10 I48.x in encounter, or device-detected AFib in `observation`)
- MI hospitalization (I21.x as primary or secondary in inpatient)
- HF decompensation (I50.x as primary with hospital admission)
- Cardiovascular mortality (death table linked, cause = cardiovascular)

## Tokenizer design (v1)

**Wearable stream (continuous):**
- 5-min binned heart rate → eCDF decile token: `HR_D0..D9`
- Daily steps → decile: `STEPS_D0..D9`
- Sleep stages per night: `SLEEP_REM_<pct>`, `SLEEP_DEEP_<pct>`
- HRV daily summary: `HRV_D0..D9`
- ECG events (if available): `ECG_AFIB`, `ECG_NORMAL`

**EHR stream (episodic):**
- Conditions: `DX10:<3char>` (consistent with the existing GPT-EHR tokenization)
- Medications: `MED:<class>` (cardiac classes mostly: beta-blockers, ACE-I, ARB, anticoagulants, diuretics)
- Procedures: `PX10:<3char>`
- Labs: `LAB:<itemid>_D<0..9>` for top cardiac labs (troponin, BNP, creatinine, K, Na)

**Sequence assembly:** chronological merge of wearable + EHR tokens, with `<DAY_SEP>` between calendar days and `<ENC_SEP>` between EHR encounters.

## Model export from All of Us — START EARLY

All of Us policy permits trained model weights to leave the Workbench but each request is individually reviewed. Submit the request **Day 1 of Week 2**, not at end.

Request template at `aou_export_request.md`.

## Team

- 1 ML/data engineer (Workbench cohort + training): ~30 hr/wk
- 1 iOS engineer (SwiftUI + HealthKit + Core ML): ~25 hr/wk
- Olivier + Shahram (clinical, product, sign-offs): ~5 hr/wk each

## Critical-path dependencies

1. All of Us Controlled Tier ✅ (Olivier already credentialed)
2. Workbench compute budget — confirm ~$500 cap for 4 weeks
3. All of Us model-export approval — **submit Week 2 Day 1**
4. Apple Developer account ($99/yr) for TestFlight
5. iPhone running iOS 18+ for Apple Foundation Models framework

## What we'll know at Week 4

- Whether All of Us Fitbit + EHR is rich enough for AUROC > 0.75 on 30-day cardio events
- Whether on-device inference pipeline runs end-to-end
- Whether the UX feels useful (not just another wellness app)
- Concrete sense of SaMD-clearance pathway for v2

## After Week 4

- **SaMD pre-submission** to FDA (12–18 mo for de novo clearance — start now)
- **Federated continuous learning** so each user's phone fine-tunes locally
- **Cohort expansion** to UK Biobank + WCM opt-in
- **Multi-condition** beyond cardio: diabetes, mental health, CKD
- **Hospital ↔ phone handshake** — patient phone-FM queries hospital LLM-in-a-Box FM with consent
