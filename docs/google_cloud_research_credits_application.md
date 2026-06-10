# Google Cloud Research Credits — PhoneFM Application Draft

Form: https://edu.google.com/intl/ALL_us/programs/credits/research/

---

## Personal information

| Field | Value |
|-------|-------|
| First name | Olivier |
| Last name | Elemento |
| Email | ole2001@med.cornell.edu |
| Country | United States |
| Job Title | Director, Englander Institute for Precision Medicine; Professor of Physiology and Biophysics |
| Organization Name | Weill Cornell Medicine |
| Type of institution | University |
| Department name | Department of Systems and Computational Biomedicine / Englander Institute for Precision Medicine |
| Faculty directory link | https://elementolab.weill.cornell.edu/ |
| How did you hear about the program? | Colleague referral *(or whichever applies)* |

---

## Project information

**Project name:** PhoneFM — On-Device Cardiovascular Foundation Model from Wearable + EHR Data

**Project start date:** 2026-06-23 (or whatever works)

### Proposal (~245 words; 250 word maximum)

Cardiovascular disease (CVD) is the leading cause of death globally, yet existing risk tools (ASCVD, CHA₂DS₂-VASc) require clinic visits and a small set of static inputs. Smartphones already collect continuous Fitbit/Apple Health signals from hundreds of millions of users, but no validated, on-device foundation model translates these into actionable CVD risk.

PhoneFM is a 13M-parameter transformer pretrained via self-supervised masked daily-vector reconstruction on 920K wearable+EHR windows from the All of Us cohort (12,500 participants). Our v3 model achieves AUROC=0.88 on 30-day cardiovascular composite (AFib+MI+HF) prediction, with three pre-registered negative-control endpoints (skin neoplasm, refractive errors, dental caries) tracking to ≈0.5 — confirming the model captures real wearable physiology rather than healthcare-utilization confounders.

**GCP tools:** Vertex AI Training (A100/H100) for backbone scale-up to 50M parameters; BigQuery for cohort extraction across All of Us and UK Biobank; Cloud Storage for tokenized data and model checkpoints; Vertex AI Endpoints for inference latency benchmarking and federated-learning prototypes; Cloud Healthcare API for FHIR-mediated EHR integration.

**12-month milestones:**
- M1-3: Scale to 50M-parameter backbone; integrate published polygenic risk scores
- M4-6: UK Biobank external validation; iPhone Core ML deployment via TestFlight
- M7-9: Federated continuous-learning prototype across institutional partners
- M10-12: Manuscript (Nature Medicine target); NIH R01 submission

**Future support:** Sustained Vertex AI infrastructure for federated training across our institutional collaborators (BCCA, Sage Bionetworks) and a HIPAA-aligned model registry for clinical deployment.

### Field of research

Health and Life Sciences (medical AI / digital health)

---

## Additional information

### Google Scholar link

https://scholar.google.com/citations?user=AYa6tNcAAAAJ

### How do you intend to apply awarded cloud credits?

**Train machine learning models** (Vertex AI Training) and **process/analyze large datasets** (BigQuery). Specifically: scale PhoneFM backbone training, run cohort extraction across multiple biobanks, and prototype federated continuous-learning.

### Does your research project have an external funding source?

**Yes.** Project is supported by Weill Cornell Medicine / Englander Institute for Precision Medicine internal funds, and PI Olivier Elemento holds active NIH grants. An NIH R01 application leveraging PhoneFM as preliminary data is planned within the credit period.

### Have you used cloud infrastructure for your research before?

**Yes.** The Elemento Lab routinely uses cloud HPC (Cornell SCU, Cayuga) and the All of Us Research Workbench (Verily / GCP) for genomic and wearable analyses.

### Have you used Google Cloud before?

**Yes.** All of Us Research Workbench (Verily Workbench) runs on GCP — current PhoneFM v3 pretraining and supervised fine-tuning were executed there. Initial AoU credits are about to expire, motivating this application.

### Expected costs

Submit estimate via [GCP Pricing Calculator](https://cloud.google.com/products/calculator).

**Suggested itemization for the calculator:**

| Resource | Quantity | Unit price | Annual |
|---|---|---|---|
| A100 40GB (a2-highgpu-1g) | 1,500 GPU-hours | $3.70/hr | $5,550 |
| n1-highmem-2 (CPU pre-processing) | 1,000 CPU-hours | $0.15/hr | $150 |
| GCS Standard storage | 500 GB × 12 months | $0.020/GB/mo | $120 |
| BigQuery on-demand queries | ~5 TB processed | $5/TB | $25 |
| Vertex AI Endpoints (benchmarking) | 200 hours | $0.30/hr | $60 |
| Cloud Healthcare API | light usage | est. | $200 |
| **Total annual estimate** | | | **~$6,100** |

**Recommended ask:** **$10,000** (gives ~50% headroom for unanticipated scale-ups: 50M-parameter backbone, federated multi-institution training, longer training sweeps for ablations).

### Google Cloud Billing Account ID

*(Create a new Cloud Billing Account at https://console.cloud.google.com/billing if needed — separate from the AoU-provided one which is read-only.)*

Format: XXXXXX-XXXXXX-XXXXXX

### After your credit expires, how do you intend to continue funding your project?

Project funding will transition to (a) active NIH R01 grants held by PI Elemento, (b) Englander Institute for Precision Medicine internal infrastructure budget, and (c) Weill Cornell Medicine departmental research support. PhoneFM is also a component of a broader translational initiative ("LLM-in-a-Box") co-developed with industry partners; revenue from successful TestFlight deployment and any subsequent clinical-decision-support licensing would sustain ongoing cloud infrastructure costs.

---

## Notes / suggested polish before submitting

1. **Faculty directory link** — confirm the lab page renders as a researcher profile (some Google reviewers prefer the institutional bio page over a lab page).
2. **NIH grant numbers** — adding 1-2 specific grant numbers under "external funding" makes the application much stronger; cite the active R01s.
3. **Past Google Cloud usage** — mention if you've previously been awarded any Google credits or have published GCP-acknowledged papers; reviewers reward demonstrated efficient credit use.
4. **Manuscript target** — replace "Nature Medicine" with whichever venue you're actually targeting; reviewers can tell if it's aspirational vs realistic.
5. **Collaborators** — listing named co-investigators (BCCA, Sage Bionetworks, Shahram's group) lends weight; one-line bios optional.
6. **Pricing calculator URL** — must be a real saved URL from cloud.google.com/products/calculator. Build it once with the items in the table above and paste the resulting share link.
7. **iPhone TestFlight deployment** — calling this out as a deliverable is unusual for academic credit applications and is a differentiator. Lean into it: "validated AI deployed on consumer devices" is precisely the demonstration Google Cloud likes to point to.
8. **Negative controls** — keep the sentence about negative controls tracking to 0.5. Most ML applications don't have this kind of methodological rigor and reviewers will notice.

Once your billing account ID is generated and the Pricing Calculator URL is saved, the application should be 20 minutes to submit end-to-end.
