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

**Project name:** On-Device Cardiovascular Foundation Model from Wearable + EHR Data

**Project start date:** 2026-06-23 (or whatever works)

### Proposal (~245 words; 250 word maximum)

Cardiovascular disease is the leading global cause of death, yet current risk tools such as ASCVD require clinic visits and a handful of static inputs. Smartphones already collect continuous wearable signals, but no validated on-device foundation model converts these into actionable cardiovascular risk.

Our preliminary foundation model is a 13-million-parameter transformer pretrained on 920K wearable + EHR windows from the All of Us cohort. It achieves AUROC = 0.88 on a 30-day cardiovascular composite (AFib + MI + HF), while pre-registered negative-control endpoints (skin neoplasm, refractive errors, dental caries) track near 0.5, consistent with capturing real wearable physiology rather than healthcare-utilization confounders.

The proposed work pursues four research questions. (M1–3) Which architectures, tokenizations, and input modalities (steps, heart rate, sleep, polygenic risk scores, labs, social determinants) most improve over clinical risk scores and tree-based baselines, with per-modality ablations quantifying the contribution of each signal? (M4–6) Does the model transfer to UK Biobank, and where do demographic performance gaps emerge across pre-registered subgroups (race, age, sex, SES)? (M7–9) Can the model run on consumer smartphones (iPhone Core ML/HealthKit; Android TensorFlow Lite/Health Connect) with acceptable latency, ingesting locally-stored wearable signals and producing on-device cardiovascular predictions? (M10–12) Can federated continuous learning across simulated sites match centralized performance? Results will be reported in a manuscript.

Google Cloud underpins every question, with Vertex AI Training (A100/H100) for architecture sweeps and ablations, BigQuery for cohort extraction from All of Us and UK Biobank, Cloud Storage for data and checkpoints, and Cloud Healthcare API for FHIR integration.

### Field of research

Health and Life Sciences (medical AI / digital health)

---

## Additional information

### Google Scholar link

https://scholar.google.com/citations?user=IP7KtcQAAAAJ

### How do you intend to apply awarded cloud credits?

**Train machine learning models** (Vertex AI Training) and **process/analyze large datasets** (BigQuery). Specifically: scale PhoneFM backbone training, run cohort extraction across multiple biobanks, and prototype federated continuous-learning.

### Does your research project have an external funding source?

**Yes.** This project is currently supported by Weill Cornell Medicine / Englander Institute for Precision Medicine internal research funds. PI Olivier Elemento holds multiple active NIH grants across the broader translational AI program at EIPM. PhoneFM is the preliminary-data engine for a planned NIH R01 submission within the requested credit period — the credit-supported work directly enables that grant application.

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

**Program cap (verified 2026-06-10 via Cloud for Education Help):** Faculty / postdoc / non-profit-lab researcher awards are worth **up to $5,000 USD** (PhD students up to $1,000). One award per faculty member; additional credits possible via the referral program (refer 2 qualified applicants → eligible for new grant).

**Recommended ask:** **$5,000** (the program cap). The calculator estimate above (~$6,100/yr) slightly exceeds this; the request fully consumes the credit on the planned A100 backbone + cohort-extraction work, with any overage absorbed by Englander Institute internal funds. For scale-ups beyond the credit (50M-parameter backbone, federated multi-institution training, longer ablation sweeps), pursue the referral pathway and/or transition to NIH R01 funding.

### Google Cloud Billing Account ID

**`01043B-772AC9-33DDE1`** — currently funding the active PhoneFM CT cohort (Copy) workspace `wb-shrewd-lime-9770` on Verily Workbench.

### After your credit expires, how do you intend to continue funding your project?

Active NIH R01 grants held by PI Elemento, a planned R01 for which this work generates preliminary data, Englander Institute for Precision Medicine infrastructure budget, and Weill Cornell Medicine departmental research support.

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
