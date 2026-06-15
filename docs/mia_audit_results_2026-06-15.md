# Membership-inference audit — results

**Date:** 2026-06-15
**Model:** PhoneFM v3, 13.3M-parameter transformer pretrained on All of Us Fitbit + EHR data; supervised checkpoint `best.pt` (epoch 0, sum_primary_auroc = 2.1032).
**Audit code:** `workbench/mia_audit.py` v2.1 @ `0158756` on `v3-spec` (post-adversarial-review).
**Pod:** Copy workspace (`wb-shrewd-lime-9770`), T4 GPU, n1-standard-8, fp32 (T4 lacks bf16).
**Wall-clock:** ~25 min.
**Artifact:** `gs://phonefm-data-wb-shrewd-lime-9770/phonefm_v3/mia_audit.json` (5,480 B). Aggregate / de-identified — no PHI.

---

## Headline

**Primary metric (cv_composite_30d single-head BCE):**

| Comparison | Point AUROC | 95% CI | Reading |
|---|---|---|---|
| **train vs test** | **0.5048** | **[0.4827, 0.5261]** | clean pass |
| train vs val (diagnostic) | 0.5011 | [0.4793, 0.5227] | pass even with checkpoint-selection bias |

CI upper bound 0.5261 is comfortably below the 0.55 pre-registered threshold; interval covers chance (0.50). **No detectable memorization of training participants.**

**Multihead score (corroboration, sum_h [head_weight * BCE * mask] over 10 active heads):**

| Comparison | Point AUROC | 95% CI |
|---|---|---|
| train vs test | 0.5054 | [0.4827, 0.5283] |
| train vs val (diagnostic) | 0.5197 | [0.4970, 0.5417] |

Multihead corroborates the primary at train-vs-test; diagnostic val arm is borderline (CI HI 0.5417) but expected given checkpoint-selection bias and not used as a verdict-eligible signal.

---

## Per-arm summary (n_persons=1300 per arm, K=8 windows per person, n_windows=10,400)

| Arm | Per-person primary loss | Per-person multihead score |
|---|---|---|
| train | 0.5491 | 4.7910 |
| test | 0.6874 | 6.3955 |
| val | 0.9389 | 6.3232 |

Train-test loss gap (~0.14 nats primary) is real but small — consistent with normal generalization, **not** the per-window leakage that MIA detects. Within-arm variance dwarfs between-arm gap, which is why AUROC stays near 0.50.

---

## Methodology

- **Loss replication:** cv_composite_30d single-head BCE with pos_weight matching training; multihead is the un-batch-normalized sum across active heads (heads with n_pos≥50 in train, matching `05_train_v3.py` rule).
- **Active heads (10):** cv_composite_{30,180,365}d, t2d_{180,365}d, dep_{180,365}d, skin_neoplasm_365d, refractive_errors_365d, dental_caries_365d. Mortality (3 heads) dropped at training time and at audit time.
- **Sampling:** persons-first, deterministic plan via `np.random.default_rng(seed)` with per-arm seeds (test=SEED+10, val=SEED+20, train=SEED+30). Up to K=8 windows per person. Sample plan verified at runtime via per-batch person_id assertion.
- **Aggregation:** per-person mean loss across the K (≤8) sampled windows, ignoring NaN windows.
- **Bootstrap:** person-cluster, 2000 resamples, two-sided 95% CI; persons resampled with replacement within each arm separately.
- **Precision:** fp32 throughout on T4 (Turing arch lacks bf16). The original v2 script attempted bf16 to match training; v2.1 (commit `0158756`) falls back to fp32 when `torch.cuda.is_bf16_supported()` returns False. fp32 is a defensibly conservative audit-time precision (more accurate than training bf16, not less).
- **Reproducibility:** num_workers=0, shuffle=False, Subset over deterministic sampling plan, per-arm seeds. Same seed produces same numbers across reruns.

---

## Caveats

1. **Loss-threshold MIA only.** A clean pass here rules out memorization detectable by loss-based MIA. Stronger attacks (shadow-model LiRA, model inversion) are not run; standard egress practice accepts the loss-threshold MIA as evidence given the structural priors below.
2. **Val arm is diagnostic, not a verdict-eligible non-member control.** best.pt was selected on val via early-stopping, so any train-vs-val gap conflates memorization with checkpoint-selection bias. Reported for transparency, not used as a primary signal.
3. **Multihead score is NOT the training loss.** Training's per-head batch-level normalizer (`1/mask.sum()`) cannot be reconstructed at audit time; the multihead metric is a complementary diagnostic, documented honestly in the JSON methodology block.
4. **Cross-stream column-count confound** (per the §16/§17 ablation caveats): the multihead score sums BCE across heads with very different positive rates and pos_weights; magnitude comparisons across head subsets are confounded. The primary single-head metric is not affected.

---

## Structural priors that reinforce the result (cite in egress request)

- **Capacity vs data:** 13.3M parameters shared across ~6,496 train participants and ~620K train windows — very low per-example capacity; the model is forced to generalize rather than memorize.
- **Architecture:** plain feed-forward transformer with rotary positional embedding on calendar-day index; no memory bank, retrieval head, nearest-neighbor table, or any other component that could "echo" individuals.
- **Training:** standard regularization (weight decay 0.01, dropout, early-stopping); best.pt is epoch 0 with effectively ~9.6K supervised gradient steps on top of the pretrained backbone.
- **Egress contains parameters only:** verified by `prep_model_export.sh`'s manifest (sha256 + parameter count + dtype histogram + non-tensor key inspection).

---

## Recommended paragraph for the AoU egress request

> "Membership-inference audit on the proposed export checkpoint (`best.pt`, 13.3M parameters) was performed via a loss-threshold attack with person-cluster bootstrap (n=1300 train participants vs n=1300 held-out test participants, 8 windows per person, 2000 bootstrap resamples). The primary attack AUROC (cv_composite_30d single-head BCE, the headline endpoint) was **0.5048 with 95% CI [0.4827, 0.5261]** — comfortably below the 0.55 pre-registered threshold and statistically indistinguishable from chance. A complementary multi-head BCE attack gave an equivalent reading (0.5054 [0.4827, 0.5283]). Combined with the model's low parameter-to-participant ratio (13.3M / 6,496 train participants), the absence of any retrieval or memory-bank component in the architecture, and the parameter-only manifest of the egress bundle (sha256 attached), the checkpoint shows no detectable memorization of training participants."

---

## Reproducibility

```bash
# Pod: T4 + n1-standard-8 in Copy workspace
cd ~/repos/PhoneFM/workbench
git pull origin v3-spec   # at least 0158756
pip install --quiet 'numpy<2'   # one-time, fixes torch.from_numpy on numpy 2.x
nohup python3 -u mia_audit.py > /tmp/mia_audit.log 2>&1 &
```

Output: `~/workspace/phonefm-data/phonefm_v3/mia_audit.json`. Wall-clock ~25 min on T4 fp32. Same seed → same numbers.
