# Membership-inference / memorization audit — plan

**Purpose.** Pre-emptive evidence for the AoU/Verily Controlled-Tier egress request that the released checkpoint (`best.pt`, 13.3M params) does **not** memorize training participants in a way that enables re-identification. A reviewer's core question is *"can this model leak who was in its training set?"* — this audit answers it with a number.

## Threat model
An attacker holds the released weights and a candidate window's features and wants to decide whether that window was in the training set. If the model assigns systematically **lower loss / higher confidence** to training members than to comparable non-members, the attacker can threshold that signal to infer membership.

## Method — loss-based threshold MIA (baseline, sufficient for this purpose)
1. **Balanced samples, comparable distributions.** Draw `N` member windows from the **train** split and `N` non-member windows from the held-out **test** split. Same cohort, same random-split pipeline → the two sets are distributionally matched, so a loss gap reflects *memorization*, not domain shift. (Use the same `make_loader_v3`, same masking.)
2. **Membership score = per-window loss.** For each window compute the model's loss exactly as in training (masked BCE summed over the prediction heads that are valid for that window; also report the headline `cv_composite_30d` head alone). Lower loss ⇒ "more likely a member."
3. **Attack performance.** Compute the **AUROC of (−loss ⇒ member)** over the pooled members + non-members. Also report the raw **mean loss gap** (train − test) and per-head breakdown.
4. **Interpretation.**
   - MIA-AUROC ≈ **0.50** and loss gap ≈ 0 ⇒ members and non-members are indistinguishable ⇒ **no detectable memorization** → strong support for export.
   - MIA-AUROC ≫ 0.50 ⇒ memorization present → mitigate (more regularization / early stop / DP) before requesting export.

## Controls & caveats
- **Distribution-shift control.** To be airtight, add a *second* non-member set drawn to match member calendar-time / utilization, so a residual loss gap can't be blamed on train/test drift. If the matched-control gap is also ~0, the case is clean.
- **This is the baseline, not the gold standard.** The rigorous version is a **shadow-model / LiRA** attack (train many shadow models on known in/out splits, calibrate a per-example likelihood-ratio test). That is far heavier (dozens of retrains) and is the escalation *only if* reviewers demand it. For a small model on a large cohort, the loss-threshold MIA plus the structural argument below is normally sufficient.

## Structural priors that already argue low risk (state these in the request)
- **Capacity vs data:** 13.3M parameters shared across ~12.5k participants and ~810k windows — very low per-example capacity; the model is forced to generalize, not store.
- **No retrieval/lookup component** — it is a feed-forward transformer over tokens, with no memory bank or nearest-neighbor table that could echo individuals.
- Standard training regularization; the export contains parameters only (verified by `prep_model_export.sh`'s manifest), not gradients/data.

## Output for the egress request
A short paragraph: *"Membership-inference AUROC = X.XX (≈ chance) with a mean train−test loss gap of Y.YYY; combined with the model's parameter-to-participant ratio and absence of any retrieval component, the checkpoint shows no detectable memorization of training participants."*

## Implementation
Skeleton at `workbench/mia_audit.py` — reuses the eval loaders/model. **Needs code review + one GPU pass** (train + test forward passes) before the number is trustworthy; it is not a finished result. Match the loss reconstruction to the exact training objective before relying on the gap.
