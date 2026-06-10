# v3 Training — Pre-Shutdown State Snapshot

**Captured:** 2026-06-10 12:04 UTC, while v3 supervised training was running on AoU CT workspace, just before the AoU initial-credits billing was expected to lapse.

This document is the source of truth for resuming v3 work from a new pod / new workspace if the current one becomes inaccessible.

---

## 1. GCP project + bucket identity

| Var | Value |
|---|---|
| **GOOGLE_PROJECT** | `wb-sparkly-lentil-9368` |
| **Actual workspace bucket** | `gs://phonefm-data-wb-sparkly-lentil-9368/` |
| **`$WORKSPACE_BUCKET` env (BROKEN)** | `gs://cloned-mybucket-wb-sparkly-lentil-9368` (404 — do NOT use) |
| **WORKSPACE_CDR** | `wb-silky-artichoke-2408.C2024Q3R8` |
| **WORKSPACE_UFID** | `phonefm-ct-cohort` |
| **Workspace name** | `PhoneFM CT cohort` |
| **App name** | `PhoneFM_CT_AoU_Jupyter` |
| **Pod (AoU-managed, read-only billing)** | `user-pod-oelemento-0ea3` (aou-prod org) |
| **GCP billing on pod** | `01043B-772AC9-33DDE1` (AoU-owned, not editable) |

The `WORKSPACE_BUCKET` env var is wrong (points to a non-existent bucket); the actual bucket is `gs://phonefm-data-wb-sparkly-lentil-9368/` and is fuse-mounted at `~/workspace/phonefm-data` per the `mount` output:

```
phonefm-data-wb-sparkly-lentil-9368 on /home/jupyter/workspace/phonefm-data type fuse.gcsfuse (rw,nosuid,nodev,relatime,user_id=1000,group_id=100,default_permissions,allow_other)
```

---

## 2. Bucket inventory (frozen at 2026-06-10 12:04 UTC)

**Total: 516 objects, 5.93 GiB**

Top-level contents:
- `tokenized_v3/` — 124 train + 13 val + 13 test parquet shards (~888 MB)
- `tokenized_v2/` — v2 tokenized data (still present)
- `phonefm_pretrain_v3/` — pretrain artifacts (~256 MB)
- `phonefm_v3/` — supervised artifacts (current run target)
- `phonefm_v2/` — v2 supervised artifacts (epoch 2 best.pt)
- `endpoint_concept_ids_v3.json` — codeset JSON (3.7 MB)
- `confounders_per_person_v3.parquet` — 9-dim confounders for 9,242 ppts
- `subgroup_metadata_v3.parquet` — race/sex/SES/birth_date
- Plus `cohort_base.parquet`, `splits.json`, `vocab.json`, `endpoint_concept_ids.json` from v2

Complete file listing is in `_snapshot_2026-06-10_v3_training/full_bucket_inventory.txt`.

---

## 3. Current training state

- **Process:** PID 3298, started ~11:27 UTC, etime ~35 min by 12:04
- **Progress at snapshot:** epoch 1 of 5, step ~13,800 / 48,280 (~29% total = ~57% through epoch 1)
- **Loss:** descending; bouncing 4-15 (down from 16-26 in warmup)
- **LR:** 8.30e-05 (cosine decay from 1e-04 peak at step 1000)
- **GPU:** 63% util, 4.6 GB used on A100-SXM4-40GB

### Epoch 0 results (already saved as best.pt at sum_primary_auroc=2.1032)

| Head | AUROC | Note |
|---|---|---|
| cv_composite_30d | **0.8775** | primary ⭐ |
| cv_composite_180d | 0.8598 | |
| cv_composite_365d | 0.8566 | |
| t2d_180d | 0.6343 | |
| t2d_365d | **0.6391** | primary |
| dep_180d | 0.6022 | |
| dep_365d | **0.5866** | primary |
| skin_neoplasm_365d | 0.6210 | neg control (target ~0.5) |
| refractive_errors_365d | **0.5286** | neg control ✓ |
| dental_caries_365d | 0.6366 | neg control |
| mortality_30d / 180d / 365d | NaN | n_pos=0 in val (all dropped from loss too) |

**Comparison to v2:** v2 AFib val AUROC at epoch 2 = 0.9285 (cluster bootstrap). v3 cv_composite_30d val at epoch 0 = 0.8775 — already in the same range, expected to improve through epochs 1-4.

**Negative control flags:**
- refractive_errors clean at 0.53 (good)
- skin (0.62) and dental (0.64) mildly above 0.5 — possible utilization-confounding signal in wearable data. Worth flagging in any paper but not a fail.

---

## 4. Critical artifacts that must survive

All in `gs://phonefm-data-wb-sparkly-lentil-9368/`:

| Artifact | Path within bucket | Status |
|---|---|---|
| Pretrain backbone | `phonefm_pretrain_v3/backbone_only.pt` (52 MB) | ✅ done, val_masked_MSE=0.1805 at epoch 19 |
| Pretrain full | `phonefm_pretrain_v3/best.pt` (52 MB) | ✅ done |
| Pretrain optimizer state | `phonefm_pretrain_v3/last.pt` (157 MB) | ✅ done |
| Supervised best.pt | `phonefm_v3/best.pt` (53 MB) | ✅ epoch 0 saved |
| Supervised last.pt | `phonefm_v3/last.pt` (53 MB) | ✅ updates each epoch |
| Supervised config | `phonefm_v3/config.json` | ✅ |
| Tokenized v3 shards | `tokenized_v3/{train,val,test}_*.parquet` | ✅ 150 shards total |
| Codeset JSON | `endpoint_concept_ids_v3.json` | ✅ |
| Confounders | `confounders_per_person_v3.parquet` | ✅ |
| Subgroup metadata | `subgroup_metadata_v3.parquet` | ✅ |

---

## 5. Resume procedure (when on a new pod with bucket access)

```bash
# Verify bucket access
gsutil ls gs://phonefm-data-wb-sparkly-lentil-9368/ | head

# Ensure repo at right commit
cd ~/repos/PhoneFM
git fetch -q origin
git reset --hard origin/v3-spec   # currently at f474201

# Env fix (always needed on fresh AoU pods)
pip install --quiet 'numpy<2'
pip install --quiet --force-reinstall --no-deps pyarrow
python3 -c "import torch, numpy; assert torch.from_numpy(numpy.zeros(3)).sum().item()==0; print('OK', numpy.__version__, torch.__version__)"

# If new workspace mounts data differently, find the bucket:
mount | grep gcsfuse
# Expected to see something like:
#   phonefm-data-wb-sparkly-lentil-9368 on /home/jupyter/workspace/phonefm-data type fuse.gcsfuse
```

### Resume supervised training from saved last.pt

If we want to keep training where we left off:

```bash
cd ~/repos/PhoneFM/workbench
# Edit 05_train_v3.py to load from last.pt at the start of training
# (currently it only loads pretrained backbone; would need a manual edit)
```

### Run eval on best.pt as-is

If we just want the current epoch-0 numbers locked in:

```bash
cd ~/repos/PhoneFM/workbench
python3 06_eval_v3_test.py   # writes test_results.json with cluster-bootstrap CIs
python3 06_subgroup_analysis.py
```

---

## 6. Data export approval (export model weights out of CT)

Per AoU DUA:
- **Cannot just `gsutil cp` `.pt` files out** of the workspace bucket to a personal/lab bucket
- Must initiate the **export-approval review** before weights can leave the CT
- Logs, metrics, configs (this file, run.log, metrics.json) are generally exportable but should still go through review for anything traceable
- See `support.researchallofus.org` for the export workflow

---

## 7. AoU initial-credits billing (the reason for this snapshot)

- AoU researchers get $300 in initial credits attached to a managed billing account they don't control: `01043B-772AC9-33DDE1` for this user
- That billing account is **read-only to researchers** — the "You don't have access" error in the pod Edit dialog is by design
- To use a different billing account (e.g. WCM lab GCP), you must:
  1. Set up your own GCP billing account; add `billing@workbench.verily.com` as Billing Account User
  2. **Duplicate the workspace** and select your own pod/billing at creation time — billing is fixed per workspace
- Verify credit balance: Verily Workbench profile page

---

## 8. Open questions to resolve

- **Mortality endpoint is sparse** (n_pos=0 in val, 15-32 in train across the three horizons). Either the AoU death table is genuinely sparse in our cohort (~6,496 train ppts), or there's a labeling-boundary bug in `02_tokenizer_v3.py`. The 545-d cohort filter may also be over-restricting. Investigate before claiming "mortality" as a results endpoint.
- **Negative controls (skin, dental)** at 0.62-0.64 — mild but real upward bias. Likely picks up utilization signal (people who see doctors get diagnosed AND wear Fitbits). Worth showing in the paper but does affect "wearable signal is real" framing.
- **Should supervised training continue past epoch 0?** Epoch 0 already saved best.pt at sum_primary_auroc=2.1032. Epochs 1-4 might improve, but the run will also burn ~1.5h more A100 ($5.50). If credits expire mid-run we lose the time but not the data.

---

## 9. Files inside `_snapshot_2026-06-10_v3_training/` (bucket-internal copies)

```
endpoint_concept_ids_v3.json        3.7 MB
full_bucket_inventory.txt           56 KB
v3_pretrain_config.json             876 B
v3_pretrain_metrics.json            1.7 KB
v3_pretrain_run.log                 130 KB
v3_supervised_config.json           4.9 KB
v3_supervised_metrics.json          3.3 KB   (epoch 0 only at snapshot time; updates as training progresses)
v3_supervised_run.log               23 KB    (frozen at step ~13,800)
```

Snapshot taken 2026-06-10 12:04 UTC. Re-run the snapshot to refresh:

```bash
SNAP=~/workspace/phonefm-data/_snapshot_$(date +%Y-%m-%d_%H%M)_v3_training
mkdir -p $SNAP
cp ~/workspace/phonefm-data/phonefm_v3/run.log $SNAP/v3_supervised_run.log
cp ~/workspace/phonefm-data/phonefm_v3/metrics.json $SNAP/v3_supervised_metrics.json
cp ~/workspace/phonefm-data/phonefm_v3/config.json $SNAP/v3_supervised_config.json
gsutil ls -lr gs://phonefm-data-wb-sparkly-lentil-9368/ > $SNAP/full_bucket_inventory.txt 2>&1
```
