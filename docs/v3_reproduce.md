# v3 — How to Reproduce + Mistakes to Avoid

Compact playbook for re-running PhoneFM v3 from scratch in a new AoU CT workspace (or the same one, on a fresh pod). Pair this with `docs/v3_resume_state_2026-06-10.md` for the snapshot-in-time state.

The goal: anyone (including future-you) should be able to follow these steps in order and arrive at the same v3 model without re-discovering the bugs we already fixed.

---

## TL;DR — the canonical pipeline

```
build_endpoint_concept_ids_v3.py   (~15 min, CPU, BQ-bound)   ─┐
02_tokenizer_v3.py                  (~3-4 h, CPU n1-highmem-2) ─┼─ tokenized data + JSON
04_pretrain_v3.py                   (~3-4 h, A100)             ─┤
05_train_v3.py                      (~2-3 h, A100)             ─┤
06_eval_v3_test.py                  (~30 min, A100)            ─┤
06_subgroup_analysis.py             (~30 min, A100)            ─┘
```

All four scripts assume the repo is at branch `v3-spec`, currently at commit `f474201`.

---

## 1. Branch + commit alignment

```bash
cd ~/repos/PhoneFM
git fetch -q origin
git reset --hard origin/v3-spec
git log --oneline -5
# Expected top commit: f474201 "Fix tz-aware vs tz-naive crash in age_at_end_date"
```

**Gotcha:** `git pull -q` has been unreliable on the workbench pods (silent stale state). **Always use `git fetch origin && git reset --hard origin/v3-spec`** when you need to be sure.

---

## 2. Environment setup (REQUIRED on every fresh pod)

```bash
pip install --quiet 'numpy<2'
pip install --quiet --force-reinstall --no-deps pyarrow
python3 -c "import torch, numpy; assert torch.from_numpy(numpy.zeros(3)).sum().item()==0; print('OK', numpy.__version__, torch.__version__)"
# Expected output:  OK 1.26.4 2.0.1+cu118
```

**Why:** Fresh AoU pods ship with NumPy 2.x, but `torch==2.0.1+cu118` was compiled against NumPy 1.x. `torch.from_numpy(np.zeros(3))` raises `RuntimeError: Numpy is not available` at the first DataLoader batch and the entire training run dies inside a worker. The `--force-reinstall --no-deps pyarrow` step is needed because the system's pyarrow was also compiled against NumPy 2.

You'll see noisy warnings from `hail`, `tensorflow`, `pandas-profiling` about "numpy 1.26.4 is incompatible" — **ignore them**, those packages aren't used by PhoneFM.

This is the FIRST thing that fails on every new pod. Always do it before any python script that touches torch.

---

## 3. Find the real workspace bucket

```bash
mount | grep gcsfuse
# Look for a line like:
#   phonefm-data-wb-sparkly-lentil-9368 on /home/jupyter/workspace/phonefm-data type fuse.gcsfuse
```

**Gotcha:** `$WORKSPACE_BUCKET` env is wrong (points at a `cloned-mybucket-...` path that 404s on gsutil). Use the actual bucket name from `mount`. For this workspace it's `gs://phonefm-data-wb-sparkly-lentil-9368/`.

All artifacts live under `~/workspace/phonefm-data/` which is fuse-mounted to that bucket.

---

## 4. Codeset build — `build_endpoint_concept_ids_v3.py`

```bash
cd ~/repos/PhoneFM/workbench
nohup python3 -u build_endpoint_concept_ids_v3.py > ~/workspace/phonefm-data/build_codesets.log 2>&1 < /dev/null &
```

**Expected self-test output:** dep.dx=112,614 persons, refractive_errors.dx=57,555, dental_caries.dx=18,590. If any of those reads 0, the SNOMED roots in the script are wrong again (see below).

### Mistakes we fixed
- `dep.dx` original roots `[192080]` matched **0** persons. Correct: `[440383, 4152280]` (Depressive disorder + Major depressive disorder).
- `refractive_errors.dx` original roots `[4218554]` matched **0**. Correct: `[4191597, 377861, 4182553]` (Disorder of refraction + Disorder of refraction AND/OR accommodation + Disorder of accommodation).
- `dental_caries.dx` original roots `[4210708]` matched **0**. Correct: `[133228]`.

**How to find the right SNOMED roots if AoU CDR drifts:** see the BQ-query approach in `docs/v3_resume_state_2026-06-10.md` §8 — use `concept_relationship` 'Maps to' from the ICD-10 leaf codes back to their SNOMED standard targets.

---

## 5. Tokenization — `02_tokenizer_v3.py`

```bash
cd ~/repos/PhoneFM/workbench
nohup python3 -u 02_tokenizer_v3.py > ~/workspace/phonefm-data/tokenize_v3.log 2>&1 < /dev/null &
```

**Expected:** cohort filter 9,242 / 12,453 (74.2%), train 6,496 / val 1,356 / test 1,390. Writes 124 + 13 + 13 = 150 parquet shards (~888 MB). First-shard sanity check should print `sanity OK: train_0000.parquet wearable_feats max = 22876.0` (or similar non-trivial value > 100).

### Mistakes we fixed
1. **BQ helpers diverged from v2's verified queries** — my initial v3 draft of `fetch_sleep_daily` referenced `s.sleep_start_time` (column doesn't exist) and used `is_main_sleep = TRUE` (BOOL) against a STRING column. The whole tokenizer crashed at chunk 1. **Fix:** ported `fetch_steps_daily / fetch_hr_daily / fetch_sleep_daily / fetch_ehr_events` verbatim from `02_tokenizer_v2.py` (commit `11a877f`). **Lesson:** copy-paste from a known-working v2 helper; do NOT re-derive BQ queries from memory.
2. **age_at_end_date tz crash** — AoU's `person.birth_datetime` is tz-aware UTC; `.dt.normalize()` preserves tz; subtracting from tz-naive `end_date` (derived from `fb_start.dt.normalize()`) raises `TypeError: Cannot subtract tz-naive and tz-aware datetime-like objects`. **Fix:** in `precompute_subgroup_metadata`, use `pd.to_datetime(df["birth_datetime"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()`. Defensive guard also in `encode_window_v3` (commit `f474201`).
3. **Event-on-end_date label boundary** — events on exactly `end_date` fell in neither input window (`d < end_date`) nor horizon (`d > end_date`). Silent label loss on true positives. **Fix:** include `end_date` in horizon (`d >= end_date`). M1 finding in code review.
4. **Schema assertion only on shard 0** — orphan shards from a prior run with a different head list would crash mid-training with KeyError. **Fix:** iterate the assertion across ALL shards in `PhoneFMV3Dataset.__init__`. M2 finding.
5. **age_at_end_date year-difference biased late-year births** — formula `fb_start.year - yob` overestimates by 0-11 months. Persons born in December are pushed one age-bin up. **Fix:** use actual `birth_date` from `person.birth_datetime` and compute `(end_date - birth_date).days // 365`. Falls back to year-diff if AoU has suppressed the birth date.

---

## 6. Self-supervised pretrain — `04_pretrain_v3.py`

```bash
cd ~/repos/PhoneFM/workbench
nohup python3 -u 04_pretrain_v3.py > ~/workspace/phonefm-data/phonefm_pretrain_v3/run.log 2>&1 < /dev/null &
```

**Expected:** 20 epochs × 4988 steps/epoch = 99,760 total. ~3-4h on A100. val_masked_MSE descends from ~0.40 (epoch 0) → 0.18 (epoch 19). Writes `backbone_only.pt` (52 MB) — this is what `05_train_v3.py` will load.

### Mistakes we fixed (commit `04d74c9`)
1. **Confounder placeholder modules had wrong shape** — `PhoneFMV2Config` default `n_confounders=8`; v3 supervised uses 9 (adds baseline_osa). Pretrain model declared `confounder_input_norm = BatchNorm1d(8)` and saved them into `backbone_only.pt`. v3 `load_state_dict(strict=False)` HARD-CRASHED with "size mismatch" (strict=False does NOT silence shape mismatches). **Fix:** removed the confounder layers entirely from `PhoneFMPretrainV3` — pretrain forward doesn't take confounders anyway. v3 supervised builds them fresh.
2. **`target_std` buffer leaked into backbone_only.pt** — `register_buffer("target_std", ...)` was a real M3 fix for per-feature MSE standardization, but the save filter only excluded `pretrain_head.` keys. `target_std` ended up in `backbone_only.pt`. v3 supervised has no such buffer → `unexpected_keys` non-empty → my explicit guard in `05_train_v3.py:138` correctly raised. **Fix:** added `target_std` to the save-filter exclusion in `04_pretrain_v3.py:493-497`.
3. **Dropout 0.10 vs spec 0.15** — minor; bumped to match v3 supervised.

**Lesson:** when you save a "backbone_only" subset, you need to know EXACTLY which keys are in the backbone vs not. The filter must be aligned with the consuming model. Adding a code-review pass before kicking off pretrain caught this — saved an entire 3.5h pretrain run that would have produced an unusable checkpoint.

---

## 7. Supervised — `05_train_v3.py`

```bash
cd ~/repos/PhoneFM/workbench
nohup python3 -u 05_train_v3.py > ~/workspace/phonefm-data/phonefm_v3/run.log 2>&1 < /dev/null &
```

**Expected:** 13 heads, 5 epochs × 9656 steps/epoch = 48,280 total. ~2-3h on A100. First epoch eval prints all per-head AUROCs. Best-metric is `sum_primary_auroc` over `cv_composite_30d`, `mortality_365d`, `t2d_365d`, `dep_365d`.

### Mistakes we fixed (commit `04d74c9`, C3 finding)
1. **Degenerate-val all-NaN failure mode** — tiny val + rare endpoint → all primary AUROCs NaN → `sum_primary_auroc = 0.0` → `0.0 > -1.0` (initial best) → epoch-0 random-init heads saved as `best.pt` → script silently early-stops. The headline numbers would be from an untrained model. **Fix:** preflight check refuses to start if every primary head has val n_pos<5. At eval time, refuses to save `best.pt` if `n_contrib==0` (no primary head returned a real AUROC).
2. **Mortality endpoints are sparse in AoU** — at v3 cohort sizes (6,496 train, 1,356 val, 1,390 test), val mortality_30d / 180d / 365d all have n_pos=0. The preflight check correctly catches this and **falls back to sum over the 3 viable primary heads** (cv_composite_30d, t2d_365d, dep_365d). This is a structural feature of the data, not a bug. If you want mortality to work you need a bigger cohort or a longer max-horizon.

### Other supervised-side gotchas
- **Preflight reports `WARNING: dropping heads ['mortality_30d', 'mortality_180d', 'mortality_365d'] from loss (<50 positives in train)`** — expected for this cohort. Heads still emit logits at eval but don't contribute to loss.
- **Negative controls (skin/refractive/dental)** should land near AUROC 0.5 ± 0.05. If they're consistently >0.6 across runs, that's a utilization-confounding signal worth flagging in the paper.

---

## 8. Eval — `06_eval_v3_test.py` and `06_subgroup_analysis.py`

```bash
cd ~/repos/PhoneFM/workbench
python3 06_eval_v3_test.py        # writes test_results.json with cluster-bootstrap CIs
python3 06_subgroup_analysis.py   # writes subgroup_results.json
```

### Mistakes we fixed (commit `04d74c9`)
- **Dead importlib block** in `06_eval_v3_test.py` that imported v2's `cluster_bootstrap_metric` but never called it — and triggered v2's module-level `print` as a side effect. **Removed.**
- **RNG mismatch with v2** — used `np.random.default_rng`; v2 uses `np.random.RandomState`. CIs not bit-comparable to v2. **Fixed** to `RandomState(20260609)` for parity.
- **Tuple round-trip on head_specs** — saved as list-of-lists, loaded with `tuple((ep, h) for (ep, h) in ...)` which kept inner items as lists. **Fixed** to `tuple(tuple(s) for s in ...)`.

---

## 9. Subgroup definitions JSON (`subgroup_definitions.json`)

The file is **pre-registered** and the SHA1 is checked by both eval scripts. If you edit it (even whitespace), the eval will detect the drift and the analysis is invalid. Either revert OR pre-register a v2 of the file BEFORE running new analyses.

---

## 10. Workbench operational gotchas

### Pod swap (CPU ↔ A100)
1. Stop the app via the apps page Stop button. After click, **navigate away + back** to see the updated state — the page caches "Running" status stalely. Same issue with Start clicks.
2. Edit machine type via kebab → Edit. App MUST be Stopped first.
3. **For A100:** Compute tab → GPUs → NVIDIA Tesla A100 → Update. A100 80GB is grayed out as "Only available on app creation."
4. **For CPU:** Compute tab → General → typeahead "n1-highmem-2" in the Machine type dropdown.
5. Start. **The Start button HTML coordinates differ from screenshot coordinates by ~70 px** on Verily Workbench — if `computer.left_click` at screenshot coordinates does nothing, fall back to JavaScript: `Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'Start').click()`.

### Pod startup
- Fresh pod hostname will differ each time (`530e696b06ef` → `3617113aaa07` → `2c0132fb3aae` so far). The bucket fuse-mount survives the change.
- A100 boot takes 3-5 min; CPU is faster.
- Setting up the Python env (numpy<2 + pyarrow) takes ~30s.

### Process management
- `kill PID` only kills the parent; DataLoader workers continue. **Always use `pkill -9 -f script_name.py`** to clean up before restart.
- Verify with `ps -eo cmd | grep <script_name> | grep -v grep | wc -l` — should print 0 before relaunch.

### Browser extension flakiness
- Claude-in-Chrome extension disconnects intermittently. **nohup-detached processes survive** but live monitoring breaks. Set up the extension again if needed (`claude.ai/chrome`).
- A reload of the workbench tab is sometimes needed after a disconnect.

---

## 11. Sequence of git commits (the audit trail)

If you're reading this in the future and wondering why a particular file looks the way it does, the commits tell the story:

```
f474201  Fix tz-aware vs tz-naive crash in age_at_end_date
11a877f  Port v2's verified BQ helpers into v3 tokenizer
5108065  Fix broken SNOMED roots for dep / refractive_errors / dental_caries
04d74c9  v3 pipeline fixes from correctness + adversarial review
09d1f9a  v3 supervised pipeline draft (13 heads, multi-horizon, masked BCE)
215d2bd  Val cluster-bootstrap eval for direct comparison with test
```

Run `git log --stat <commit>` to see exactly what changed.

---

## 12. Things to validate after every full re-run

Before claiming a v3 result is reproducible, verify these:

- `endpoint_concept_ids_v3.json` self-test: dep.dx ~112k persons, refractive_errors.dx ~57k, dental_caries.dx ~18k (±10% drift is acceptable across AoU CDR releases).
- Tokenizer: cohort retention ~74% of v2's 12,453 (~9,242 ppts after 545d filter).
- Tokenizer: `sanity OK: train_0000.parquet wearable_feats max = <number>` printed with max > 100.
- Pretrain: val_masked_MSE descends from ~0.4 to ~0.18 over 20 epochs.
- Supervised: epoch 0 cv_composite_30d val AUROC > 0.85 (~baseline). If it's stuck at 0.5, the pretrained backbone failed to load — check the `loaded backbone OK. N missing keys` line.
- Negative controls (skin_neoplasm_365d, refractive_errors_365d, dental_caries_365d) val AUROC < 0.65. If > 0.7, suspect data leakage.

---

## 13. What we know we DON'T know yet (after v3 epoch 0)

- Does cv_composite_30d AUROC improve in epochs 1-4? At epoch 0 it was 0.8775. Need to inspect the full metrics.json once supervised training completes.
- Is the negative-controls upward bias (0.62-0.64) reproducible across different cohort splits? Could be utilization confounding (people who get diagnosed are people who see doctors), could be sampling noise.
- Subgroup analysis hasn't been run yet. Race × age interaction may surface disparities worth highlighting (or worth investigating).
- The mortality endpoint is essentially unobservable in this cohort. Either accept it as a structural limitation OR redesign the cohort (require >2 years follow-up; include older participants) to make mortality detectable.

If you find answers to these in a future session, please add them here.
