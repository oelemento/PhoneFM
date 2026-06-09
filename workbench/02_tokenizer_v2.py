"""PhoneFM v2 tokenizer — daily-aggregate wearable + EHR events + per-endpoint labels.

Designed to be run on Workbench Jupyter (CPU machine is fine; it's BQ + pandas bound).

INPUTS (assumed to be on the bucket FUSE mount):
  /home/jupyter/workspace/phonefm-data/cohort_base.parquet         (12,453 participants, fitbit_start/end)
  /home/jupyter/workspace/phonefm-data/splits.json                 (train/val/test person_id lists)
  /home/jupyter/workspace/phonefm-data/endpoint_concept_ids.json   (afib/mi/hf ICD-10-CM source-concept ids)
  /home/jupyter/workspace/phonefm-data/vocab.json                  (EHR token name -> token id)

OUTPUTS:
  /home/jupyter/workspace/phonefm-data/tokenized_v2/train_NNNN.parquet  (shards of 5000 rows)
  ...val_NNNN.parquet, test_NNNN.parquet

INPUT WINDOW: 180 days of Fitbit + EHR
PREDICTION HORIZON: 30 days after the window end
TRAIN STRIDE: 14 days (sliding windows per participant)
VAL/TEST STRIDE: 30 days

WEARABLE FEATURE COLUMNS (per day; np.float32, shape [180, 11]):
  0: total_steps
  1: mean_hr (5-min bins -> mean of means across day)
  2: resting_hr (10th percentile of 5-min HR over day)
  3: max_hr (95th percentile of 5-min HR over day)
  4: sdann (SD of 5-min mean HR, single number per day)
  5: rem_pct (% of main-sleep period in REM)
  6: deep_pct
  7: light_pct
  8: sleep_duration_hr (main sleep total)
  9: sleep_onset_hour (clock hour 0-24, 0 if no sleep that day)
 10: sleep_irregularity_7d_sd (SD of last 7 days of sleep_duration_hr, computed Python-side)

If a day has no Fitbit data: wearable_mask[d] = False, wearable_feats[d] = 0.

EHR token types and id space (must match the v1 vocab.json):
  4=DX10  (ids 1000-2357)
  5=MED   (ids 4000-4230)
  6=PX10  (ids 5000-5423)
  7=LAB   (ids 6000-6006)

CONFOUNDERS (per participant, shape [8]) — order matches PhoneFMV2Config:
  0: age (years, raw — model learns scaling via input Linear)
  1: sex_female (0/1; AoU sex_at_birth_concept_id == 45878463)
  2: bmi (kg/m^2, raw)
  3: sbp (mmHg, raw — systolic; concept 3004249, most recent before fitbit_start)
  4: dbp (mmHg, raw — diastolic; concept 3012888)
  5: baseline_cad (0/1; reuses ENDPOINT_IDS['cad_baseline'] if present, else 0)
  6: baseline_cancer (0/1; ENDPOINT_IDS['cancer_baseline'])
  7: prior_afib (0/1; ANY afib code before fitbit_start — reuses ENDPOINT_IDS['afib'])

LABELS (per window):
  label_afib, label_mi, label_hf, label_cv_death, label_composite
"""

# TODO before first run: verify these AoU CDR v8 tables exist with the assumed columns.
# Quick check from Jupyter terminal:
#   bq query --use_legacy_sql=false "SELECT column_name FROM `{CDR}.INFORMATION_SCHEMA.COLUMNS`
#     WHERE table_name = 'sleep_daily_summary'"
#
#   Likely tables (from Zheng 2024 supplement + AoU CDR v8 docs):
#     sleep_daily_summary          (person_id, sleep_date, total_sleep_minutes, etc.)
#     sleep_level                  (person_id, start_datetime, end_datetime, level_name in {rem,deep,light,wake})
#     activity_summary             (person_id, activity_date, steps, ...)
#     steps_intraday               (person_id, datetime, steps)         -- minute-level
#     heart_rate_minute_level      (person_id, datetime, heart_rate_value)
#     heart_rate_summary           (person_id, date, ...zone_name...)   -- HR zones, NOT HRV

# Schema versions of these queries are PLACEHOLDERS until first-run verification on the VM.

# ============================================================
# CELL 1 — environment
# ============================================================
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-not-found]
from google.cloud import bigquery  # type: ignore[import-not-found]

bq = bigquery.Client()
CDR = os.environ["WORKSPACE_CDR"]
print(f"CDR = {CDR}", flush=True)

DATA_DIR = Path("/home/jupyter/workspace/phonefm-data")
OUT_DIR = DATA_DIR / "tokenized_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_DAYS_INPUT = 180
N_DAYS_HORIZON = 30
TRAIN_STRIDE = 14
EVAL_STRIDE = 30
SHARD_SIZE = 5000
CHUNK_SIZE = 200   # participants per BQ chunk; same as v1 (constrained by to_dataframe peak memory)


# ============================================================
# CELL 2 — load cohort + splits + vocab
# ============================================================
cohort = pd.read_parquet(DATA_DIR / "cohort_base.parquet")
# CRITICAL: normalize cohort timestamps to midnight. AoU stores fitbit_start/end
# as full DATETIME values (e.g. '2021-02-17 07:23:15'). All wearable/EHR keys
# derived from BQ are at midnight via .dt.normalize(). If we don't strip the
# time component here, `day in steps_by_d` returns False for EVERY day in EVERY
# window, producing all-zero wearable_feats arrays and downstream NaN at training
# time. This bug cost ~2h of compute on the first tokenization pass.
cohort["fitbit_start"] = pd.to_datetime(cohort["fitbit_start"]).dt.normalize()
cohort["fitbit_end"] = pd.to_datetime(cohort["fitbit_end"]).dt.normalize()
# Require >= 180 + 30 = 210 days of observation
cohort["observation_days"] = (cohort["fitbit_end"] - cohort["fitbit_start"]).dt.days
cohort = cohort[cohort["observation_days"] >= N_DAYS_INPUT + N_DAYS_HORIZON].reset_index(drop=True)
print(f"cohort after v2 length filter: {len(cohort):,} (was 12,453 in v1)", flush=True)

with open(DATA_DIR / "splits.json") as f:
    splits = json.load(f)
# Re-filter splits to participants surviving the length filter
valid_ids = set(cohort["person_id"].tolist())
splits = {k: [p for p in v if p in valid_ids] for k, v in splits.items()}
print({k: len(v) for k, v in splits.items()})

with open(DATA_DIR / "vocab.json") as f:
    VOCAB = json.load(f)
# id -> token type, by name prefix
PREFIX_TO_TYPE = {"DX10": 4, "MED": 5, "PX10": 6, "LAB": 7}
ID_TO_TYPE: dict[int, int] = {}
for name, tid in VOCAB.items():
    for prefix, ttype in PREFIX_TO_TYPE.items():
        if name.startswith(prefix + ":"):
            ID_TO_TYPE[tid] = ttype
            break

with open(DATA_DIR / "endpoint_concept_ids.json") as f:
    ENDPOINT_IDS = json.load(f)
# Build inverse: SNOMED id -> endpoint name
CID_TO_ENDPOINT = {cid: name for name, cids in ENDPOINT_IDS.items() for cid in cids}


# ============================================================
# CELL 3 — confounder reference (computed on TRAIN)
# ============================================================
# TODO: implement BQ queries for baseline confounders.
# For now: pre-compute on TRAIN cohort, save to confounder_norm.json,
# and reuse stats when encoding VAL/TEST. Skeleton:
#
#   train_pids = splits["train"]
#   demos = bq.query(f"""
#     SELECT p.person_id, p.year_of_birth, p.gender_concept_id, m.value_as_number AS bmi
#     FROM `{CDR}.person` p
#     LEFT JOIN `{CDR}.measurement` m ON m.person_id = p.person_id AND m.measurement_concept_id = <BMI>
#     WHERE p.person_id IN UNNEST({train_pids})
#   """).to_dataframe()
#   ... derive smoking, CAD, cancer, SBP from observation / condition / measurement
#   train_mean = ...; train_std = ...
#   json.dump({"age": [m,s], "bmi": [m,s], "sbp": [m,s]}, open(DATA_DIR / "confounder_norm.json", "w"))

def load_or_compute_confounder_norm() -> dict:
    p = DATA_DIR / "confounder_norm.json"
    if p.exists():
        return json.load(open(p))
    # First-run path: compute on train ids using AoU person + measurement tables.
    # Until that BQ is filled in, fall back to neutral z-score values.
    print("WARNING: confounder_norm.json missing; using neutral z-score (mean=0, std=1). "
          "Implement compute_confounder_norm() before training!", flush=True)
    return {"age": [60.0, 10.0], "bmi": [28.0, 5.0], "sbp": [130.0, 15.0]}

CONF_NORM = load_or_compute_confounder_norm()


# ============================================================
# CELL 4 — bulk-fetch helpers (one chunk = ~200 participants)
# ============================================================

def _bq_to_df(sql: str) -> pd.DataFrame:
    """Run BQ query and return dataframe; small wrapper."""
    return bq.query(sql).to_dataframe()


def fetch_steps_daily(pids: list[int]) -> pd.DataFrame:
    """Aggregate steps_intraday into daily totals.

    AoU CDR v8 has NO `activity_summary` table — only `steps_intraday` (row_id,
    datetime DATETIME, steps NUMERIC, person_id, src_id). Sum across the day.
    """
    sql = f"""
    WITH per_minute AS (
      SELECT person_id, datetime,
             MAX(CAST(steps AS INT64)) AS steps   -- dedupe over src_id
      FROM `{CDR}.steps_intraday`
      WHERE person_id IN UNNEST({pids})
        AND steps IS NOT NULL
      GROUP BY person_id, datetime
    )
    SELECT person_id, DATE(datetime) AS d, SUM(steps) AS steps
    FROM per_minute
    GROUP BY person_id, DATE(datetime)
    """
    df = _bq_to_df(sql)
    df["d"] = pd.to_datetime(df["d"]).dt.normalize()
    return df


def fetch_hr_daily(pids: list[int]) -> pd.DataFrame:
    """Aggregate minute-HR into daily mean/resting/max + SDANN proxy.

    AoU CDR v8 `heart_rate_minute_level` columns: datetime DATETIME, heart_rate_value INT64.
    Use DATETIME_TRUNC (not TIMESTAMP_TRUNC) because the column is DATETIME.
    """
    sql = f"""
    WITH min1 AS (
      SELECT person_id,
             DATETIME_TRUNC(datetime, MINUTE) AS m,
             AVG(heart_rate_value) AS hr
      FROM `{CDR}.heart_rate_minute_level`
      WHERE person_id IN UNNEST({pids})
        AND heart_rate_value IS NOT NULL
      GROUP BY person_id, m
    )
    SELECT person_id,
           DATE(m) AS d,
           AVG(hr)                                              AS mean_hr,
           APPROX_QUANTILES(hr, 100)[OFFSET(10)]                AS resting_hr,
           APPROX_QUANTILES(hr, 100)[OFFSET(95)]                AS max_hr,
           STDDEV(hr)                                           AS sdann   -- daily SD of per-minute mean HR
    FROM min1
    GROUP BY person_id, d
    """
    df = _bq_to_df(sql)
    df["d"] = pd.to_datetime(df["d"]).dt.normalize()
    return df


def fetch_sleep_daily(pids: list[int]) -> pd.DataFrame:
    """Daily sleep summary: total sleep + REM/deep/light pct + onset hour.

    AoU CDR v8 `sleep_daily_summary` real columns (verified): minute_asleep,
    minute_rem, minute_deep, minute_light, minute_wake, minute_awake,
    minute_restless, minute_in_bed, minute_after_wakeup, is_main_sleep STRING,
    sleep_date DATE. NO sleep_start_datetime — onset is derived from
    `sleep_level` (MIN(start_datetime) per person/date where is_main_sleep='true').
    """
    # NOTE: minute_asleep in AoU's sleep_daily_summary is broken for many
    # users (smoke test 2026-06-09: ≈1 min for some records while stage
    # minutes rem+deep+light totaled ~7h). Use stage sum as total — robust
    # against the bad field and tighter to actual sleep architecture.
    sql = f"""
    WITH onset AS (
      SELECT person_id, sleep_date,
             MIN(start_datetime) AS first_start
      FROM `{CDR}.sleep_level`
      WHERE person_id IN UNNEST({pids})
        AND is_main_sleep = 'true'
      GROUP BY person_id, sleep_date
    ),
    daily AS (
      SELECT s.person_id, s.sleep_date,
             MAX(s.minute_rem)    AS minute_rem,
             MAX(s.minute_deep)   AS minute_deep,
             MAX(s.minute_light)  AS minute_light
      FROM `{CDR}.sleep_daily_summary` s
      WHERE s.person_id IN UNNEST({pids})
        AND s.is_main_sleep = 'true'
      GROUP BY s.person_id, s.sleep_date
    )
    SELECT d.person_id,
           d.sleep_date AS d,
           (COALESCE(d.minute_rem,0) + COALESCE(d.minute_deep,0)
              + COALESCE(d.minute_light,0)) / 60.0                AS sleep_duration_hr,
           SAFE_DIVIDE(d.minute_rem,
                       COALESCE(d.minute_rem,0)+COALESCE(d.minute_deep,0)+COALESCE(d.minute_light,0))   AS rem_pct,
           SAFE_DIVIDE(d.minute_deep,
                       COALESCE(d.minute_rem,0)+COALESCE(d.minute_deep,0)+COALESCE(d.minute_light,0))   AS deep_pct,
           SAFE_DIVIDE(d.minute_light,
                       COALESCE(d.minute_rem,0)+COALESCE(d.minute_deep,0)+COALESCE(d.minute_light,0))   AS light_pct,
           EXTRACT(HOUR   FROM o.first_start) +
             EXTRACT(MINUTE FROM o.first_start) / 60.0             AS sleep_onset_hour
    FROM daily d
    LEFT JOIN onset o
           ON o.person_id = d.person_id AND o.sleep_date = d.sleep_date
    WHERE COALESCE(d.minute_rem,0) + COALESCE(d.minute_deep,0)
            + COALESCE(d.minute_light,0) > 60   -- ≥1h of staged sleep
    """
    df = _bq_to_df(sql)
    df["d"] = pd.to_datetime(df["d"]).dt.normalize()
    return df


def fetch_ehr_events(pids: list[int]) -> pd.DataFrame:
    """All vocab-relevant EHR events with (person_id, event_date, token_id, token_type).

    We re-use v1's vocab.json. Map each row's source_concept_id -> token_name -> id.
    """
    # condition_occurrence (DX10)
    sql_cond = f"""
    SELECT co.person_id, co.condition_start_date AS d,
           c.concept_code AS code, 'DX10' AS prefix
    FROM `{CDR}.condition_occurrence` co
    JOIN `{CDR}.concept` c ON co.condition_source_concept_id = c.concept_id
    WHERE co.person_id IN UNNEST({pids})
      AND c.vocabulary_id IN ('ICD10CM', 'ICD10')
    """
    # drug_exposure (MED) -- use RxNorm / ATC mapping; for v2 we use ingredient name
    sql_drug = f"""
    SELECT de.person_id, de.drug_exposure_start_date AS d,
           c.concept_code AS code, 'MED' AS prefix
    FROM `{CDR}.drug_exposure` de
    JOIN `{CDR}.concept` c ON de.drug_source_concept_id = c.concept_id
    WHERE de.person_id IN UNNEST({pids})
    """
    sql_proc = f"""
    SELECT po.person_id, po.procedure_date AS d,
           c.concept_code AS code, 'PX10' AS prefix
    FROM `{CDR}.procedure_occurrence` po
    JOIN `{CDR}.concept` c ON po.procedure_source_concept_id = c.concept_id
    WHERE po.person_id IN UNNEST({pids})
      AND c.vocabulary_id IN ('ICD10PCS', 'CPT4')
    """
    sql_lab = f"""
    SELECT m.person_id, m.measurement_date AS d,
           c.concept_code AS code, 'LAB' AS prefix
    FROM `{CDR}.measurement` m
    JOIN `{CDR}.concept` c ON m.measurement_concept_id = c.concept_id
    WHERE m.person_id IN UNNEST({pids})
      AND c.vocabulary_id = 'LOINC'
    """
    df_cond = _bq_to_df(sql_cond)
    df_drug = _bq_to_df(sql_drug)
    df_proc = _bq_to_df(sql_proc)
    df_lab = _bq_to_df(sql_lab)
    ev = pd.concat([df_cond, df_drug, df_proc, df_lab], ignore_index=True)
    ev["d"] = pd.to_datetime(ev["d"]).dt.normalize()
    ev["token_name"] = ev["prefix"] + ":" + ev["code"].astype(str)
    ev["token_id"] = ev["token_name"].map(VOCAB)
    # Drop events whose code isn't in the v1 vocab (UNK) — model never sees them
    ev = ev.dropna(subset=["token_id"]).copy()
    ev["token_id"] = ev["token_id"].astype(np.int32)
    ev["token_type"] = ev["prefix"].map(PREFIX_TO_TYPE).astype(np.uint8)
    return ev[["person_id", "d", "token_id", "token_type"]]


def fetch_cardio_events(pids: list[int]) -> pd.DataFrame:
    """Per-endpoint dated events; used both for labels (future window) and baseline CAD flag."""
    all_ids = sorted({cid for ids in ENDPOINT_IDS.values() for cid in ids})
    sql = f"""
    SELECT co.person_id, co.condition_start_date AS d, co.condition_source_concept_id AS cid
    FROM `{CDR}.condition_occurrence` co
    WHERE co.person_id IN UNNEST({pids})
      AND co.condition_source_concept_id IN ({','.join(str(x) for x in all_ids)})
    """
    df = _bq_to_df(sql)
    df["d"] = pd.to_datetime(df["d"]).dt.normalize()
    df["endpoint"] = df["cid"].map(CID_TO_ENDPOINT)
    return df


def fetch_deaths(pids: list[int]) -> pd.DataFrame:
    """Death table (verified): person_id, death_date DATE, death_datetime,
    death_type_concept_id, cause_concept_id, cause_source_value,
    cause_source_concept_id.

    For v2 we treat ANY death within the 30-day horizon as cv_death=1 (proxy);
    a refined version would gate on cause_concept_id ∈ cardiovascular SNOMED set.
    """
    sql = f"""
    SELECT person_id, death_date AS d
    FROM `{CDR}.death`
    WHERE person_id IN UNNEST({pids})
      AND death_date IS NOT NULL
    """
    df = _bq_to_df(sql)
    df["d"] = pd.to_datetime(df["d"]).dt.normalize()
    return df


# ============================================================
# CELL 5 — per-participant window encoding
# ============================================================

def encode_window(
    pid: int,
    end_date: pd.Timestamp,
    steps_by_d: dict[pd.Timestamp, float],
    hr_by_d: dict[pd.Timestamp, tuple[float, float, float, float]],   # mean,resting,max,sdann
    sleep_by_d: dict[pd.Timestamp, tuple[float, float, float, float, float]],
    ehr_in_window: pd.DataFrame,
    cardio_in_horizon: pd.DataFrame,
    death_dates: list[pd.Timestamp],
    confounders: np.ndarray,
) -> dict:
    """Build a single window's tokenized row."""
    # Wearable feature matrix (180, 11)
    feats = np.zeros((N_DAYS_INPUT, 11), dtype=np.float32)
    mask = np.zeros(N_DAYS_INPUT, dtype=np.bool_)
    start = end_date - pd.Timedelta(days=N_DAYS_INPUT)
    for d_idx in range(N_DAYS_INPUT):
        day = start + pd.Timedelta(days=d_idx)
        any_data = False
        if day in steps_by_d:
            feats[d_idx, 0] = steps_by_d[day]
            any_data = True
        if day in hr_by_d:
            m, r, mx, sd = hr_by_d[day]
            feats[d_idx, 1:5] = (m, r, mx, sd)
            any_data = True
        if day in sleep_by_d:
            dur, rem, deep, light, onset = sleep_by_d[day]
            feats[d_idx, 5:10] = (rem, deep, light, dur, onset)
            any_data = True
        # 7-day sleep irregularity: SD of prior 7 days of sleep_duration
        if d_idx >= 7:
            window7 = feats[d_idx - 7:d_idx, 8]
            valid = window7[window7 > 0]
            if len(valid) >= 2:
                feats[d_idx, 10] = float(np.std(valid))
        mask[d_idx] = any_data

    # EHR events INSIDE the input window only.
    # The empty-DataFrame fallback path (no events for this pid) has object-dtype
    # "d", so (ew["d"] - start) raises TypeError on numpy scalars. Guard explicitly.
    if len(ehr_in_window) > 0:
        ew = ehr_in_window[
            (ehr_in_window["d"] >= start) & (ehr_in_window["d"] < end_date)
        ].copy()
    else:
        ew = ehr_in_window.iloc[0:0]
    if len(ew) > 0:
        day_idx = (pd.to_datetime(ew["d"]) - start).dt.days.astype(np.int16)
        ehr_token_ids = ew["token_id"].to_numpy().astype(np.int32)
        ehr_day_indices = day_idx.to_numpy()
        ehr_token_types = ew["token_type"].to_numpy().astype(np.uint8)
    else:
        ehr_token_ids = np.zeros(0, dtype=np.int32)
        ehr_day_indices = np.zeros(0, dtype=np.int16)
        ehr_token_types = np.zeros(0, dtype=np.uint8)

    # Labels: any cardio event of each type in (end_date, end_date + 30d]
    horizon_end = end_date + pd.Timedelta(days=N_DAYS_HORIZON)
    in_horizon = cardio_in_horizon[
        (cardio_in_horizon["d"] > end_date) & (cardio_in_horizon["d"] <= horizon_end)
    ]
    labels = {ep: int((in_horizon["endpoint"] == ep).any())
              for ep in ("afib", "mi", "hf")}
    labels["cv_death"] = int(any(
        end_date < d <= horizon_end for d in death_dates
    ))
    labels["composite"] = int(any(labels[ep] for ep in ("afib", "mi", "hf", "cv_death")))

    return {
        "person_id": pid,
        "end_date": end_date.date(),
        "wearable_feats": feats.tobytes(),
        "wearable_mask": mask.tobytes(),
        "ehr_token_ids": ehr_token_ids.tobytes(),
        "ehr_day_indices": ehr_day_indices.tobytes(),
        "ehr_token_types": ehr_token_types.tobytes(),
        "confounders": confounders.astype(np.float32).tobytes(),
        **{f"label_{k}": int(v) for k, v in labels.items()},
    }


# ============================================================
# CELL 6 — split-level driver
# ============================================================

def tokenize_split(split: str, stride: int) -> None:
    pids = splits[split]
    out_rows: list[dict] = []
    shard_idx = 0

    def flush() -> None:
        nonlocal shard_idx
        if not out_rows:
            return
        df = pd.DataFrame(out_rows)
        path = OUT_DIR / f"{split}_{shard_idx:04d}.parquet"
        df.to_parquet(path, compression="snappy")
        print(f"  wrote {path.name}  rows={len(df)}", flush=True)
        out_rows.clear()
        shard_idx += 1

    n_chunks = math.ceil(len(pids) / CHUNK_SIZE)
    print(f"\n=== {split}: {len(pids):,} participants, {n_chunks} chunks, stride={stride}d ===", flush=True)

    for ci, i in enumerate(range(0, len(pids), CHUNK_SIZE)):
        chunk = pids[i:i + CHUNK_SIZE]
        print(f"chunk {ci+1}/{n_chunks} ({len(chunk)} ppts) — fetching...", flush=True)
        steps_df = fetch_steps_daily(chunk)
        hr_df = fetch_hr_daily(chunk)
        sleep_df = fetch_sleep_daily(chunk)
        ehr_df = fetch_ehr_events(chunk)
        cardio_df = fetch_cardio_events(chunk)
        death_df = fetch_deaths(chunk)
        death_by = {pid: sorted(g["d"].tolist()) for pid, g in death_df.groupby("person_id")}
        # Diagnostics — silent SQL failures cost hours. Print row counts every chunk.
        print(
            f"    steps={len(steps_df):>7,d}  hr={len(hr_df):>7,d}  sleep={len(sleep_df):>6,d}"
            f"  ehr={len(ehr_df):>7,d}  cardio={len(cardio_df):>5,d}  death={len(death_df):>3,d}",
            flush=True,
        )
        # Per-(person,date) uniqueness must hold post-aggregation.
        assert sleep_df.groupby(["person_id", "d"]).size().max() <= 1 if len(sleep_df) else True
        assert steps_df.groupby(["person_id", "d"]).size().max() <= 1 if len(steps_df) else True
        assert hr_df.groupby(["person_id", "d"]).size().max() <= 1 if len(hr_df) else True

        # group dicts for fast lookup
        steps_by = steps_df.groupby("person_id").apply(
            lambda g: dict(zip(g["d"], g["steps"]))).to_dict()
        hr_by = hr_df.groupby("person_id").apply(
            lambda g: dict(zip(g["d"],
                               zip(g["mean_hr"], g["resting_hr"], g["max_hr"], g["sdann"])))).to_dict()
        sleep_by = sleep_df.groupby("person_id").apply(
            lambda g: dict(zip(g["d"],
                               zip(g["sleep_duration_hr"], g["rem_pct"], g["deep_pct"],
                                   g["light_pct"], g["sleep_onset_hour"])))).to_dict()
        ehr_by = ehr_df.groupby("person_id")
        cardio_by = cardio_df.groupby("person_id")

        # Per-participant window iteration
        for pid in chunk:
            pcohort = cohort[cohort["person_id"] == pid].iloc[0]
            fb_start, fb_end = pcohort["fitbit_start"], pcohort["fitbit_end"]
            confounders = build_confounders(pid)
            steps_d = steps_by.get(pid, {})
            hr_d = hr_by.get(pid, {})
            sleep_d = sleep_by.get(pid, {})
            ehr_pid = ehr_by.get_group(pid) if pid in ehr_by.groups else \
                      pd.DataFrame(columns=["person_id", "d", "token_id", "token_type"])
            cardio_pid = cardio_by.get_group(pid) if pid in cardio_by.groups else \
                         pd.DataFrame(columns=["person_id", "d", "cid", "endpoint"])

            death_dates = death_by.get(pid, [])
            end_d = fb_start + pd.Timedelta(days=N_DAYS_INPUT)
            last_valid_end = fb_end - pd.Timedelta(days=N_DAYS_HORIZON)
            while end_d <= last_valid_end:
                out_rows.append(encode_window(
                    pid, end_d, steps_d, hr_d, sleep_d, ehr_pid, cardio_pid,
                    death_dates, confounders,
                ))
                if len(out_rows) >= SHARD_SIZE:
                    flush()
                end_d = end_d + pd.Timedelta(days=stride)
    flush()


_CONFOUNDER_CACHE: dict[int, np.ndarray] = {}


# OMOP/AoU concept IDs used for v2 baseline confounders. All standard concepts
# (measurement.measurement_concept_id is OMOP-standard, NOT ICD/LOINC source).
BMI_CONCEPT_ID = 3038553           # LOINC 39156-5  Body mass index
SBP_CONCEPT_ID = 3004249           # LOINC 8480-6   Systolic blood pressure
DBP_CONCEPT_ID = 3012888           # LOINC 8462-4   Diastolic blood pressure
SEX_FEMALE_CONCEPT_ID = 45878463   # AoU sex_at_birth = "Female"


def _baseline_date_per_person() -> pd.Series:
    """Per-person fitbit_start: the look-back cutoff for baseline confounders."""
    return cohort.set_index("person_id")["fitbit_start"]


def _fetch_measurement_most_recent_before(
    pids: list[int], concept_id: int,
) -> pd.DataFrame:
    """Most recent value_as_number for one concept, taken BEFORE fitbit_start."""
    sql = f"""
    WITH baseline AS (
      SELECT person_id, fitbit_start FROM UNNEST([
        {",".join(f"STRUCT({pid} AS person_id, DATE('{bd.date()}') AS fitbit_start)"
                  for pid, bd in _baseline_date_per_person().loc[pids].items())}
      ])
    )
    SELECT m.person_id,
           ARRAY_AGG(m.value_as_number ORDER BY m.measurement_date DESC LIMIT 1)[OFFSET(0)] AS val
    FROM `{CDR}.measurement` m
    JOIN baseline b ON b.person_id = m.person_id
    WHERE m.measurement_concept_id = {concept_id}
      AND m.value_as_number IS NOT NULL
      AND m.measurement_date < b.fitbit_start
    GROUP BY m.person_id
    """
    return _bq_to_df(sql)


def _fetch_condition_ever_before(
    pids: list[int], concept_ids: list[int],
) -> pd.DataFrame:
    """1 if person had any of these source-concept conditions before fitbit_start."""
    if not concept_ids:
        return pd.DataFrame({"person_id": [], "has_cond": []})
    sql = f"""
    WITH baseline AS (
      SELECT person_id, fitbit_start FROM UNNEST([
        {",".join(f"STRUCT({pid} AS person_id, DATE('{bd.date()}') AS fitbit_start)"
                  for pid, bd in _baseline_date_per_person().loc[pids].items())}
      ])
    )
    SELECT DISTINCT b.person_id, 1 AS has_cond
    FROM `{CDR}.condition_occurrence` co
    JOIN baseline b ON b.person_id = co.person_id
    WHERE co.condition_source_concept_id IN ({",".join(str(c) for c in concept_ids)})
      AND co.condition_start_date < b.fitbit_start
    """
    return _bq_to_df(sql)


def precompute_confounders() -> None:
    """One-shot precompute of per-person baseline confounders.

    8-dim vector (NO z-scoring — model learns scaling via input projection):
       0: age_years     (year of fitbit_start - year_of_birth)
       1: sex_female    (0/1 from sex_at_birth_concept_id)
       2: bmi           (most recent before fitbit_start; NaN -> 0 with mask via output)
       3: sbp_mmHg      (most recent systolic before baseline)
       4: dbp_mmHg      (most recent diastolic before baseline)
       5: baseline_cad  (0/1; ever before baseline)
       6: baseline_cancer (0/1)
       7: prior_afib    (0/1; AFib already diagnosed before baseline)

    Cached at DATA_DIR/confounders_per_person.parquet.
    """
    cache = DATA_DIR / "confounders_per_person.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        for r in df.itertuples(index=False):
            _CONFOUNDER_CACHE[r.person_id] = np.array(
                [r.age, r.sex_female, r.bmi, r.sbp, r.dbp,
                 r.baseline_cad, r.baseline_cancer, r.prior_afib],
                dtype=np.float32,
            )
        print(f"loaded {len(df):,} confounder rows from cache", flush=True)
        return

    all_pids = cohort["person_id"].tolist()
    base = cohort.set_index("person_id")[["fitbit_start"]].copy()

    # 1) demographics — single query, all pids at once
    print("confounder fetch: demographics...", flush=True)
    demo = _bq_to_df(f"""
      SELECT person_id, year_of_birth, sex_at_birth_concept_id
      FROM `{CDR}.person`
      WHERE person_id IN UNNEST({all_pids})
    """).set_index("person_id")

    rows: list[dict] = []
    # Process in chunks (the STRUCT-UNNEST per-chunk SQL has a max literal size)
    CONF_CHUNK = 500
    n_cc = math.ceil(len(all_pids) / CONF_CHUNK)
    for ci, i in enumerate(range(0, len(all_pids), CONF_CHUNK)):
        chunk = all_pids[i:i + CONF_CHUNK]
        print(f"  confounder chunk {ci+1}/{n_cc}", flush=True)

        bmi = _fetch_measurement_most_recent_before(chunk, BMI_CONCEPT_ID).set_index("person_id")
        sbp = _fetch_measurement_most_recent_before(chunk, SBP_CONCEPT_ID).set_index("person_id")
        dbp = _fetch_measurement_most_recent_before(chunk, DBP_CONCEPT_ID).set_index("person_id")
        cad = _fetch_condition_ever_before(chunk, ENDPOINT_IDS.get("cad_baseline", [])).set_index("person_id")
        cancer = _fetch_condition_ever_before(chunk, ENDPOINT_IDS.get("cancer_baseline", [])).set_index("person_id")
        prior_afib = _fetch_condition_ever_before(chunk, ENDPOINT_IDS.get("afib", [])).set_index("person_id")

        for pid in chunk:
            d = demo.loc[pid] if pid in demo.index else None
            yob = d["year_of_birth"] if d is not None else 1960
            sex_cid = d["sex_at_birth_concept_id"] if d is not None else 0
            fb_start = base.loc[pid, "fitbit_start"]
            age = float(fb_start.year - int(yob))
            rows.append({
                "person_id": pid,
                "age": age,
                "sex_female": 1.0 if int(sex_cid) == SEX_FEMALE_CONCEPT_ID else 0.0,
                "bmi": float(bmi.loc[pid, "val"]) if pid in bmi.index else 0.0,
                "sbp": float(sbp.loc[pid, "val"]) if pid in sbp.index else 0.0,
                "dbp": float(dbp.loc[pid, "val"]) if pid in dbp.index else 0.0,
                "baseline_cad": 1.0 if pid in cad.index else 0.0,
                "baseline_cancer": 1.0 if pid in cancer.index else 0.0,
                "prior_afib": 1.0 if pid in prior_afib.index else 0.0,
            })

    df = pd.DataFrame(rows)
    df.to_parquet(cache, compression="snappy")
    print(f"wrote {cache.name}  rows={len(df)}", flush=True)

    for r in df.itertuples(index=False):
        _CONFOUNDER_CACHE[r.person_id] = np.array(
            [r.age, r.sex_female, r.bmi, r.sbp, r.dbp,
             r.baseline_cad, r.baseline_cancer, r.prior_afib],
            dtype=np.float32,
        )


def build_confounders(pid: int) -> np.ndarray:
    """Look up precomputed 8-dim confounder vector for one person.

    precompute_confounders() must be called once before tokenize_split().
    Missing pid returns zeros with a warning (rare; would mean the person was
    in cohort but had no person table row — shouldn't happen).
    """
    v = _CONFOUNDER_CACHE.get(pid)
    if v is None:
        return np.zeros(8, dtype=np.float32)
    return v


# ============================================================
# CELL 7 — run
# ============================================================

if __name__ == "__main__":
    precompute_confounders()
    tokenize_split("train", TRAIN_STRIDE)
    tokenize_split("val", EVAL_STRIDE)
    tokenize_split("test", EVAL_STRIDE)
    print("done", flush=True)
