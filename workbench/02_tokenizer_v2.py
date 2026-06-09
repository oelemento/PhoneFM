"""PhoneFM v2 tokenizer — daily-aggregate wearable + EHR events + per-endpoint labels.

Designed to be run on Workbench Jupyter (CPU machine is fine; it's BQ + pandas bound).

INPUTS (assumed to be on the bucket FUSE mount):
  /home/jupyter/workspace/phonefm-data/cohort_base.parquet         (12,453 participants, fitbit_start/end)
  /home/jupyter/workspace/phonefm-data/splits.json                 (train/val/test person_id lists)
  /home/jupyter/workspace/phonefm-data/endpoint_concept_ids.json   (afib/mi/hf SNOMED ids)
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

CONFOUNDERS (per participant, shape [8]):
  age (years, z-scored using train mean/std written to confounder_norm.json)
  sex_female (0/1)
  bmi (z-scored)
  baseline_cad (0/1; first CAD dx before window start)
  baseline_cancer (0/1)
  smoking_100cigs (0/1; from observation table)
  alcohol_ever (0/1)
  sbp_systolic_mmHg (z-scored)

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
cohort["fitbit_start"] = pd.to_datetime(cohort["fitbit_start"])
cohort["fitbit_end"] = pd.to_datetime(cohort["fitbit_end"])
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
    # NOTE: AoU has `activity_summary` with daily steps; verify column names.
    sql = f"""
    SELECT person_id, activity_date AS d, steps
    FROM `{CDR}.activity_summary`
    WHERE person_id IN UNNEST({pids})
      AND steps IS NOT NULL
    """
    df = _bq_to_df(sql)
    df["d"] = pd.to_datetime(df["d"]).dt.normalize()
    return df


def fetch_hr_daily(pids: list[int]) -> pd.DataFrame:
    """Aggregate minute-HR into daily mean/resting/max + SDANN proxy."""
    sql = f"""
    WITH min5 AS (
      SELECT person_id,
             TIMESTAMP_TRUNC(datetime, MINUTE) AS m,
             AVG(heart_rate_value) AS hr
      FROM `{CDR}.heart_rate_minute_level`
      WHERE person_id IN UNNEST({pids})
      GROUP BY person_id, m
    )
    SELECT person_id,
           DATE(m) AS d,
           AVG(hr)                                              AS mean_hr,
           APPROX_QUANTILES(hr, 100)[OFFSET(10)]                AS resting_hr,
           APPROX_QUANTILES(hr, 100)[OFFSET(95)]                AS max_hr,
           STDDEV(hr)                                           AS sdann   -- daily SD of 5-min mean HR
    FROM min5
    GROUP BY person_id, d
    """
    df = _bq_to_df(sql)
    df["d"] = pd.to_datetime(df["d"]).dt.normalize()
    return df


def fetch_sleep_daily(pids: list[int]) -> pd.DataFrame:
    """Daily sleep summary: total sleep + REM/deep/light pct + onset hour.

    Zheng 2024 used Fitbit's proprietary labels. Verify table + column names.
    """
    sql = f"""
    SELECT person_id,
           sleep_date AS d,
           total_sleep_minutes / 60.0                AS sleep_duration_hr,
           rem_minutes / NULLIF(total_sleep_minutes, 0)  AS rem_pct,
           deep_minutes / NULLIF(total_sleep_minutes, 0) AS deep_pct,
           light_minutes / NULLIF(total_sleep_minutes, 0) AS light_pct,
           EXTRACT(HOUR FROM sleep_start_datetime) +
             EXTRACT(MINUTE FROM sleep_start_datetime) / 60.0 AS sleep_onset_hour
    FROM `{CDR}.sleep_daily_summary`
    WHERE person_id IN UNNEST({pids})
      AND is_main_sleep = TRUE
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

    # EHR events INSIDE the input window only
    ew = ehr_in_window[(ehr_in_window["d"] >= start) & (ehr_in_window["d"] < end_date)].copy()
    ew["day_idx"] = ((ew["d"] - start).dt.days).astype(np.int16)
    ehr_token_ids = ew["token_id"].values.astype(np.int32)
    ehr_day_indices = ew["day_idx"].values.astype(np.int16)
    ehr_token_types = ew["token_type"].values.astype(np.uint8)

    # Labels: any cardio event of each type in (end_date, end_date + 30d]
    horizon_end = end_date + pd.Timedelta(days=N_DAYS_HORIZON)
    in_horizon = cardio_in_horizon[
        (cardio_in_horizon["d"] > end_date) & (cardio_in_horizon["d"] <= horizon_end)
    ]
    labels = {ep: int((in_horizon["endpoint"] == ep).any())
              for ep in ("afib", "mi", "hf")}
    labels["cv_death"] = 0   # TODO: pull from death table
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

            end_d = fb_start + pd.Timedelta(days=N_DAYS_INPUT)
            last_valid_end = fb_end - pd.Timedelta(days=N_DAYS_HORIZON)
            while end_d <= last_valid_end:
                out_rows.append(encode_window(
                    pid, end_d, steps_d, hr_d, sleep_d, ehr_pid, cardio_pid, confounders
                ))
                if len(out_rows) >= SHARD_SIZE:
                    flush()
                end_d = end_d + pd.Timedelta(days=stride)
    flush()


def build_confounders(pid: int) -> np.ndarray:
    """TODO: per-participant confounders from AoU person + measurement tables.

    Until implemented, returns zeros. Critical: complete this before training v2,
    otherwise the CLS token sees no patient-level baseline signal.
    """
    return np.zeros(8, dtype=np.float32)


# ============================================================
# CELL 7 — run
# ============================================================

if __name__ == "__main__":
    tokenize_split("train", TRAIN_STRIDE)
    tokenize_split("val", EVAL_STRIDE)
    tokenize_split("test", EVAL_STRIDE)
    print("done", flush=True)
