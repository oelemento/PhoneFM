"""PhoneFM v3 tokenizer — multi-horizon labels + per-endpoint baseline masks + neg controls.

Fork of 02_tokenizer_v2.py per docs/v3_spec.md §5.2. The wearable + EHR fetch
machinery (steps, HR, sleep, EHR events) is byte-identical to v2; the diffs
versus v2 live in:

  - Cohort filter: observation_days >= 545 (was 210) — preserves 365d horizon
  - endpoint_concept_ids_v3.json with multi-source codesets (Dx + Lab + Rx)
  - fetch_endpoint_events: union over all endpoints + dx/lab/drug evidence
  - precompute_confounders_v3: 9-dim vector adding baseline_osa
  - encode_window_v3: 13 (label, mask) pairs per window
  - Subgroup metadata columns (race, sex, age, ses) added to each row
  - Sanity assertion on first written shard (v2 lesson §14.5)

INPUTS:
  /home/jupyter/workspace/phonefm-data/cohort_base.parquet
  /home/jupyter/workspace/phonefm-data/splits.json
  /home/jupyter/workspace/phonefm-data/endpoint_concept_ids_v3.json    <-- NEW (from build_endpoint_concept_ids_v3.py)
  /home/jupyter/workspace/phonefm-data/vocab.json

OUTPUTS:
  /home/jupyter/workspace/phonefm-data/tokenized_v3/{train,val,test}_NNNN.parquet
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-not-found]
from google.cloud import bigquery  # type: ignore[import-not-found]

# ============================================================
# Config
# ============================================================

bq = bigquery.Client()
CDR = os.environ["WORKSPACE_CDR"]
print(f"CDR = {CDR}", flush=True)

DATA_DIR = Path("/home/jupyter/workspace/phonefm-data")
OUT_DIR = DATA_DIR / "tokenized_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Window / horizon — v3 spec §3, §13
N_DAYS_INPUT = 180
HORIZONS = (30, 180, 365)                # all horizons we may emit
N_DAYS_HORIZON_MAX = max(HORIZONS)       # cohort filter uses this
TRAIN_STRIDE = 14
EVAL_STRIDE = 30
SHARD_SIZE = 5000
CHUNK_SIZE = 200  # BQ chunk size — see v2 lesson on to_dataframe peak memory

# Head specs — locked to match phonefm_model_v3.ALL_HEADS
# (endpoint, horizon_days) tuples in canonical order
PRIMARY_HEADS: tuple[tuple[str, int], ...] = (
    ("mortality", 30),
    ("mortality", 180),
    ("mortality", 365),
    ("t2d", 180),
    ("t2d", 365),
    ("dep", 180),
    ("dep", 365),
    ("cv_composite", 30),
    ("cv_composite", 180),
    ("cv_composite", 365),
)
NEGATIVE_CONTROL_HEADS: tuple[tuple[str, int], ...] = (
    ("skin_neoplasm", 365),
    ("refractive_errors", 365),
    ("dental_caries", 365),
)
ALL_HEADS = PRIMARY_HEADS + NEGATIVE_CONTROL_HEADS

# Endpoints that get a baseline-exclusion mask (mask=0 if prior diagnosis)
BASELINE_EXCLUDED = frozenset({"t2d", "dep", "skin_neoplasm", "refractive_errors", "dental_caries"})


# ============================================================
# Cohort + splits + vocab + endpoint codesets
# ============================================================

cohort = pd.read_parquet(DATA_DIR / "cohort_base.parquet")
cohort["fitbit_start"] = pd.to_datetime(cohort["fitbit_start"]).dt.normalize()
cohort["fitbit_end"] = pd.to_datetime(cohort["fitbit_end"]).dt.normalize()
cohort["observation_days"] = (cohort["fitbit_end"] - cohort["fitbit_start"]).dt.days
n_pre = len(cohort)
cohort = cohort[cohort["observation_days"] >= N_DAYS_INPUT + N_DAYS_HORIZON_MAX].reset_index(drop=True)
print(f"cohort after v3 length filter (>= {N_DAYS_INPUT + N_DAYS_HORIZON_MAX}d): "
      f"{len(cohort):,} ({100*len(cohort)/max(n_pre,1):.1f}% of {n_pre:,})", flush=True)

with open(DATA_DIR / "splits.json") as f:
    splits = json.load(f)
valid_ids = set(cohort["person_id"].tolist())
splits = {k: [p for p in v if p in valid_ids] for k, v in splits.items()}
print({k: len(v) for k, v in splits.items()})

with open(DATA_DIR / "vocab.json") as f:
    VOCAB = json.load(f)
PREFIX_TO_TYPE = {"DX10": 4, "MED": 5, "PX10": 6, "LAB": 7}
ID_TO_TYPE: dict[int, int] = {}
for name, tid in VOCAB.items():
    for prefix, ttype in PREFIX_TO_TYPE.items():
        if name.startswith(prefix + ":"):
            ID_TO_TYPE[tid] = ttype
            break

# Multi-source endpoint codesets (from build_endpoint_concept_ids_v3.py)
EP_JSON_PATH = DATA_DIR / "endpoint_concept_ids_v3.json"
if not EP_JSON_PATH.exists():
    raise FileNotFoundError(
        f"{EP_JSON_PATH} not found. Run `python3 build_endpoint_concept_ids_v3.py` first."
    )
with open(EP_JSON_PATH) as f:
    EP_CODES = json.load(f)


def _flatten_endpoint_dx(ep: str) -> list[int]:
    """Return all 'dx*' concept_ids for an endpoint in EP_CODES['endpoints'][ep]."""
    block = EP_CODES["endpoints"].get(ep, {})
    out: list[int] = []
    for key, ids in block.items():
        if isinstance(ids, list) and (key == "dx" or key.startswith("dx_")):
            out.extend(int(x) for x in ids)
    return sorted(set(out))


# NOTE: Baseline exclusion at encode time uses the `endpoint_pid` event frame
# (any prior dx/lab/drug evidence for the endpoint), not a separate baseline_dx
# codeset lookup — see encode_window_v3 below. The baseline_codes JSON block is
# still used by precompute_confounders_v3 for the OSA/CAD/cancer/AFib flags.


# ============================================================
# BQ helpers
# ============================================================

def _bq_to_df(sql: str) -> pd.DataFrame:
    return bq.query(sql).to_dataframe()


def _bq_in_list(ids: list[int]) -> str:
    """Render an IN-list literal. Empty list yields '(NULL)' which never matches."""
    if not ids:
        return "(NULL)"
    return "(" + ",".join(str(int(x)) for x in ids) + ")"


# ---- Wearable + EHR fetchers — PORTED VERBATIM from v2 (working against AoU CDR v8).
# Initial v3 draft of these used wrong column names + a BOOL comparison against
# the STRING `is_main_sleep` column and joined on raw source_value, all of which
# crashed against the live CDR. v2's queries were verified during v2 tokenization,
# so v3 reuses them exactly.

def fetch_steps_daily(pids: list[int]) -> pd.DataFrame:
    """Aggregate steps_intraday into daily totals.

    AoU CDR v8 has NO `activity_summary` table — only `steps_intraday`. Sum
    across the day. Inner CTE dedupes over `src_id` to avoid double-counting
    when one minute has multiple source rows.
    """
    sql = f"""
    WITH per_minute AS (
      SELECT person_id, datetime,
             MAX(CAST(steps AS INT64)) AS steps
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

    `heart_rate_minute_level` column is DATETIME, so use DATETIME_TRUNC
    (not TIMESTAMP_TRUNC).
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
           AVG(hr) AS mean_hr,
           APPROX_QUANTILES(hr, 100)[OFFSET(10)] AS resting_hr,
           APPROX_QUANTILES(hr, 100)[OFFSET(95)] AS max_hr,
           STDDEV(hr) AS sdann
    FROM min1
    GROUP BY person_id, d
    """
    df = _bq_to_df(sql)
    df["d"] = pd.to_datetime(df["d"]).dt.normalize()
    return df


def fetch_sleep_daily(pids: list[int]) -> pd.DataFrame:
    """Daily sleep summary: total sleep + REM/deep/light pct + onset hour.

    AoU CDR v8 `sleep_daily_summary` columns (verified): minute_asleep,
    minute_rem, minute_deep, minute_light, ..., is_main_sleep STRING,
    sleep_date DATE. NO sleep_start_datetime — onset is derived from
    `sleep_level` (MIN(start_datetime) per person/date where is_main_sleep='true').

    minute_asleep is broken for many users (≈1 min while stage minutes sum to
    ~7h); use stage sum as total.
    """
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
              + COALESCE(d.minute_light,0)) / 60.0 AS sleep_duration_hr,
           SAFE_DIVIDE(d.minute_rem,
                       COALESCE(d.minute_rem,0)+COALESCE(d.minute_deep,0)+COALESCE(d.minute_light,0)) AS rem_pct,
           SAFE_DIVIDE(d.minute_deep,
                       COALESCE(d.minute_rem,0)+COALESCE(d.minute_deep,0)+COALESCE(d.minute_light,0)) AS deep_pct,
           SAFE_DIVIDE(d.minute_light,
                       COALESCE(d.minute_rem,0)+COALESCE(d.minute_deep,0)+COALESCE(d.minute_light,0)) AS light_pct,
           EXTRACT(HOUR FROM o.first_start)
             + EXTRACT(MINUTE FROM o.first_start) / 60.0 AS sleep_onset_hour
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

    Reuses v1's vocab.json. Map each row's source_concept_id → concept_code →
    token_name → id. NOTE: token_name keys use the ICD-10 / RxNorm / CPT
    concept_code from the concept table, NOT raw `*_source_value` (which is the
    EMR's free-text value and doesn't reliably match the vocab).
    """
    sql_cond = f"""
    SELECT co.person_id, co.condition_start_date AS d,
           c.concept_code AS code, 'DX10' AS prefix
    FROM `{CDR}.condition_occurrence` co
    JOIN `{CDR}.concept` c ON co.condition_source_concept_id = c.concept_id
    WHERE co.person_id IN UNNEST({pids})
      AND c.vocabulary_id IN ('ICD10CM', 'ICD10')
    """
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
    ev = ev.dropna(subset=["token_id"]).copy()
    ev["token_id"] = ev["token_id"].astype(np.int32)
    ev["token_type"] = ev["prefix"].map(PREFIX_TO_TYPE).astype(np.uint8)
    return ev[["person_id", "d", "token_id", "token_type"]]


# ---- Endpoint event fetching (v3-specific: multi-source)

def fetch_endpoint_events(pids: list[int]) -> pd.DataFrame:
    """Return one row per (person_id, date, endpoint, evidence_type).

    Each endpoint may have multiple evidence types (dx / lab / drug). The
    per-endpoint "first-recorded" date is the minimum of the union of these
    rows. The baseline-exclusion flag is "any row before fitbit_start".

    For lab evidence we ALSO enforce the threshold (HbA1c≥6.5, etc.) — that
    happens inline in this fetcher rather than at encoding time so the
    baseline/exclusion logic uses correctly-thresholded rows.
    """
    rows: list[pd.DataFrame] = []

    # ---- Dx evidence (condition_occurrence) — primary mechanism
    for ep in EP_CODES["endpoints"].keys():
        if ep == "mortality":
            continue
        ids = _flatten_endpoint_dx(ep)
        if not ids:
            continue
        sql = f"""
        SELECT person_id, condition_start_date AS d, '{ep}' AS endpoint, 'dx' AS evidence
        FROM `{CDR}.condition_occurrence`
        WHERE person_id IN UNNEST({pids})
          AND condition_concept_id IN {_bq_in_list(ids)}
        """
        rows.append(_bq_to_df(sql))

    # ---- Lab evidence (measurement) — T2D only in v3 (HbA1c / glucose)
    t2d_lab = EP_CODES["endpoints"].get("t2d", {}).get("lab", [])
    if t2d_lab:
        # threshold logic: HbA1c >= 6.5 OR fasting glucose >= 126 OR random glucose >= 200
        # We pull all hits with value_as_number then filter Python-side per concept
        sql = f"""
        SELECT person_id, measurement_date AS d, measurement_concept_id AS cid, value_as_number AS v
        FROM `{CDR}.measurement`
        WHERE person_id IN UNNEST({pids})
          AND measurement_concept_id IN {_bq_in_list(t2d_lab)}
          AND value_as_number IS NOT NULL
        """
        lab_df = _bq_to_df(sql)
        # Apply per-concept thresholds. The concept_ids we received from
        # build_endpoint_concept_ids_v3 are LOINC-derived OMOP-standard ids; we
        # match by LOINC code via the spec's documented thresholds. To stay
        # robust to AoU concept-id drift we look up the LOINC code per row.
        if len(lab_df):
            concept_codes = _bq_to_df(f"""
                SELECT concept_id, concept_code FROM `{CDR}.concept`
                WHERE concept_id IN {_bq_in_list(t2d_lab)}
            """).set_index("concept_id")["concept_code"].to_dict()
            lab_df["loinc"] = lab_df["cid"].map(concept_codes).fillna("")
            # threshold per LOINC
            def _is_t2d_hit(row):
                code, v = row["loinc"], row["v"]
                if code == "4548-4":  # HbA1c %
                    return v >= 6.5
                if code == "1558-6":  # fasting glucose
                    return v >= 126
                if code == "2345-7":  # random/serum glucose
                    return v >= 200
                return False
            lab_df = lab_df[lab_df.apply(_is_t2d_hit, axis=1)]
            lab_df = lab_df.assign(endpoint="t2d", evidence="lab")[
                ["person_id", "d", "endpoint", "evidence"]
            ]
            rows.append(lab_df)

    # ---- Drug evidence (drug_exposure) — T2D + depression
    for ep in ("t2d", "dep"):
        drug_ids = EP_CODES["endpoints"].get(ep, {}).get("drug", [])
        if not drug_ids:
            continue
        sql = f"""
        SELECT person_id, drug_exposure_start_date AS d, '{ep}' AS endpoint, 'drug' AS evidence
        FROM `{CDR}.drug_exposure`
        WHERE person_id IN UNNEST({pids})
          AND drug_concept_id IN {_bq_in_list(drug_ids)}
        """
        rows.append(_bq_to_df(sql))

    if not rows:
        return pd.DataFrame(columns=["person_id", "d", "endpoint", "evidence"])
    df = pd.concat(rows, ignore_index=True)
    df["d"] = pd.to_datetime(df["d"]).dt.normalize()
    return df


def fetch_deaths(pids: list[int]) -> pd.DataFrame:
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
# Confounders — 9-dim vector (v3 adds baseline_osa)
# ============================================================

BMI_CONCEPT_ID = 3038553
SBP_CONCEPT_ID = 3004249
DBP_CONCEPT_ID = 3012888
SEX_FEMALE_CONCEPT_ID = 45878463

_CONFOUNDER_CACHE: dict[int, np.ndarray] = {}


def _baseline_date_per_person() -> pd.Series:
    return cohort.set_index("person_id")["fitbit_start"]


def _fetch_measurement_most_recent_before(pids: list[int], concept_id: int) -> pd.DataFrame:
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


def _fetch_condition_ever_before(pids: list[int], concept_ids: list[int]) -> pd.DataFrame:
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
    WHERE co.condition_concept_id IN {_bq_in_list(concept_ids)}
      AND co.condition_start_date < b.fitbit_start
    """
    return _bq_to_df(sql)


def precompute_confounders_v3() -> None:
    """9-dim per-person baseline confounder vector for v3.

    Index map:
       0: age_years
       1: sex_female
       2: bmi
       3: sbp
       4: dbp
       5: baseline_cad
       6: baseline_cancer
       7: prior_afib
       8: baseline_osa     <-- NEW vs v2
    """
    cache = DATA_DIR / "confounders_per_person_v3.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        for r in df.itertuples(index=False):
            _CONFOUNDER_CACHE[r.person_id] = np.array(
                [r.age, r.sex_female, r.bmi, r.sbp, r.dbp,
                 r.baseline_cad, r.baseline_cancer, r.prior_afib, r.baseline_osa],
                dtype=np.float32,
            )
        print(f"loaded {len(df):,} v3 confounder rows from cache", flush=True)
        return

    all_pids = cohort["person_id"].tolist()
    base = cohort.set_index("person_id")[["fitbit_start"]].copy()

    print("v3 confounder fetch: demographics...", flush=True)
    demo = _bq_to_df(f"""
      SELECT person_id, year_of_birth, sex_at_birth_concept_id
      FROM `{CDR}.person`
      WHERE person_id IN UNNEST({all_pids})
    """).set_index("person_id")

    # Baseline codeset shortcuts
    cad_ids = EP_CODES["baseline_codes"].get("cad", {}).get("dx", [])
    cancer_ids = EP_CODES["baseline_codes"].get("cancer", {}).get("dx", [])
    afib_ids = EP_CODES["baseline_codes"].get("afib", {}).get("dx", [])
    osa_ids = EP_CODES["baseline_codes"].get("osa", {}).get("dx", [])

    rows: list[dict[str, Any]] = []
    CONF_CHUNK = 500
    n_cc = math.ceil(len(all_pids) / CONF_CHUNK)
    for ci, i in enumerate(range(0, len(all_pids), CONF_CHUNK)):
        chunk = all_pids[i:i + CONF_CHUNK]
        print(f"  confounder chunk {ci+1}/{n_cc}", flush=True)

        bmi = _fetch_measurement_most_recent_before(chunk, BMI_CONCEPT_ID).set_index("person_id")
        sbp = _fetch_measurement_most_recent_before(chunk, SBP_CONCEPT_ID).set_index("person_id")
        dbp = _fetch_measurement_most_recent_before(chunk, DBP_CONCEPT_ID).set_index("person_id")
        cad = _fetch_condition_ever_before(chunk, cad_ids).set_index("person_id")
        cancer = _fetch_condition_ever_before(chunk, cancer_ids).set_index("person_id")
        prior_afib = _fetch_condition_ever_before(chunk, afib_ids).set_index("person_id")
        osa = _fetch_condition_ever_before(chunk, osa_ids).set_index("person_id")

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
                "baseline_osa": 1.0 if pid in osa.index else 0.0,
            })

    df = pd.DataFrame(rows)
    df.to_parquet(cache, compression="snappy")
    print(f"wrote {cache.name}  rows={len(df)}", flush=True)
    for r in df.itertuples(index=False):
        _CONFOUNDER_CACHE[r.person_id] = np.array(
            [r.age, r.sex_female, r.bmi, r.sbp, r.dbp,
             r.baseline_cad, r.baseline_cancer, r.prior_afib, r.baseline_osa],
            dtype=np.float32,
        )


def build_confounders(pid: int) -> np.ndarray:
    v = _CONFOUNDER_CACHE.get(pid)
    if v is None:
        print(f"WARNING: no confounder vector for pid={pid}; using zeros", flush=True)
        return np.zeros(9, dtype=np.float32)
    return v


# ============================================================
# Subgroup metadata — pulled once for the full cohort
# ============================================================

_SUBGROUP_CACHE: dict[int, dict[str, int]] = {}
# Per-person actual birth date (pd.Timestamp at midnight) for true-age computation.
# Populated by precompute_subgroup_metadata alongside _SUBGROUP_CACHE. Falls back
# to year-of-birth-only when month/day are suppressed by AoU. None means "no
# birth date at all"; encode_window_v3 falls back to year-diff in that case.
_BIRTH_DATE_CACHE: dict[int, pd.Timestamp | None] = {}


def precompute_subgroup_metadata() -> None:
    """Per-person race/sex/SES + birth date (for accurate age) fixed once."""
    cache = DATA_DIR / "subgroup_metadata_v3.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        for r in df.itertuples(index=False):
            _SUBGROUP_CACHE[r.person_id] = {
                "race_concept_id": int(r.race_concept_id),
                "sex_at_birth_concept_id": int(r.sex_at_birth_concept_id),
                "ses_income_quartile": int(r.ses_income_quartile),
            }
            bd = getattr(r, "birth_date", None)
            _BIRTH_DATE_CACHE[r.person_id] = (
                pd.Timestamp(bd).normalize() if bd is not None and not pd.isna(bd) else None
            )
        print(f"loaded {len(df):,} subgroup metadata rows from cache", flush=True)
        return

    pids = cohort["person_id"].tolist()
    sql = f"""
      SELECT person_id,
             COALESCE(race_concept_id, 0) AS race_concept_id,
             COALESCE(sex_at_birth_concept_id, 0) AS sex_at_birth_concept_id,
             birth_datetime
      FROM `{CDR}.person`
      WHERE person_id IN UNNEST({pids})
    """
    df = _bq_to_df(sql)
    # SES income quartile: AoU's basics_survey has 'Income' answer concept IDs;
    # mapping to quartiles is documented in the AoU PPI codebook. For v3 we
    # default to 0 (missing) and refine in a follow-up pass if AoU exposes
    # income_concept_id directly.
    df["ses_income_quartile"] = 0
    # AoU often suppresses month/day on birth_datetime for rare birthdates and
    # leaves YYYY-01-01 as the default. We keep the value as-is; encode_window_v3
    # falls back to year-difference if the date arithmetic is suspicious.
    # AoU stores birth_datetime as TIMESTAMP (tz-aware UTC); strip tz so it can
    # be subtracted from `end_date` (tz-naive, derived from fb_start.normalize()).
    bd_series = pd.to_datetime(df["birth_datetime"], errors="coerce", utc=True)
    df["birth_date"] = bd_series.dt.tz_convert(None).dt.normalize()
    df = df.drop(columns=["birth_datetime"])
    df.to_parquet(cache, compression="snappy")
    for r in df.itertuples(index=False):
        _SUBGROUP_CACHE[r.person_id] = {
            "race_concept_id": int(r.race_concept_id),
            "sex_at_birth_concept_id": int(r.sex_at_birth_concept_id),
            "ses_income_quartile": int(r.ses_income_quartile),
        }
        bd = getattr(r, "birth_date", None)
        _BIRTH_DATE_CACHE[r.person_id] = (
            pd.Timestamp(bd).normalize() if bd is not None and not pd.isna(bd) else None
        )


# ============================================================
# Per-window encoding (v3-specific labels + masks)
# ============================================================

def _has_event_before(events_df: pd.DataFrame, before: pd.Timestamp,
                      endpoint: str) -> bool:
    """Any (dx | lab | drug) evidence for `endpoint` strictly before `before`."""
    if len(events_df) == 0:
        return False
    sel = (events_df["endpoint"] == endpoint) & (events_df["d"] < before)
    return bool(sel.any())


def _has_event_in_window(events_df: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp,
                         endpoint: str) -> bool:
    """Any evidence for `endpoint` in [lo, hi].

    Boundary inclusive on BOTH ends. Events on lo (== end_date) ARE counted as
    in-horizon because (a) they would otherwise fall in neither the input
    window (`d < end_date`) nor the horizon, silently producing label=0 for
    true positives, and (b) leakage-wise, an event exactly on end_date was
    NOT in the input window so this poses no train/test contamination.
    See v3-spec adversarial review finding M1.
    """
    if len(events_df) == 0:
        return False
    sel = (
        (events_df["endpoint"] == endpoint) &
        (events_df["d"] >= lo) & (events_df["d"] <= hi)
    )
    return bool(sel.any())


def encode_window_v3(
    pid: int,
    end_date: pd.Timestamp,
    fb_end: pd.Timestamp,
    steps_by_d: dict[pd.Timestamp, float],
    hr_by_d: dict[pd.Timestamp, tuple[float, float, float, float]],
    sleep_by_d: dict[pd.Timestamp, tuple[float, float, float, float, float]],
    ehr_pid: pd.DataFrame,
    endpoint_pid: pd.DataFrame,
    death_dates: list[pd.Timestamp],
    confounders: np.ndarray,
    fb_start: pd.Timestamp,
) -> dict:
    """Build one window: wearable matrix + EHR tokens + 13 (label, mask) pairs.

    Mask rules:
      - mortality_*, cv_composite_*: mask = 1 always (no baseline exclusion)
      - t2d_*, dep_*, skin_neoplasm_365d, refractive_errors_365d, dental_caries_365d:
          mask = 1 if no prior diagnosis (any evidence) before end_date AND
                     end_date + horizon <= fb_end (right-censoring)
          else mask = 0 (and label = 0)
      - all heads: mask = 0 if end_date + horizon > fb_end (no future window data)
    """
    # ---- Wearable feature matrix (180, 11)
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
        if d_idx >= 7:
            window7 = feats[d_idx - 7:d_idx, 8]
            valid = window7[window7 > 0]
            if len(valid) >= 2:
                feats[d_idx, 10] = float(np.std(valid))
        mask[d_idx] = any_data

    # ---- EHR events inside the input window
    if len(ehr_pid) > 0:
        ew = ehr_pid[(ehr_pid["d"] >= start) & (ehr_pid["d"] < end_date)].copy()
    else:
        ew = ehr_pid.iloc[0:0]
    if len(ew) > 0:
        day_idx = (pd.to_datetime(ew["d"]) - start).dt.days.astype(np.int16)
        ehr_token_ids = ew["token_id"].to_numpy().astype(np.int32)
        ehr_day_indices = day_idx.to_numpy()
        ehr_token_types = ew["token_type"].to_numpy().astype(np.uint8)
    else:
        ehr_token_ids = np.zeros(0, dtype=np.int32)
        ehr_day_indices = np.zeros(0, dtype=np.int16)
        ehr_token_types = np.zeros(0, dtype=np.uint8)

    # ---- Multi-horizon labels + masks
    out: dict[str, Any] = {
        "person_id": pid,
        "end_date": end_date.date(),
        "wearable_feats": feats.tobytes(),
        "wearable_mask": mask.tobytes(),
        "ehr_token_ids": ehr_token_ids.tobytes(),
        "ehr_day_indices": ehr_day_indices.tobytes(),
        "ehr_token_types": ehr_token_types.tobytes(),
        "confounders": confounders.astype(np.float32).tobytes(),
    }

    for (ep, h) in ALL_HEADS:
        horizon_end = end_date + pd.Timedelta(days=h)
        observable = horizon_end <= fb_end

        # Default: usable, not in baseline-excluded class
        baseline_hit = False
        if ep in BASELINE_EXCLUDED:
            # Use multi-source endpoint evidence (Dx + Lab + Drug) before end_date
            baseline_hit = _has_event_before(endpoint_pid, end_date, ep)

        usable = observable and not baseline_hit
        label = 0
        if usable:
            if ep == "mortality":
                # Include events on exactly end_date (see _has_event_in_window
                # for the boundary rationale — finding M1)
                label = int(any(end_date <= d <= horizon_end for d in death_dates))
            elif ep in ("cv_composite",):
                # cv_composite rows in endpoint_pid come from fetch_endpoint_events
                # after the dx_afib/dx_mi/dx_hf rewrite in tokenize_split.
                label = int(_has_event_in_window(endpoint_pid, end_date, horizon_end, "cv_composite"))
            else:
                label = int(_has_event_in_window(endpoint_pid, end_date, horizon_end, ep))

        name = f"{ep}_{h}d"
        out[f"label_{name}"] = int(label)
        out[f"mask_{name}"]  = int(usable)

    # ---- Subgroup metadata for stratified eval
    sg = _SUBGROUP_CACHE.get(pid, {})
    out["race_concept_id"] = int(sg.get("race_concept_id", 0))
    out["sex_at_birth_concept_id"] = int(sg.get("sex_at_birth_concept_id", 0))
    out["ses_income_quartile"] = int(sg.get("ses_income_quartile", 0))
    # Real age at end_date computed from year + month + day of birth (looked up
    # in precompute_subgroup_metadata as `_BIRTH_DATE_CACHE`). Year-difference
    # alone biases late-year births +1 age bin (finding M3). Fallback to year-
    # diff when month/day are unavailable (AoU often suppresses month/day for
    # privacy on rare birth dates).
    birth_date = _BIRTH_DATE_CACHE.get(pid)
    if birth_date is not None:
        # Strip tz defensively — birth_date from cache may be tz-aware while
        # end_date is tz-naive; pandas refuses to subtract across tz states.
        if getattr(birth_date, "tz", None) is not None:
            birth_date = birth_date.tz_convert(None)
        age_years = int((end_date - birth_date).days // 365)
    else:
        age_years = int(end_date.year - (fb_start.year - int(confounders[0])))
    out["age_at_end_date"] = max(0, age_years)
    return out


# ============================================================
# Per-split driver
# ============================================================

def _assert_first_shard_sanity(split: str) -> None:
    """v2 lesson §14.5: a tokenizer that silently produces all-zero wearable_feats
    cost ~$5 and 1h45m. Catch it on the first shard, not 6 hours into the run."""
    first = OUT_DIR / f"{split}_0000.parquet"
    if not first.exists():
        return
    df = pd.read_parquet(first)
    samples = [np.frombuffer(df.iloc[i]["wearable_feats"], dtype=np.float32)
               for i in range(min(10, len(df)))]
    max_val = float(max(arr.max() for arr in samples))
    assert max_val > 100, (
        f"wearable_feats all near-zero (max={max_val}) on {first.name}. "
        f"Tokenizer is producing degenerate output — abort."
    )
    print(f"sanity OK: {first.name} wearable_feats max = {max_val:.1f}", flush=True)


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
        if shard_idx == 0:
            _assert_first_shard_sanity(split)
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
        ep_df_raw = fetch_endpoint_events(chunk)
        death_df = fetch_deaths(chunk)
        death_by = {pid: sorted(g["d"].tolist()) for pid, g in death_df.groupby("person_id")}

        # Collapse dx_afib/dx_mi/dx_hf into a single cv_composite endpoint row
        # so encode_window_v3's lookup matches the head name.
        ep_df = ep_df_raw.copy()
        ep_df.loc[
            ep_df["endpoint"].isin(("afib", "mi", "hf", "cv_composite")), "endpoint"
        ] = "cv_composite"

        print(
            f"    steps={len(steps_df):>7,d}  hr={len(hr_df):>7,d}  sleep={len(sleep_df):>6,d}"
            f"  ehr={len(ehr_df):>7,d}  endpoint={len(ep_df):>5,d}  death={len(death_df):>3,d}",
            flush=True,
        )

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
        ep_by = ep_df.groupby("person_id")

        for pid in chunk:
            pcohort = cohort[cohort["person_id"] == pid].iloc[0]
            fb_start, fb_end = pcohort["fitbit_start"], pcohort["fitbit_end"]
            confounders = build_confounders(pid)
            steps_d = steps_by.get(pid, {})
            hr_d = hr_by.get(pid, {})
            sleep_d = sleep_by.get(pid, {})
            ehr_pid = (ehr_by.get_group(pid) if pid in ehr_by.groups
                       else pd.DataFrame(columns=["person_id", "d", "token_id", "token_type"]))
            endpoint_pid = (ep_by.get_group(pid) if pid in ep_by.groups
                            else pd.DataFrame(columns=["person_id", "d", "endpoint", "evidence"]))

            death_dates = death_by.get(pid, [])
            end_d = fb_start + pd.Timedelta(days=N_DAYS_INPUT)
            # Generate windows out to fb_end; the per-head mask handles partial horizons
            while end_d <= fb_end - pd.Timedelta(days=HORIZONS[0]):
                out_rows.append(encode_window_v3(
                    pid, end_d, fb_end, steps_d, hr_d, sleep_d, ehr_pid, endpoint_pid,
                    death_dates, confounders, fb_start,
                ))
                if len(out_rows) >= SHARD_SIZE:
                    flush()
                end_d = end_d + pd.Timedelta(days=stride)
    flush()


# ============================================================
# Main
# ============================================================

def main() -> None:
    precompute_confounders_v3()
    precompute_subgroup_metadata()

    tokenize_split("train", TRAIN_STRIDE)
    tokenize_split("val", EVAL_STRIDE)
    tokenize_split("test", EVAL_STRIDE)
    print("\nv3 tokenization complete.", flush=True)


if __name__ == "__main__":
    main()
