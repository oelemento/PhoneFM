"""All of Us Researcher Workbench — cardio cohort extraction.

Paste each cell block into a Workbench Jupyter notebook (Python 3.10 kernel).
Assumes Controlled Tier access; uses BigQuery via `google.cloud.bigquery`
and the `os.environ['WORKSPACE_CDR']` dataset variable that Workbench injects
automatically.

Output: a parquet cohort table at the workspace bucket plus a manifest JSON.
"""

# ============================================================
# CELL 1 — environment + workspace config
# ============================================================
import os
import json
import pandas as pd
import numpy as np
from google.cloud import bigquery, storage
from datetime import datetime, timedelta

bq = bigquery.Client()
CDR = os.environ["WORKSPACE_CDR"]          # eg. "fc-aou-cdr-prod-ct.R2024Q3R1"
BUCKET = os.environ["WORKSPACE_BUCKET"]    # eg. "gs://fc-secure-...."
print(f"CDR    = {CDR}")
print(f"BUCKET = {BUCKET}")


# ============================================================
# CELL 2 — cardio endpoint codesets (ICD-10 / SNOMED)
# ============================================================
CARDIO_ENDPOINTS = {
    "afib":     ["I48"],                      # all I48.x
    "mi":       ["I21", "I22"],               # acute + recurrent MI
    "hf":       ["I50"],                      # all heart failure
    "cardio_mortality": [],                   # via death table cause-of-death
}

def expand_icd10(prefixes):
    """Get OMOP concept ids for ICD-10-CM codes starting with any prefix."""
    q = f"""
    SELECT concept_id, concept_code, concept_name
    FROM `{CDR}.concept`
    WHERE vocabulary_id = 'ICD10CM'
      AND ({' OR '.join([f"concept_code LIKE '{p}%'" for p in prefixes])})
    """
    return bq.query(q).to_dataframe()

endpoint_concept_ids = {}
for name, prefixes in CARDIO_ENDPOINTS.items():
    if not prefixes:
        continue
    df = expand_icd10(prefixes)
    endpoint_concept_ids[name] = df["concept_id"].tolist()
    print(f"{name}: {len(df)} concepts")

# Persist so 02_tokenizer.py can read it even if the kernel was restarted.
with open("/tmp/endpoint_concept_ids.json", "w") as f:
    json.dump(endpoint_concept_ids, f)


# ============================================================
# CELL 3 — identify cohort with Fitbit + EHR linkage
# ============================================================
COHORT_SQL = f"""
WITH fitbit_pts AS (
  -- Participants with at least 180 days of Fitbit heart-rate data
  SELECT person_id, MIN(datetime) AS fitbit_start, MAX(datetime) AS fitbit_end
  FROM `{CDR}.heart_rate_minute_level`
  GROUP BY person_id
  HAVING DATE_DIFF(DATE(MAX(datetime)), DATE(MIN(datetime)), DAY) >= 180
),
ehr_pts AS (
  -- Participants with at least one condition_occurrence record (EHR linkage)
  SELECT DISTINCT person_id
  FROM `{CDR}.condition_occurrence`
),
demographics AS (
  SELECT p.person_id, p.year_of_birth, p.gender_concept_id, p.race_concept_id,
         p.ethnicity_concept_id
  FROM `{CDR}.person` p
)
SELECT
  f.person_id, f.fitbit_start, f.fitbit_end,
  d.year_of_birth,
  EXTRACT(YEAR FROM f.fitbit_start) - d.year_of_birth AS age_at_fitbit_start
FROM fitbit_pts f
JOIN ehr_pts e USING (person_id)
JOIN demographics d USING (person_id)
WHERE EXTRACT(YEAR FROM f.fitbit_start) - d.year_of_birth >= 40
"""
cohort = bq.query(COHORT_SQL).to_dataframe()
print(f"Eligible participants (Fitbit + EHR + age >= 40): {len(cohort):,}")
cohort.to_parquet(f"/tmp/cohort_base.parquet", index=False)


# ============================================================
# CELL 4 — extract cardio events per participant + 30-day labels
# ============================================================
all_endpoint_ids = sorted({cid for ids in endpoint_concept_ids.values() for cid in ids})
ids_csv = ",".join(str(i) for i in all_endpoint_ids)

EVENTS_SQL = f"""
SELECT
  co.person_id,
  co.condition_start_date,
  co.condition_concept_id,
  vo.visit_concept_id,
  vo.visit_start_date,
  vo.visit_end_date,
  CASE
    WHEN vo.visit_concept_id IN (9201, 9203, 262) THEN 'inpatient'
    ELSE 'outpatient'
  END AS encounter_type
FROM `{CDR}.condition_occurrence` co
LEFT JOIN `{CDR}.visit_occurrence` vo USING (visit_occurrence_id)
WHERE co.condition_source_concept_id IN ({ids_csv})
  AND co.person_id IN UNNEST({cohort['person_id'].tolist()})
"""
events = bq.query(EVENTS_SQL).to_dataframe()
print(f"Cardio events in cohort: {len(events):,}")

# Per-participant first cardio event (post-fitbit-start)
events = events.merge(
    cohort[["person_id", "fitbit_start", "fitbit_end"]],
    on="person_id",
)
events["event_date"] = pd.to_datetime(events["condition_start_date"])
events = events[events["event_date"] >= events["fitbit_start"]]
first_event = events.sort_values("event_date").groupby("person_id").first().reset_index()
print(f"Participants with cardio event in Fitbit window: {len(first_event):,}")


# ============================================================
# CELL 5 — train/val/test split with deterministic seed
# ============================================================
from sklearn.model_selection import train_test_split

cohort["has_event"] = cohort["person_id"].isin(first_event["person_id"]).astype(int)
print(f"Event rate in cohort: {cohort['has_event'].mean():.3f}")

train_val, test = train_test_split(
    cohort, test_size=0.15, random_state=20260606,
    stratify=cohort["has_event"],
)
train, val = train_test_split(
    train_val, test_size=0.15/0.85, random_state=20260606,
    stratify=train_val["has_event"],
)
print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}")

# Save split manifest
splits = {
    "train": train["person_id"].tolist(),
    "val":   val["person_id"].tolist(),
    "test":  test["person_id"].tolist(),
}
with open("/tmp/splits.json", "w") as f:
    json.dump(splits, f)


# ============================================================
# CELL 6 — sketch the time-series feature extraction window
# ============================================================
# Per participant: produce 30-day windows ending at random calendar days
# during the Fitbit observation period. Label = 1 if a cardio event occurs
# in the 30 days AFTER the window end; else 0.
# Each window contains:
#   - hourly heart-rate (24 * 30 = 720 ticks)
#   - daily steps (30)
#   - daily sleep summary (30 records of sleep stages)
#   - daily HRV (30)
#   - all EHR events in the 30-day window
# Stub here; flesh out in 02_tokenizer.py.

def example_window_sql(person_id, end_date):
    return f"""
    SELECT person_id, datetime, heart_rate_value
    FROM `{CDR}.heart_rate_minute_level`
    WHERE person_id = {person_id}
      AND datetime BETWEEN '{end_date - timedelta(days=30)}' AND '{end_date}'
    """

print("Cohort extraction complete. Hand off to 02_tokenizer.py.")
