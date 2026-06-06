"""PhoneFM tokenizer — wearable + EHR streams → one token sequence.

Run inside All of Us Workbench. Builds the vocab on the TRAINING split only
(no validation/test leakage into vocab build) and writes:
  - /tmp/vocab.json
  - /tmp/tokenized/<person_id>.npz   (one file per training participant)

Cell order mirrors 01_cohort_extraction.py.
"""

# ============================================================
# CELL 1 — imports + paths
# ============================================================
import os, json
from collections import Counter
from datetime import timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery, storage

bq = bigquery.Client()
CDR    = os.environ["WORKSPACE_CDR"]
BUCKET = os.environ["WORKSPACE_BUCKET"]

OUT_DIR = Path("/tmp/tokenized"); OUT_DIR.mkdir(exist_ok=True)
SPLITS  = json.load(open("/tmp/splits.json"))
COHORT  = pd.read_parquet("/tmp/cohort_base.parquet")

# Window definition
WINDOW_DAYS = 30
LOOKAHEAD_DAYS = 30                  # label = any cardio event in next 30 days
HR_BIN_MIN = 5                       # 5-min binning of HR
MAX_LEN    = 4096                    # cap per-window sequence length


# ============================================================
# CELL 2 — special tokens + reserved ID ranges
# ============================================================
SPECIAL = {
    "<PAD>":     0,
    "<BOS>":     1,
    "<EOS>":     2,
    "<DAY_SEP>": 3,
    "<ENC_SEP>": 4,
    "<UNK>":     5,
}

# Reserved ID ranges (kept stable across data versions so the trained model
# can deploy on phones without renumbering)
RANGES = {
    "HR":       (100, 109),   # 5-min HR decile  HR_D0..HR_D9
    "STEPS":    (110, 119),   # daily-step decile
    "HRV":      (120, 129),   # daily HRV decile
    "SLEEP":    (130, 159),   # SLEEP_REM_<bin>, SLEEP_DEEP_<bin>  (0..4 bins per stage * 3 stages)
    "ECG":      (160, 169),   # ECG events
    "WORKOUT":  (170, 199),   # workout type bucket
    "DX10":     (1000, 3999), # ICD-10 3-char prefix
    "MED":      (4000, 4999), # medication class
    "PX10":     (5000, 5999), # procedure 3-char prefix
    "LAB":      (6000, 7999), # LAB_<itemid>_D<0..9>
}


# ============================================================
# CELL 3 — wearable binning helpers
# ============================================================
def ecdf_decile(values, sorted_ref):
    """Map values to deciles using a pre-computed sorted reference array."""
    if len(values) == 0 or len(sorted_ref) == 0:
        return np.array([], dtype=np.int8)
    ranks = np.searchsorted(sorted_ref, values, side="right")
    return np.clip((ranks * 10 // len(sorted_ref)).astype(np.int8), 0, 9)


def build_decile_reference(stream_name, sql):
    """Pull a stratified random sample (10K participants from training split)
    and compute global decile bin edges. Saves to /tmp/decile_ref_<name>.npy."""
    train_ids = SPLITS["train"]
    sample_ids = np.random.RandomState(42).choice(train_ids,
                                                   size=min(10000, len(train_ids)),
                                                   replace=False)
    df = bq.query(sql.format(ids=",".join(str(i) for i in sample_ids))).to_dataframe()
    arr = np.sort(df["value"].dropna().to_numpy())
    np.save(f"/tmp/decile_ref_{stream_name}.npy", arr)
    print(f"{stream_name}: ref n={len(arr):,}  q10={arr[len(arr)//10]:.2f}  q50={arr[len(arr)//2]:.2f}  q90={arr[9*len(arr)//10]:.2f}")
    return arr

HR_SQL = f"""
SELECT AVG(heart_rate_value) AS value
FROM `{CDR}.heart_rate_minute_level`
WHERE person_id IN ({{ids}})
GROUP BY person_id, DATE(datetime), DIV(EXTRACT(MINUTE FROM datetime), {HR_BIN_MIN})
LIMIT 5000000
"""
STEPS_SQL = f"""
SELECT SUM(steps) AS value
FROM `{CDR}.steps_intraday`
WHERE person_id IN ({{ids}})
GROUP BY person_id, DATE(datetime)
"""
HRV_SQL = f"""
SELECT value
FROM `{CDR}.heart_rate_summary`
WHERE person_id IN ({{ids}})
  AND metric = 'hrv_rmssd'
"""

hr_ref    = build_decile_reference("hr",    HR_SQL)
steps_ref = build_decile_reference("steps", STEPS_SQL)
hrv_ref   = build_decile_reference("hrv",   HRV_SQL)


# ============================================================
# CELL 4 — EHR vocab (from training cohort only)
# ============================================================
def build_ehr_vocab():
    """Count token frequency across training participants, assign IDs within
    the reserved RANGES. Keep top-N within each range; rest map to <UNK>."""
    train_ids = SPLITS["train"]
    ids_csv = ",".join(str(i) for i in train_ids)

    counts = {"DX10": Counter(), "MED": Counter(), "PX10": Counter(), "LAB": Counter()}

    # Diagnoses
    dx = bq.query(f"""
      SELECT co.person_id, c.concept_code
      FROM `{CDR}.condition_occurrence` co
      JOIN `{CDR}.concept` c ON co.condition_concept_id = c.concept_id
      WHERE co.person_id IN ({ids_csv}) AND c.vocabulary_id = 'ICD10CM'
    """).to_dataframe()
    for code in dx["concept_code"].dropna():
        counts["DX10"][f"DX10:{code[:3]}"] += 1

    # Medications  (use ATC class via concept_ancestor)
    meds = bq.query(f"""
      SELECT de.person_id, a.ancestor_concept_id, c.concept_code
      FROM `{CDR}.drug_exposure` de
      JOIN `{CDR}.concept_ancestor` a ON de.drug_concept_id = a.descendant_concept_id
      JOIN `{CDR}.concept` c ON a.ancestor_concept_id = c.concept_id
      WHERE de.person_id IN ({ids_csv}) AND c.vocabulary_id = 'ATC' AND LENGTH(c.concept_code) = 4
    """).to_dataframe()
    for code in meds["concept_code"].dropna():
        counts["MED"][f"MED:{code}"] += 1

    # Procedures
    px = bq.query(f"""
      SELECT po.person_id, c.concept_code
      FROM `{CDR}.procedure_occurrence` po
      JOIN `{CDR}.concept` c ON po.procedure_concept_id = c.concept_id
      WHERE po.person_id IN ({ids_csv}) AND c.vocabulary_id = 'ICD10PCS'
    """).to_dataframe()
    for code in px["concept_code"].dropna():
        counts["PX10"][f"PX10:{code[:3]}"] += 1

    # Labs (LOINC) — keep cardiac labs only
    CARDIAC_LOINC_PREFIXES = ["10839", "30934", "2160-0", "2823-3", "2951-2",  # troponin, BNP, creatinine, K, Na
                              "33762", "33747", "6598-7", "33914"]            # NT-proBNP, hsTnT
    labs = bq.query(f"""
      SELECT m.person_id, c.concept_code, m.value_as_number
      FROM `{CDR}.measurement` m
      JOIN `{CDR}.concept` c ON m.measurement_concept_id = c.concept_id
      WHERE m.person_id IN ({ids_csv}) AND c.vocabulary_id = 'LOINC'
        AND ({' OR '.join([f"c.concept_code LIKE '{p}%'" for p in CARDIAC_LOINC_PREFIXES])})
    """).to_dataframe()
    for code in labs["concept_code"].dropna():
        counts["LAB"][f"LAB:{code}"] += 1

    # Assign IDs within reserved ranges, sorted by frequency (most-frequent gets lowest ID)
    vocab = dict(SPECIAL)
    # Wearable tokens occupy 100..199 — handled inline at encode time, not in vocab here
    for stream, ctr in counts.items():
        lo, hi = RANGES[stream]
        next_id = lo
        for tok, _ in ctr.most_common(hi - lo + 1):
            vocab[tok] = next_id
            next_id += 1
        print(f"{stream}: {len(ctr):,} unique, kept {min(len(ctr), hi-lo+1):,}")
    return vocab

VOCAB = build_ehr_vocab()
with open("/tmp/vocab.json", "w") as f:
    json.dump(VOCAB, f)
print(f"Total vocab size: {len(VOCAB):,}")


# ============================================================
# CELL 5 — encode_window: per-participant 30-day window → token IDs
# ============================================================
def encode_window(person_id: int, end_date: date) -> dict:
    """Pull HR + steps + HRV + EHR for [end_date-30, end_date] and emit
    a single chronologically-merged token sequence.

    Returns {"input_ids": np.int32, "positions": np.int32, "label": int}
    where label = 1 if a cardio event occurs in [end_date+1, end_date+30].
    """
    start_date = end_date - timedelta(days=WINDOW_DAYS)

    # ---- HR (5-min binned, decile token) ----
    hr_df = bq.query(f"""
      SELECT DATE(datetime) AS d, DIV(EXTRACT(MINUTE FROM datetime), {HR_BIN_MIN}) AS m,
             EXTRACT(HOUR FROM datetime) AS h, AVG(heart_rate_value) AS v
      FROM `{CDR}.heart_rate_minute_level`
      WHERE person_id = {person_id} AND DATE(datetime) BETWEEN '{start_date}' AND '{end_date}'
      GROUP BY d, h, m
    """).to_dataframe()
    if len(hr_df):
        hr_df["dec"] = ecdf_decile(hr_df["v"].to_numpy(), hr_ref)
        hr_df["ts"] = pd.to_datetime(hr_df["d"].astype(str)) + pd.to_timedelta(hr_df["h"]*60 + hr_df["m"]*HR_BIN_MIN, unit="m")

    # ---- Steps (daily decile token) ----
    st_df = bq.query(f"""
      SELECT DATE(datetime) AS d, SUM(steps) AS v
      FROM `{CDR}.steps_intraday`
      WHERE person_id = {person_id} AND DATE(datetime) BETWEEN '{start_date}' AND '{end_date}'
      GROUP BY d
    """).to_dataframe()
    if len(st_df):
        st_df["dec"] = ecdf_decile(st_df["v"].to_numpy(), steps_ref)
        st_df["ts"] = pd.to_datetime(st_df["d"].astype(str)) + pd.Timedelta(hours=12)

    # ---- HRV (daily decile token) ----
    hv_df = bq.query(f"""
      SELECT DATE(date) AS d, AVG(value) AS v
      FROM `{CDR}.heart_rate_summary`
      WHERE person_id = {person_id} AND metric = 'hrv_rmssd'
        AND DATE(date) BETWEEN '{start_date}' AND '{end_date}'
      GROUP BY d
    """).to_dataframe()
    if len(hv_df):
        hv_df["dec"] = ecdf_decile(hv_df["v"].to_numpy(), hrv_ref)
        hv_df["ts"] = pd.to_datetime(hv_df["d"].astype(str)) + pd.Timedelta(hours=23)

    # ---- EHR events in window ----
    ehr_sql = f"""
    SELECT * FROM (
      SELECT co.condition_start_date AS ts, CONCAT('DX10:', SUBSTR(c.concept_code, 1, 3)) AS tok
      FROM `{CDR}.condition_occurrence` co
      JOIN `{CDR}.concept` c ON co.condition_concept_id = c.concept_id
      WHERE co.person_id = {person_id}
        AND co.condition_start_date BETWEEN '{start_date}' AND '{end_date}'
        AND c.vocabulary_id = 'ICD10CM'
      UNION ALL
      SELECT po.procedure_date AS ts, CONCAT('PX10:', SUBSTR(c.concept_code, 1, 3)) AS tok
      FROM `{CDR}.procedure_occurrence` po
      JOIN `{CDR}.concept` c ON po.procedure_concept_id = c.concept_id
      WHERE po.person_id = {person_id}
        AND po.procedure_date BETWEEN '{start_date}' AND '{end_date}'
        AND c.vocabulary_id = 'ICD10PCS'
    )
    """
    ehr_df = bq.query(ehr_sql).to_dataframe()
    ehr_df["ts"] = pd.to_datetime(ehr_df["ts"])

    # ---- merge all streams chronologically ----
    rows = []
    for _, r in hr_df.iterrows():
        rows.append((r["ts"], f"HR_D{int(r['dec'])}", 100 + int(r["dec"])))
    for _, r in st_df.iterrows():
        rows.append((r["ts"], f"STEPS_D{int(r['dec'])}", 110 + int(r["dec"])))
    for _, r in hv_df.iterrows():
        rows.append((r["ts"], f"HRV_D{int(r['dec'])}", 120 + int(r["dec"])))
    for _, r in ehr_df.iterrows():
        tid = VOCAB.get(r["tok"], SPECIAL["<UNK>"])
        rows.append((r["ts"], r["tok"], tid))

    rows.sort(key=lambda x: x[0])

    # Build token sequence with <DAY_SEP> between calendar days
    tokens = [SPECIAL["<BOS>"]]
    positions = [0]
    cur_day = None
    day_count = 0
    for ts, _, tid in rows:
        if cur_day is None:
            cur_day = ts.date()
        elif ts.date() != cur_day:
            tokens.append(SPECIAL["<DAY_SEP>"])
            day_count += 1
            positions.append(day_count)
            cur_day = ts.date()
        tokens.append(tid)
        positions.append(day_count)
    tokens.append(SPECIAL["<EOS>"])
    positions.append(day_count)

    # Truncate
    if len(tokens) > MAX_LEN:
        tokens = tokens[-MAX_LEN:]
        positions = positions[-MAX_LEN:]

    # ---- label = any cardio event in [end_date+1, end_date+30] ----
    label_sql = f"""
    SELECT 1 AS hit FROM `{CDR}.condition_occurrence`
    WHERE person_id = {person_id}
      AND condition_start_date BETWEEN DATE_ADD('{end_date}', INTERVAL 1 DAY)
                                  AND DATE_ADD('{end_date}', INTERVAL {LOOKAHEAD_DAYS} DAY)
      AND condition_concept_id IN UNNEST(@cardio_ids)
    LIMIT 1
    """
    job = bq.query(label_sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("cardio_ids", "INT64", CARDIO_CONCEPT_IDS)]
    ))
    label = int(len(job.to_dataframe()) > 0)

    return {
        "input_ids": np.array(tokens, dtype=np.int32),
        "positions": np.array(positions, dtype=np.int32),
        "label": label,
    }


# ============================================================
# CELL 6 — sliding-window sampling per participant + write parquet
# ============================================================
# To get more training examples per participant, sample multiple non-overlapping
# 30-day windows during the Fitbit observation period. Stride = 14 days.
#
# `endpoint_concept_ids` is persisted by 01_cohort_extraction.py to disk so
# this notebook works even after a kernel restart. Falls back to the in-memory
# variable if both files happen to live in one kernel session.
try:
    with open("/tmp/endpoint_concept_ids.json") as _f:
        _endpoints = json.load(_f)
except FileNotFoundError:
    _endpoints = endpoint_concept_ids  # noqa: F821
CARDIO_CONCEPT_IDS = sum(_endpoints.values(), [])

# Stride per split:
#   train:  14d (allow window overlap for more training examples — labels still
#               differ because the 30-day lookahead shifts with end_date)
#   val/test: 31d (non-overlapping windows AND non-overlapping label horizons —
#               prevents inflated AUROC from correlated near-duplicate samples)
STRIDE_BY_SPLIT = {"train": 14, "val": 31, "test": 31}

def emit_participant(person_id, fitbit_start, fitbit_end, stride_days: int):
    out_rows = []
    end = fitbit_end - timedelta(days=LOOKAHEAD_DAYS)  # need lookahead headroom
    cur = fitbit_start + timedelta(days=WINDOW_DAYS)
    while cur <= end:
        try:
            enc = encode_window(person_id, cur.date())
            out_rows.append({
                "person_id": person_id,
                "end_date":  cur.date().isoformat(),
                "input_ids": enc["input_ids"].tobytes(),
                "positions": enc["positions"].tobytes(),
                "n_tokens":  len(enc["input_ids"]),
                "label":     enc["label"],
            })
        except Exception as e:
            print(f"skip {person_id} @ {cur.date()}: {e}")
        cur += timedelta(days=stride_days)
    return out_rows


# Sanity: splits must be disjoint at the participant level
assert set(SPLITS["train"]).isdisjoint(SPLITS["val"]), "train/val overlap!"
assert set(SPLITS["train"]).isdisjoint(SPLITS["test"]), "train/test overlap!"
assert set(SPLITS["val"]).isdisjoint(SPLITS["test"]), "val/test overlap!"

# Stream through participants, write per-split parquet shards
for split_name, ids in SPLITS.items():
    stride = STRIDE_BY_SPLIT[split_name]
    shard, shard_idx, count = [], 0, 0
    for pid in ids:
        meta = COHORT[COHORT.person_id == pid].iloc[0]
        shard.extend(emit_participant(pid, meta.fitbit_start, meta.fitbit_end, stride))
        if len(shard) >= 5000:
            pd.DataFrame(shard).to_parquet(f"/tmp/tokenized/{split_name}_{shard_idx:04d}.parquet")
            count += len(shard); shard_idx += 1; shard = []
    if shard:
        pd.DataFrame(shard).to_parquet(f"/tmp/tokenized/{split_name}_{shard_idx:04d}.parquet")
        count += len(shard)
    print(f"{split_name}: {count:,} windows (stride={stride}d) across {shard_idx + 1} shards")


# ============================================================
# CELL 7 — sync to workspace bucket
# ============================================================
gcs = storage.Client()
bucket_name = BUCKET.replace("gs://", "")
bkt = gcs.bucket(bucket_name)
for p in OUT_DIR.glob("*.parquet"):
    bkt.blob(f"phonefm/tokenized/{p.name}").upload_from_filename(str(p))
bkt.blob("phonefm/vocab.json").upload_from_filename("/tmp/vocab.json")
print("Synced to bucket. Hand off to 03_dataset.py.")
