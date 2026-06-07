"""PhoneFM tokenizer — wearable + EHR streams → one token sequence.

Run inside All of Us Workbench. Builds the vocab on the TRAINING split only
(no validation/test leakage into vocab build) and writes:
  - /tmp/vocab.json
  - /tmp/tokenized/<split>_<idx>.parquet shards
  - /tmp/decile_ref_<stream>.npy decile references

Architecture: bulk-query per stream per chunk (~1500 participants), then
vectorized per-participant slicing + tokenization in Python. Avoids the
per-window BigQuery roundtrip pattern that would otherwise make ~500K calls.
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
WINDOW_DAYS    = 30
LOOKAHEAD_DAYS = 30
HR_BIN_MIN     = 5
MAX_LEN        = 4096
CHUNK_SIZE     = 500    # participants per bulk-fetch batch — keep ≤500 on
                        # 13 GB n1-highmem-2; bulk HR query for 1500 pids
                        # OOMed the JupyterLab server in the previous run.


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

RANGES = {
    "HR":      (100, 109),
    "STEPS":   (110, 119),
    "HRV":     (120, 129),
    "SLEEP":   (130, 159),
    "ECG":     (160, 169),
    "WORKOUT": (170, 199),
    "DX10":    (1000, 3999),
    "MED":     (4000, 4999),
    "PX10":    (5000, 5999),
    "LAB":     (6000, 7999),
}


# ============================================================
# CELL 3 — decile-reference build (one-shot bulk queries)
# ============================================================
def ecdf_decile(values, sorted_ref):
    if len(values) == 0 or len(sorted_ref) == 0:
        return np.array([], dtype=np.int8)
    ranks = np.searchsorted(sorted_ref, values, side="right")
    return np.clip((ranks * 10 // len(sorted_ref)).astype(np.int8), 0, 9)


def build_decile_reference(stream_name, sql):
    train_ids = SPLITS["train"]
    sample_ids = np.random.RandomState(42).choice(
        train_ids, size=min(10000, len(train_ids)), replace=False
    ).tolist()
    df = bq.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("ids", "INT64", sample_ids)]
        ),
    ).to_dataframe()
    arr = np.sort(df["value"].dropna().to_numpy())
    np.save(f"/tmp/decile_ref_{stream_name}.npy", arr)
    print(f"{stream_name}: ref n={len(arr):,}  "
          f"q10={arr[len(arr)//10]:.2f}  q50={arr[len(arr)//2]:.2f}  q90={arr[9*len(arr)//10]:.2f}")
    return arr


HR_SQL = f"""
SELECT AVG(heart_rate_value) AS value
FROM `{CDR}.heart_rate_minute_level`
WHERE person_id IN UNNEST(@ids)
GROUP BY person_id, DATE(datetime), DIV(EXTRACT(MINUTE FROM datetime), {HR_BIN_MIN})
LIMIT 5000000
"""
STEPS_SQL = f"""
SELECT SUM(steps) AS value
FROM `{CDR}.steps_intraday`
WHERE person_id IN UNNEST(@ids)
GROUP BY person_id, DATE(datetime)
"""
# Note: AoU CDR v8 heart_rate_summary is HR-zones (min/max HR per zone), not
# HRV. There is no HRV column. Drop HRV tokens for now — model still gets HR
# + steps + EHR. Future work: compute HRV proxy (rolling std of minute HR)
# from heart_rate_minute_level.
hr_ref    = build_decile_reference("hr",    HR_SQL)
steps_ref = build_decile_reference("steps", STEPS_SQL)
hrv_ref   = np.array([])  # disabled — see note above


# ============================================================
# CELL 4 — EHR vocab (from training cohort only)
# ============================================================
def build_ehr_vocab():
    train_ids = SPLITS["train"]

    def q(sql):
        return bq.query(
            sql,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ArrayQueryParameter("ids", "INT64", list(train_ids))]
            ),
        ).to_dataframe()

    counts = {"DX10": Counter(), "MED": Counter(), "PX10": Counter(), "LAB": Counter()}

    dx = q(f"""
      SELECT co.person_id, c.concept_code
      FROM `{CDR}.condition_occurrence` co
      JOIN `{CDR}.concept` c ON co.condition_source_concept_id = c.concept_id
      WHERE co.person_id IN UNNEST(@ids) AND c.vocabulary_id = 'ICD10CM'
    """)
    for code in dx["concept_code"].dropna():
        counts["DX10"][f"DX10:{code[:3]}"] += 1

    meds = q(f"""
      SELECT de.person_id, c.concept_code
      FROM `{CDR}.drug_exposure` de
      JOIN `{CDR}.concept_ancestor` a ON de.drug_concept_id = a.descendant_concept_id
      JOIN `{CDR}.concept` c ON a.ancestor_concept_id = c.concept_id
      WHERE de.person_id IN UNNEST(@ids) AND c.vocabulary_id = 'ATC' AND LENGTH(c.concept_code) = 4
    """)
    for code in meds["concept_code"].dropna():
        counts["MED"][f"MED:{code}"] += 1

    px = q(f"""
      SELECT po.person_id, c.concept_code
      FROM `{CDR}.procedure_occurrence` po
      JOIN `{CDR}.concept` c ON po.procedure_concept_id = c.concept_id
      WHERE po.person_id IN UNNEST(@ids) AND c.vocabulary_id = 'ICD10PCS'
    """)
    for code in px["concept_code"].dropna():
        counts["PX10"][f"PX10:{code[:3]}"] += 1

    CARDIAC_LOINC = ["10839", "30934", "2160-0", "2823-3", "2951-2",
                     "33762", "33747", "6598-7", "33914"]
    labs = q(f"""
      SELECT m.person_id, c.concept_code, m.value_as_number
      FROM `{CDR}.measurement` m
      JOIN `{CDR}.concept` c ON m.measurement_concept_id = c.concept_id
      WHERE m.person_id IN UNNEST(@ids) AND c.vocabulary_id = 'LOINC'
        AND ({' OR '.join([f"c.concept_code LIKE '{p}%'" for p in CARDIAC_LOINC])})
    """)
    for code in labs["concept_code"].dropna():
        counts["LAB"][f"LAB:{code}"] += 1

    vocab = dict(SPECIAL)
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
# CELL 5 — endpoint concept ids + bulk fetch helpers
# ============================================================
try:
    with open("/tmp/endpoint_concept_ids.json") as _f:
        _endpoints = json.load(_f)
except FileNotFoundError:
    _endpoints = endpoint_concept_ids  # noqa: F821
CARDIO_CONCEPT_IDS = sum(_endpoints.values(), [])
print(f"Cardio concept ids: {len(CARDIO_CONCEPT_IDS)}")


def _q_ids(participant_ids):
    return bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("ids", "INT64", list(participant_ids))]
    )


def fetch_hr_chunk(participant_ids):
    sql = f"""
    SELECT person_id, DATE(datetime) AS d,
           EXTRACT(HOUR FROM datetime) AS h,
           DIV(EXTRACT(MINUTE FROM datetime), {HR_BIN_MIN}) AS m,
           AVG(heart_rate_value) AS v
    FROM `{CDR}.heart_rate_minute_level`
    WHERE person_id IN UNNEST(@ids)
    GROUP BY person_id, d, h, m
    """
    df = bq.query(sql, job_config=_q_ids(participant_ids)).to_dataframe()
    if not len(df):
        return df
    df["dec"] = ecdf_decile(df["v"].to_numpy(), hr_ref)
    df["ts"] = (
        pd.to_datetime(df["d"].astype(str))
        + pd.to_timedelta(df["h"].astype(int) * 60 + df["m"].astype(int) * HR_BIN_MIN, unit="m")
    )
    return df[["person_id", "ts", "dec"]]


def fetch_steps_chunk(participant_ids):
    sql = f"""
    SELECT person_id, DATE(datetime) AS d, SUM(steps) AS v
    FROM `{CDR}.steps_intraday`
    WHERE person_id IN UNNEST(@ids)
    GROUP BY person_id, d
    """
    df = bq.query(sql, job_config=_q_ids(participant_ids)).to_dataframe()
    if not len(df):
        return df
    df["dec"] = ecdf_decile(df["v"].to_numpy(), steps_ref)
    df["ts"] = pd.to_datetime(df["d"].astype(str)) + pd.Timedelta(hours=12)
    return df[["person_id", "ts", "dec"]]


def fetch_hrv_chunk(participant_ids):
    # HRV disabled — no source column in AoU CDR v8 heart_rate_summary.
    return pd.DataFrame(columns=["person_id", "ts", "dec"])


def fetch_ehr_chunk(participant_ids):
    sql = f"""
    SELECT * FROM (
      SELECT co.person_id, co.condition_start_date AS ts,
             CONCAT('DX10:', SUBSTR(c.concept_code, 1, 3)) AS tok
      FROM `{CDR}.condition_occurrence` co
      JOIN `{CDR}.concept` c ON co.condition_source_concept_id = c.concept_id
      WHERE co.person_id IN UNNEST(@ids) AND c.vocabulary_id = 'ICD10CM'
      UNION ALL
      SELECT po.person_id, po.procedure_date AS ts,
             CONCAT('PX10:', SUBSTR(c.concept_code, 1, 3)) AS tok
      FROM `{CDR}.procedure_occurrence` po
      JOIN `{CDR}.concept` c ON po.procedure_concept_id = c.concept_id
      WHERE po.person_id IN UNNEST(@ids) AND c.vocabulary_id = 'ICD10PCS'
    )
    """
    df = bq.query(sql, job_config=_q_ids(participant_ids)).to_dataframe()
    if not len(df):
        return df
    df["ts"] = pd.to_datetime(df["ts"])
    df["tid"] = df["tok"].map(VOCAB).fillna(SPECIAL["<UNK>"]).astype(np.int32)
    return df[["person_id", "ts", "tid"]]


def fetch_cardio_events_chunk(participant_ids):
    sql = f"""
    SELECT person_id, condition_start_date AS ts
    FROM `{CDR}.condition_occurrence`
    WHERE person_id IN UNNEST(@ids)
      AND condition_source_concept_id IN UNNEST(@cardio_ids)
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("ids", "INT64", list(participant_ids)),
        bigquery.ArrayQueryParameter("cardio_ids", "INT64", CARDIO_CONCEPT_IDS),
    ])
    df = bq.query(sql, job_config=cfg).to_dataframe()
    if not len(df):
        return df
    df["ts"] = pd.to_datetime(df["ts"])
    return df[["person_id", "ts"]]


# ============================================================
# CELL 6 — vectorized per-window tokenizer (no BQ, uses caches)
# ============================================================
def encode_window_from_cache(end_date, hr_df, st_df, hv_df, ehr_df, evt_df):
    """All inputs are pre-filtered to a single participant."""
    start_ts = np.datetime64(end_date) - np.timedelta64(WINDOW_DAYS, "D")
    end_ts   = np.datetime64(end_date) + np.timedelta64(1, "D") - np.timedelta64(1, "us")

    ts_chunks = []
    tid_chunks = []
    for df, base in [(hr_df, 100), (st_df, 110), (hv_df, 120)]:
        if df is None or not len(df):
            continue
        m = (df["ts"].values >= start_ts) & (df["ts"].values <= end_ts)
        if not m.any():
            continue
        ts_chunks.append(df["ts"].values[m])
        tid_chunks.append((base + df["dec"].values[m]).astype(np.int32))

    if ehr_df is not None and len(ehr_df):
        m = (ehr_df["ts"].values >= start_ts) & (ehr_df["ts"].values <= end_ts)
        if m.any():
            ts_chunks.append(ehr_df["ts"].values[m])
            tid_chunks.append(ehr_df["tid"].values[m].astype(np.int32))

    if not ts_chunks:
        return None

    ts_all  = np.concatenate(ts_chunks)
    tid_all = np.concatenate(tid_chunks)
    order = np.argsort(ts_all, kind="stable")
    ts_all  = ts_all[order]
    tid_all = tid_all[order]

    # DAY_SEP insertion via day-boundary detection
    days = ts_all.astype("datetime64[D]")
    boundaries = np.where(days[1:] != days[:-1])[0] + 1   # indices where new day starts

    tokens    = [SPECIAL["<BOS>"]]
    positions = [0]
    day_idx   = 0
    prev = 0
    for b in boundaries:
        tokens.extend(tid_all[prev:b].tolist())
        positions.extend([day_idx] * (b - prev))
        tokens.append(SPECIAL["<DAY_SEP>"])
        day_idx += 1
        positions.append(day_idx)
        prev = b
    tokens.extend(tid_all[prev:].tolist())
    positions.extend([day_idx] * (len(tid_all) - prev))
    tokens.append(SPECIAL["<EOS>"])
    positions.append(day_idx)

    if len(tokens) > MAX_LEN:
        tokens    = tokens[-MAX_LEN:]
        positions = positions[-MAX_LEN:]

    # Label: cardio event in [end+1, end+30]
    label = 0
    if evt_df is not None and len(evt_df):
        lo = np.datetime64(end_date) + np.timedelta64(1, "D")
        hi = np.datetime64(end_date) + np.timedelta64(LOOKAHEAD_DAYS, "D")
        label = int(((evt_df["ts"].values >= lo) & (evt_df["ts"].values <= hi)).any())

    return {
        "input_ids": np.array(tokens, dtype=np.int32),
        "positions": np.array(positions, dtype=np.int32),
        "label": label,
    }


# ============================================================
# CELL 7 — chunked tokenization loop
# ============================================================
STRIDE_BY_SPLIT = {"train": 14, "val": 31, "test": 31}

# Participant-disjointness asserts
assert set(SPLITS["train"]).isdisjoint(SPLITS["val"]), "train/val overlap!"
assert set(SPLITS["train"]).isdisjoint(SPLITS["test"]), "train/test overlap!"
assert set(SPLITS["val"]).isdisjoint(SPLITS["test"]), "val/test overlap!"

# Cohort lookup by person_id
COHORT_BY_PID = COHORT.set_index("person_id")


def tokenize_chunk(participant_ids, stride_days):
    """Bulk-fetch then per-participant window encoding."""
    pids = list(participant_ids)
    print(f"  fetching HR…",   flush=True); hr   = fetch_hr_chunk(pids)
    print(f"  fetching steps…",flush=True); st   = fetch_steps_chunk(pids)
    print(f"  fetching HRV…",  flush=True); hv   = fetch_hrv_chunk(pids)
    print(f"  fetching EHR…",  flush=True); ehr  = fetch_ehr_chunk(pids)
    print(f"  fetching events…",flush=True); evt = fetch_cardio_events_chunk(pids)
    print(f"  tokenizing {len(pids):,} participants…", flush=True)

    # Group once by person_id
    def group(df):
        if df is None or not len(df):
            return {}
        return {pid: g.reset_index(drop=True) for pid, g in df.groupby("person_id", sort=False)}
    hr_g  = group(hr)
    st_g  = group(st)
    hv_g  = group(hv)
    ehr_g = group(ehr)
    evt_g = group(evt)

    rows = []
    for pid in pids:
        if pid not in COHORT_BY_PID.index:
            continue
        meta = COHORT_BY_PID.loc[pid]
        fb_start = pd.Timestamp(meta["fitbit_start"]).normalize()
        fb_end   = pd.Timestamp(meta["fitbit_end"]).normalize()
        end_day = (fb_end - timedelta(days=LOOKAHEAD_DAYS)).date()
        cur = (fb_start + timedelta(days=WINDOW_DAYS)).date()
        while cur <= end_day:
            try:
                enc = encode_window_from_cache(
                    cur, hr_g.get(pid), st_g.get(pid), hv_g.get(pid),
                    ehr_g.get(pid), evt_g.get(pid),
                )
                if enc is not None:
                    rows.append({
                        "person_id": int(pid),
                        "end_date":  cur.isoformat(),
                        "input_ids": enc["input_ids"].tobytes(),
                        "positions": enc["positions"].tobytes(),
                        "n_tokens":  int(len(enc["input_ids"])),
                        "label":     int(enc["label"]),
                    })
            except Exception as e:
                print(f"    skip {pid}@{cur}: {e}", flush=True)
            cur += timedelta(days=stride_days)
    return rows


for split_name, ids in SPLITS.items():
    stride = STRIDE_BY_SPLIT[split_name]
    print(f"\n=== {split_name}: {len(ids):,} participants, stride={stride}d ===", flush=True)
    shard, shard_idx, total = [], 0, 0
    n_chunks = (len(ids) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for ci, i in enumerate(range(0, len(ids), CHUNK_SIZE)):
        chunk = ids[i:i + CHUNK_SIZE]
        print(f"chunk {ci+1}/{n_chunks} ({len(chunk):,} participants)…", flush=True)
        rows = tokenize_chunk(chunk, stride)
        shard.extend(rows)
        while len(shard) >= 5000:
            pd.DataFrame(shard[:5000]).to_parquet(
                OUT_DIR / f"{split_name}_{shard_idx:04d}.parquet"
            )
            total += 5000
            shard = shard[5000:]
            shard_idx += 1
            print(f"  wrote shard {split_name}_{shard_idx-1:04d}.parquet  (total {total:,})", flush=True)
    if shard:
        pd.DataFrame(shard).to_parquet(OUT_DIR / f"{split_name}_{shard_idx:04d}.parquet")
        total += len(shard)
        print(f"  wrote final shard {split_name}_{shard_idx:04d}.parquet  (total {total:,})", flush=True)
    print(f"{split_name} DONE: {total:,} windows across {shard_idx+1} shards", flush=True)


# ============================================================
# CELL 8 — sync to bucket
# ============================================================
gcs = storage.Client()
bkt_name = BUCKET.replace("gs://", "")
bkt = gcs.bucket(bkt_name)
for p in OUT_DIR.glob("*.parquet"):
    bkt.blob(f"phonefm/tokenized/{p.name}").upload_from_filename(str(p))
bkt.blob("phonefm/vocab.json").upload_from_filename("/tmp/vocab.json")
for s in ["hr", "steps", "hrv"]:
    bkt.blob(f"phonefm/decile_ref_{s}.npy").upload_from_filename(f"/tmp/decile_ref_{s}.npy")
print("\nSynced to bucket. Hand off to 03_dataset.py.")
