"""PhoneFM v3 dataset — multi-horizon labels + per-endpoint baseline masks.

Parquet shard schema (produced by 02_tokenizer_v3.py):
  person_id              int64
  end_date               date32[day] / pd.Timestamp
  wearable_feats         bytes  (np.float32, shape [180, n_wearable_features])
  wearable_mask          bytes  (np.bool_,   shape [180])
  ehr_token_ids          bytes  (np.int32,   shape [n_events])
  ehr_day_indices        bytes  (np.int16,   shape [n_events])
  ehr_token_types        bytes  (np.uint8,   shape [n_events])
  confounders            bytes  (np.float32, shape [n_confounders=9])

  # For each (endpoint, horizon) in phonefm_model_v3.ALL_HEADS:
  label_{endpoint}_{horizon}d   int8     0/1
  mask_{endpoint}_{horizon}d    int8     1=valid (use for loss), 0=skip

  # Subgroup metadata (for stratified eval — never used in loss):
  race_concept_id        int32
  sex_at_birth_concept_id int32
  age_at_end_date        int16
  ses_income_quartile    int8   0=missing
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd  # type: ignore[import-not-found]
import torch
from torch.utils.data import Dataset, DataLoader

from phonefm_model_v3 import ALL_HEADS, head_name  # type: ignore[import-not-found]


# Token type IDs (mirror cfg.n_token_types order) — same as v2
TYPE_PAD = 0
TYPE_CLS = 1
TYPE_EOS = 2
TYPE_WEARABLE = 3
TYPE_DX10 = 4
TYPE_MED = 5
TYPE_PX10 = 6
TYPE_LAB = 7

# Vocab id reservations — same as v2 (MASK_ID=4 is pretrain-only and never appears in v3 supervised input)
PAD_ID = 0
CLS_ID = 1
EOS_ID = 2
WEARABLE_MARKER_ID = 3

# Names used in the label/mask column schema (sorted by ALL_HEADS order)
HEAD_NAMES: tuple[str, ...] = tuple(head_name(ep, h) for (ep, h) in ALL_HEADS)


class PhoneFMV3Dataset(Dataset):
    """Loads parquet shards; decodes bytes lazily per __getitem__.

    Returns dict of tensors per item; collate_v3 pads variable-length EHR sequences.
    """

    def __init__(
        self,
        shards_glob: str,
        max_seq_len: int = 512,
        n_wearable_features: int = 11,
        n_confounders: int = 9,
        n_days: int = 180,
    ):
        self.shards = sorted(glob.glob(shards_glob))
        if not self.shards:
            raise FileNotFoundError(f"no shards match {shards_glob}")
        self.max_seq_len = max_seq_len
        self.n_wearable_features = n_wearable_features
        self.n_confounders = n_confounders
        self.n_days = n_days
        self.frames = [pd.read_parquet(p) for p in self.shards]
        self.index = [(si, ri) for si, f in enumerate(self.frames) for ri in range(len(f))]

        # Sanity-check first shard has the expected label/mask columns
        first = self.frames[0]
        for name in HEAD_NAMES:
            assert f"label_{name}" in first.columns, f"missing label column: label_{name}"
            assert f"mask_{name}" in first.columns, f"missing mask column: mask_{name}"

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        si, ri = self.index[idx]
        row = self.frames[si].iloc[ri]

        wearable_feats = np.frombuffer(
            row["wearable_feats"], dtype=np.float32
        ).reshape(self.n_days, self.n_wearable_features).copy()
        # Defensive nan_to_num — inherited from v2 (see phonefm_dataset_v2.py:84-91)
        np.nan_to_num(wearable_feats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        wearable_mask = np.frombuffer(row["wearable_mask"], dtype=np.bool_).copy()
        ehr_ids = np.frombuffer(row["ehr_token_ids"], dtype=np.int32).copy()
        ehr_day = np.frombuffer(row["ehr_day_indices"], dtype=np.int16).copy().astype(np.int64)
        ehr_type = np.frombuffer(row["ehr_token_types"], dtype=np.uint8).copy().astype(np.int64)
        confounders = np.frombuffer(
            row["confounders"], dtype=np.float32
        ).reshape(self.n_confounders).copy()
        np.nan_to_num(confounders, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        labels = {name: int(row[f"label_{name}"]) for name in HEAD_NAMES}
        masks = {name: int(row[f"mask_{name}"]) for name in HEAD_NAMES}

        return {
            "person_id": int(row["person_id"]),
            "wearable_feats": torch.from_numpy(wearable_feats),
            "wearable_mask": torch.from_numpy(wearable_mask),
            "ehr_ids": torch.from_numpy(ehr_ids.astype(np.int64)),
            "ehr_day": torch.from_numpy(ehr_day),
            "ehr_type": torch.from_numpy(ehr_type),
            "confounders": torch.from_numpy(confounders),
            "labels": {k: torch.tensor(v, dtype=torch.float32) for k, v in labels.items()},
            "masks": {k: torch.tensor(v, dtype=torch.float32) for k, v in masks.items()},
        }


def collate_v3(batch: list[dict], max_seq_len: int = 512, n_days: int = 180,
               n_wearable_features: int = 11) -> dict:
    """Build (B, L) input tensors with CLS + 180 wearable + EHR events + EOS + PAD.

    Mirrors collate_v2 exactly except for the multi-horizon label/mask layout.
    """
    B = len(batch)
    L = max_seq_len

    input_ids = torch.full((B, L), PAD_ID, dtype=torch.long)
    token_types = torch.full((B, L), TYPE_PAD, dtype=torch.long)
    day_index = torch.zeros((B, L), dtype=torch.long)
    wearable_feats = torch.zeros((B, L, n_wearable_features), dtype=torch.float32)
    attn_mask = torch.zeros((B, L), dtype=torch.bool)

    confounders = torch.zeros((B, batch[0]["confounders"].shape[0]), dtype=torch.float32)
    person_ids = torch.zeros(B, dtype=torch.long)
    labels = {name: torch.zeros(B, dtype=torch.float32) for name in HEAD_NAMES}
    masks = {name: torch.zeros(B, dtype=torch.float32) for name in HEAD_NAMES}

    for i, item in enumerate(batch):
        person_ids[i] = item["person_id"]
        confounders[i] = item["confounders"]
        for name in HEAD_NAMES:
            labels[name][i] = item["labels"][name]
            masks[name][i] = item["masks"][name]

        # CLS at position 0
        input_ids[i, 0] = CLS_ID
        token_types[i, 0] = TYPE_CLS
        day_index[i, 0] = 0
        attn_mask[i, 0] = True

        # Positions 1..n_days: wearable day tokens — always attend (model sees zero-feats on missing days)
        for d in range(n_days):
            p = 1 + d
            input_ids[i, p] = WEARABLE_MARKER_ID
            token_types[i, p] = TYPE_WEARABLE
            day_index[i, p] = d
            wearable_feats[i, p] = item["wearable_feats"][d]
            attn_mask[i, p] = True

        # EHR events sorted by day_index for clean RoPE behavior
        order = torch.argsort(item["ehr_day"])
        ids_sorted = item["ehr_ids"][order]
        day_sorted = item["ehr_day"][order]
        type_sorted = item["ehr_type"][order]
        budget = L - (n_days + 1) - 1  # reserve EOS slot
        keep = min(len(ids_sorted), budget)
        if len(ids_sorted) > keep:
            ids_sorted = ids_sorted[-keep:]
            day_sorted = day_sorted[-keep:]
            type_sorted = type_sorted[-keep:]

        ehr_start = n_days + 1
        ehr_end = ehr_start + keep
        input_ids[i, ehr_start:ehr_end] = ids_sorted
        token_types[i, ehr_start:ehr_end] = type_sorted
        day_index[i, ehr_start:ehr_end] = day_sorted
        attn_mask[i, ehr_start:ehr_end] = True

        input_ids[i, ehr_end] = EOS_ID
        token_types[i, ehr_end] = TYPE_EOS
        day_index[i, ehr_end] = 180
        attn_mask[i, ehr_end] = True

    return {
        "person_id": person_ids,
        "input_ids": input_ids,
        "token_types": token_types,
        "day_index": day_index,
        "wearable_feats": wearable_feats,
        "confounders": confounders,
        "attn_mask": attn_mask,
        "labels": labels,
        "masks": masks,
    }


def make_loader_v3(
    shards_glob: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    max_seq_len: int = 512,
    n_confounders: int = 9,
) -> DataLoader:
    """Standard DataLoader. WeightedRandomSampler is deliberately NOT supported
    in v3 — v2 lesson §14.2: pos_weight in BCE handles class imbalance; sampler
    on top of pos_weight is double-balancing and was the primary v2 NaN trigger.
    """
    ds = PhoneFMV3Dataset(
        shards_glob, max_seq_len=max_seq_len, n_confounders=n_confounders,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=lambda b: collate_v3(b, max_seq_len=max_seq_len),
        pin_memory=True, persistent_workers=(num_workers > 0),
    )


def report_label_distribution(shards_glob: str) -> dict[str, dict[str, float]]:
    """Per-head positives, mask-aware pos_rate, and capped pos_weight for BCE.

    pos_rate is computed over (mask=1) samples only — for baseline-excluded
    endpoints, this gives the at-risk positive rate, which is what BCE pos_weight
    should target.

    pos_weight is capped at MAX_POS_WEIGHT=50 per v2 lesson §14.2 (cv_death raw
    pos_weight 212,797 blew BCE to NaN at step 50).
    """
    frames = [pd.read_parquet(p) for p in sorted(glob.glob(shards_glob))]
    out = {}
    MAX_POS_WEIGHT = 50.0
    for name in HEAD_NAMES:
        labels = np.concatenate([f[f"label_{name}"].values for f in frames]).astype(np.int64)
        masks_c = np.concatenate([f[f"mask_{name}"].values for f in frames]).astype(np.int64)
        # Restrict to mask=1 samples for the rate calculation
        valid = masks_c == 1
        n_valid = int(valid.sum())
        n_pos = int(labels[valid].sum())
        if n_pos == 0 or n_valid == 0:
            pos_weight = 1.0
        else:
            pos_weight = min((n_valid - n_pos) / n_pos, MAX_POS_WEIGHT)
        out[name] = {
            "n_total": int(len(labels)),
            "n_valid": n_valid,
            "n_pos": n_pos,
            "pos_rate_among_valid": n_pos / max(n_valid, 1),
            "pos_weight_for_bce": float(pos_weight),
        }
    return out
