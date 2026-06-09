"""PhoneFM v2 dataset — daily wearable features + EHR events + multi-head labels.

Parquet shard schema (produced by 02_tokenizer_v2.py):
  person_id              int64
  end_date               date32[day] / pd.Timestamp
  wearable_feats         bytes  (np.float32, shape [180, n_wearable_features])
  wearable_mask          bytes  (np.bool_,   shape [180])
  ehr_token_ids          bytes  (np.int32,   shape [n_events])
  ehr_day_indices        bytes  (np.int16,   shape [n_events])  -- 0..179
  ehr_token_types        bytes  (np.uint8,   shape [n_events])  -- 4=DX10, 5=MED, 6=PX10, 7=LAB
  confounders            bytes  (np.float32, shape [n_confounders])
  label_afib             int8
  label_mi               int8
  label_hf               int8
  label_cv_death         int8
  label_composite        int8

All variable-length arrays are stored as raw bytes via np.tobytes() and decoded with np.frombuffer.
This avoids the per-cell pandas object overhead seen with native list columns.
"""

from __future__ import annotations

import glob
from typing import Optional

import numpy as np
import pandas as pd  # type: ignore[import-not-found]
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# Token type IDs (mirror cfg.n_token_types order)
TYPE_PAD = 0
TYPE_CLS = 1
TYPE_EOS = 2
TYPE_WEARABLE = 3
TYPE_DX10 = 4
TYPE_MED = 5
TYPE_PX10 = 6
TYPE_LAB = 7

# Vocab id reservations (must match tokenizer_v2 + model)
PAD_ID = 0
CLS_ID = 1
EOS_ID = 2
WEARABLE_MARKER_ID = 3

ENDPOINTS = ("afib", "mi", "hf", "cv_death", "composite")


class PhoneFMV2Dataset(Dataset):
    """Loads one or more parquet shards, decodes bytes lazily per __getitem__.

    Returns dict of tensors per item; the collate_fn pads variable-length EHR sequences.
    """

    def __init__(
        self,
        shards_glob: str,
        max_seq_len: int = 512,
        n_wearable_features: int = 11,
        n_confounders: int = 8,
        n_days: int = 180,
    ):
        self.shards = sorted(glob.glob(shards_glob))
        if not self.shards:
            raise FileNotFoundError(f"no shards match {shards_glob}")
        self.max_seq_len = max_seq_len
        self.n_wearable_features = n_wearable_features
        self.n_confounders = n_confounders
        self.n_days = n_days
        # Load all shards as pandas frames; index across them
        self.frames = [pd.read_parquet(p) for p in self.shards]
        self.index = [(si, ri) for si, f in enumerate(self.frames) for ri in range(len(f))]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        si, ri = self.index[idx]
        row = self.frames[si].iloc[ri]

        wearable_feats = np.frombuffer(
            row["wearable_feats"], dtype=np.float32
        ).reshape(self.n_days, self.n_wearable_features).copy()
        wearable_mask = np.frombuffer(row["wearable_mask"], dtype=np.bool_).copy()
        ehr_ids = np.frombuffer(row["ehr_token_ids"], dtype=np.int32).copy()
        ehr_day = np.frombuffer(row["ehr_day_indices"], dtype=np.int16).copy().astype(np.int64)
        ehr_type = np.frombuffer(row["ehr_token_types"], dtype=np.uint8).copy().astype(np.int64)
        confounders = np.frombuffer(
            row["confounders"], dtype=np.float32
        ).reshape(self.n_confounders).copy()

        labels = {ep: int(row[f"label_{ep}"]) for ep in ENDPOINTS}
        return {
            "wearable_feats": torch.from_numpy(wearable_feats),
            "wearable_mask": torch.from_numpy(wearable_mask),
            "ehr_ids": torch.from_numpy(ehr_ids.astype(np.int64)),
            "ehr_day": torch.from_numpy(ehr_day),
            "ehr_type": torch.from_numpy(ehr_type),
            "confounders": torch.from_numpy(confounders),
            "labels": {k: torch.tensor(v, dtype=torch.float32) for k, v in labels.items()},
        }


def collate_v2(batch: list[dict], max_seq_len: int = 512, n_days: int = 180,
               n_wearable_features: int = 11) -> dict:
    """Build the (B, L) input tensors with CLS + 180 wearable + EHR events + EOS + PAD."""
    B = len(batch)
    L = max_seq_len

    input_ids = torch.full((B, L), PAD_ID, dtype=torch.long)
    token_types = torch.full((B, L), TYPE_PAD, dtype=torch.long)
    day_index = torch.zeros((B, L), dtype=torch.long)
    wearable_feats = torch.zeros((B, L, n_wearable_features), dtype=torch.float32)
    attn_mask = torch.zeros((B, L), dtype=torch.bool)

    confounders = torch.zeros((B, batch[0]["confounders"].shape[0]), dtype=torch.float32)
    labels = {ep: torch.zeros(B, dtype=torch.float32) for ep in ENDPOINTS}

    for i, item in enumerate(batch):
        confounders[i] = item["confounders"]
        for ep in ENDPOINTS:
            labels[ep][i] = item["labels"][ep]

        # Position 0: CLS
        input_ids[i, 0] = CLS_ID
        token_types[i, 0] = TYPE_CLS
        day_index[i, 0] = 0
        attn_mask[i, 0] = True

        # Positions 1..n_days: wearable day tokens (always include all days; mask
        # invalid via attn_mask if no data that day so the position still counts
        # as "real" for the model — but we keep its features zero).
        for d in range(n_days):
            p = 1 + d
            input_ids[i, p] = WEARABLE_MARKER_ID
            token_types[i, p] = TYPE_WEARABLE
            day_index[i, p] = d
            wearable_feats[i, p] = item["wearable_feats"][d]
            attn_mask[i, p] = True  # always attend, even on missing days; model sees zero-feats

        # Positions n_days+1 .. : EHR events (sorted by day_index for clean RoPE behavior)
        order = torch.argsort(item["ehr_day"])
        ids_sorted = item["ehr_ids"][order]
        day_sorted = item["ehr_day"][order]
        type_sorted = item["ehr_type"][order]
        budget = L - (n_days + 1) - 1  # reserve EOS slot
        keep = min(len(ids_sorted), budget)
        # If too many events, keep the MOST RECENT (largest day_index)
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

        # EOS just past last EHR event
        input_ids[i, ehr_end] = EOS_ID
        token_types[i, ehr_end] = TYPE_EOS
        day_index[i, ehr_end] = 180  # one past the input window
        attn_mask[i, ehr_end] = True

    return {
        "input_ids": input_ids,
        "token_types": token_types,
        "day_index": day_index,
        "wearable_feats": wearable_feats,
        "confounders": confounders,
        "attn_mask": attn_mask,
        "labels": labels,
    }


def make_loader_v2(
    shards_glob: str,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    balance_on: Optional[str] = "composite",   # WeightedRandomSampler on this endpoint
    max_seq_len: int = 512,
) -> DataLoader:
    ds = PhoneFMV2Dataset(shards_glob, max_seq_len=max_seq_len)

    if balance_on is not None:
        # build weights from the requested endpoint
        labels = np.concatenate(
            [f[f"label_{balance_on}"].values for f in ds.frames]
        ).astype(np.int64)
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        w_pos = 0.5 / max(n_pos, 1)
        w_neg = 0.5 / max(n_neg, 1)
        weights = np.where(labels == 1, w_pos, w_neg)
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )
        return DataLoader(
            ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
            collate_fn=lambda b: collate_v2(b, max_seq_len=max_seq_len),
            pin_memory=True, persistent_workers=(num_workers > 0),
        )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=lambda b: collate_v2(b, max_seq_len=max_seq_len),
        pin_memory=True, persistent_workers=(num_workers > 0),
    )


def report_label_distribution(shards_glob: str) -> dict[str, dict[str, float]]:
    """Diagnostic for setting pos_weight per endpoint.

    Important: endpoints with n_pos==0 get pos_weight_for_bce=1.0 (NOT n/1) so the
    loss is not blown up by a non-existent positive class. The training loop should
    additionally drop such heads from the loss; this function only ensures the
    weight is benign if it slips through.
    """
    frames = [pd.read_parquet(p) for p in sorted(glob.glob(shards_glob))]
    n = sum(len(f) for f in frames)
    out = {}
    for ep in ENDPOINTS:
        col = np.concatenate([f[f"label_{ep}"].values for f in frames])
        n_pos = int(col.sum())
        if n_pos == 0:
            # No positives in this split: keep weight benign and flag.
            pos_weight = 1.0
        else:
            pos_weight = (n - n_pos) / n_pos
        out[ep] = {
            "n": n, "n_pos": n_pos, "pos_rate": n_pos / max(n, 1),
            "pos_weight_for_bce": float(pos_weight),
        }
    return out
