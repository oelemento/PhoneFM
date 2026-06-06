"""PhoneFM dataset — parquet shards → PyTorch DataLoader.

Run inside the same Workbench kernel as 02_tokenizer.py (or after). Loads the
tokenized parquet shards from /tmp/tokenized/ and exposes a streaming Dataset
that pads/masks on the fly.

Designed so 05_train.py can do:

    from importlib import import_module
    ds_mod = import_module("03_dataset")  # filename note below
    train_dl = ds_mod.make_loader("train", batch_size=32)
"""

# NOTE: Python module names can't start with a digit. When pasting into
# Workbench, save this notebook as `phonefm_dataset.py` for imports, or copy
# the make_loader() function directly into the training cell.

# ============================================================
# CELL 1 — imports
# ============================================================
import json, glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

VOCAB     = json.load(open("/tmp/vocab.json"))
PAD_ID    = VOCAB["<PAD>"]
VOCAB_SIZE = max(VOCAB.values()) + 1  # ID space, not unique-token count
MAX_LEN   = 4096

TOK_DIR   = Path("/tmp/tokenized")


# ============================================================
# CELL 2 — dataset class
# ============================================================
class CardioWindowsDataset(Dataset):
    """Streaming dataset over parquet shards for one split.

    Each row → (input_ids: LongTensor[L], positions: LongTensor[L],
                attn_mask: BoolTensor[L], label: float32).
    Padding to MAX_LEN happens in __getitem__ so DataLoader's default collate
    can stack into a batch.
    """

    def __init__(self, split: str, max_len: int = MAX_LEN,
                 weights_by_label: Optional[dict] = None):
        self.shards = sorted(glob.glob(str(TOK_DIR / f"{split}_*.parquet")))
        if not self.shards:
            raise RuntimeError(f"No shards found for split={split} in {TOK_DIR}")
        # Build a flat index of (shard_idx, row_idx) so we can sample with
        # standard PyTorch samplers. Shards are small enough to mmap.
        self.frames = [pd.read_parquet(p) for p in self.shards]
        self.index = [(si, ri) for si, fr in enumerate(self.frames) for ri in range(len(fr))]
        self.max_len = max_len
        self.weights_by_label = weights_by_label or {}
        # Pre-compute per-sample weights for WeightedRandomSampler (class balance)
        labels = np.concatenate([fr["label"].to_numpy() for fr in self.frames])
        if not weights_by_label:
            pos = (labels == 1).mean()
            self.weights_by_label = {0: 1.0, 1: max(1.0, (1 - pos) / max(pos, 1e-6))}
        self.sample_weights = np.array([self.weights_by_label[int(l)] for l in labels],
                                       dtype=np.float64)
        print(f"{split}: {len(self.index):,} windows  pos_rate={(labels==1).mean():.4f}  "
              f"pos_weight={self.weights_by_label[1]:.2f}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si, ri = self.index[idx]
        row = self.frames[si].iloc[ri]
        # np.frombuffer returns a read-only view of the parquet bytes; .copy()
        # gives a writable array so torch.from_numpy doesn't pin us to RO mode.
        ids = np.frombuffer(row["input_ids"], dtype=np.int32).copy()
        pos = np.frombuffer(row["positions"], dtype=np.int32).copy()
        # Bounds check — out-of-range IDs trigger silent CUDA assertion later
        if __debug__:
            assert ids.max(initial=0) < VOCAB_SIZE, (
                f"input_ids out of range: max={ids.max()}, VOCAB_SIZE={VOCAB_SIZE}"
            )
        L = min(len(ids), self.max_len)
        ids = ids[-L:]; pos = pos[-L:]
        # Pad right to max_len
        out_ids = np.full(self.max_len, PAD_ID, dtype=np.int64)
        out_pos = np.zeros(self.max_len, dtype=np.int64)
        out_ids[:L] = ids
        out_pos[:L] = pos
        attn = np.zeros(self.max_len, dtype=bool); attn[:L] = True
        return {
            "input_ids": torch.from_numpy(out_ids),
            "positions": torch.from_numpy(out_pos),
            "attn_mask": torch.from_numpy(attn),
            "label":     torch.tensor(float(row["label"])),
        }


# ============================================================
# CELL 3 — DataLoader factory
# ============================================================
def make_loader(split: str,
                batch_size: int = 32,
                num_workers: int = 4,
                shuffle: bool = True,
                balance: bool = True) -> DataLoader:
    ds = CardioWindowsDataset(split)
    if shuffle and balance:
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=ds.sample_weights.tolist(),
            num_samples=len(ds),
            replacement=True,
        )
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


# ============================================================
# CELL 4 — sanity check
# ============================================================
if __name__ == "__main__":
    dl = make_loader("val", batch_size=4, num_workers=0)
    batch = next(iter(dl))
    print({k: (v.shape, v.dtype) for k, v in batch.items()})
    print(f"labels: {batch['label'].tolist()}  "
          f"first sequence n_real_tokens: {batch['attn_mask'][0].sum().item()}")
