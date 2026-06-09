"""PhoneFM v3 self-supervised pretraining — masked daily-vector reconstruction.

Pretext task: given a 90-day wearable sequence with 15% of daily vectors
masked, predict the masked vectors via MSE regression. BERT-style with two
adaptations for continuous wearable features:
  - input_ids at masked positions become MASK_ID (=4), distinguishable from
    WEARABLE_MARKER_ID (=3) so the model knows "predict here"
  - wearable_feats at masked positions are zeroed so the model can't copy
  - regression head outputs an 11-dim vector per position; loss is MSE only
    over masked positions

Reuses the PhoneFMV2 backbone architecture exactly (BatchNorm input norm,
RoPE on day_index, 6-layer transformer). After pretraining, the backbone
state_dict (everything except the regression head) can be loaded into the
v3 supervised model for fine-tuning.

Reuses existing tokenized_v2 parquet shards — only consumes the
wearable_feats and wearable_mask columns. Zero new BQ cost.

Run on Workbench A100:
    nohup python -u 04_pretrain_v3.py > /tmp/pretrain_v3.log 2>&1 &

Outputs to /home/jupyter/workspace/phonefm-data/phonefm_pretrain_v3/
  - best.pt          backbone state_dict + pretrain head, full optimizer state
  - backbone_only.pt backbone state_dict only (for fine-tune init)
  - metrics.json     per-epoch train+val MSE
  - config.json      HP + cfg
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-not-found]
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phonefm_model_v2 import PhoneFMV2Config, RotaryPositionalEmbedding, TransformerBlock  # type: ignore[import-not-found]


# ============================================================
# CELL 1 — config
# ============================================================

HP: dict[str, Any] = dict(
    # window
    n_days_pretrain=90,
    max_seq_len=96,                # CLS + 90 wearable + EOS + a few PAD
    window_stride=7,               # days between successive windows per pid
    # masking
    mask_rate=0.15,                # fraction of valid wearable days to mask
    # backbone (must match v2 / v3 supervised so we can transfer state_dict)
    d_model=384, n_layers=6, n_heads=6, d_ff=1536, dropout=0.1,
    n_wearable_features=11,
    n_token_types=8,
    vocab_size=6007,
    # optimisation
    batch_size=128, grad_accum=1,  # shorter seq → bigger batches fit
    lr=3e-4, weight_decay=0.1, warmup_steps=1000, epochs=20, grad_clip=1.0,
    seed=20260609,
)

# Token-type and id reservations (must mirror phonefm_dataset_v2 + a new MASK_ID)
PAD_ID = 0
CLS_ID = 1
EOS_ID = 2
WEARABLE_MARKER_ID = 3
MASK_ID = 4                        # NEW — predict-here signal

TYPE_PAD = 0
TYPE_CLS = 1
TYPE_EOS = 2
TYPE_WEARABLE = 3

DATA_DIR = Path("/home/jupyter/workspace/phonefm-data/tokenized_v2")
OUT_DIR = Path("/home/jupyter/workspace/phonefm-data/phonefm_pretrain_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(HP["seed"]); np.random.seed(HP["seed"])

print(f"device={DEVICE}  out={OUT_DIR}", flush=True)


# ============================================================
# CELL 2 — dataset
# ============================================================

class WearableMaskedDataset(Dataset):
    """Reads tokenized_v2 shards lazily; for each row samples one 90-day
    subwindow and applies 15% random masking.

    A 180-day v2 window contributes ceil((180 - 90) / stride) = 13 candidate
    subwindows; we pick one uniformly at random per __getitem__ call (so each
    epoch effectively sees a different subwindow of every parent row).
    """

    def __init__(
        self,
        shards_glob: str,
        n_days_pretrain: int = 90,
        n_wearable_features: int = 11,
        n_days_input: int = 180,
    ):
        self.shards = sorted(glob.glob(shards_glob))
        if not self.shards:
            raise FileNotFoundError(f"no shards match {shards_glob}")
        self.n_days_pretrain = n_days_pretrain
        self.n_wearable_features = n_wearable_features
        self.n_days_input = n_days_input
        self.frames = [pd.read_parquet(p) for p in self.shards]
        self.index = [(si, ri) for si, f in enumerate(self.frames) for ri in range(len(f))]
        self.max_start = n_days_input - n_days_pretrain   # inclusive upper bound

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        si, ri = self.index[idx]
        row = self.frames[si].iloc[ri]

        wearable_feats = np.frombuffer(
            row["wearable_feats"], dtype=np.float32
        ).reshape(self.n_days_input, self.n_wearable_features).copy()
        wearable_mask = np.frombuffer(
            row["wearable_mask"], dtype=np.bool_
        ).copy()
        np.nan_to_num(wearable_feats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Pick a random 90-day subwindow.
        start = int(np.random.randint(0, self.max_start + 1))
        feats = wearable_feats[start:start + self.n_days_pretrain]
        valid = wearable_mask[start:start + self.n_days_pretrain]
        return {
            "wearable_feats": torch.from_numpy(feats),    # (n_days_pretrain, F)
            "valid": torch.from_numpy(valid),             # (n_days_pretrain,)
        }


def collate_masked(
    batch: list[dict],
    max_seq_len: int,
    n_days_pretrain: int,
    n_wearable_features: int,
    mask_rate: float,
) -> dict:
    """Build (B, L) tensors with CLS + 90 wearable + EOS + PAD. At masked
    positions: input_ids = MASK_ID, wearable_feats = 0. The MSE loss is
    applied only at positions in `mask_indicator`.
    """
    B = len(batch)
    L = max_seq_len

    input_ids = torch.full((B, L), PAD_ID, dtype=torch.long)
    token_types = torch.full((B, L), TYPE_PAD, dtype=torch.long)
    day_index = torch.zeros((B, L), dtype=torch.long)
    wearable_feats = torch.zeros((B, L, n_wearable_features), dtype=torch.float32)
    wearable_target = torch.zeros((B, L, n_wearable_features), dtype=torch.float32)
    attn_mask = torch.zeros((B, L), dtype=torch.bool)
    mask_indicator = torch.zeros((B, L), dtype=torch.bool)

    for i, item in enumerate(batch):
        # CLS
        input_ids[i, 0] = CLS_ID
        token_types[i, 0] = TYPE_CLS
        day_index[i, 0] = 0
        attn_mask[i, 0] = True

        # Wearable days
        valid_idx = torch.nonzero(item["valid"], as_tuple=True)[0].tolist()
        n_valid = len(valid_idx)
        n_to_mask = max(1, int(round(n_valid * mask_rate)))
        masked_idx = (
            set(np.random.choice(valid_idx, size=n_to_mask, replace=False).tolist())
            if n_valid >= n_to_mask else set()
        )

        for d in range(n_days_pretrain):
            p = 1 + d
            day_index[i, p] = d
            token_types[i, p] = TYPE_WEARABLE
            attn_mask[i, p] = True
            wearable_target[i, p] = item["wearable_feats"][d]
            if d in masked_idx:
                input_ids[i, p] = MASK_ID
                # wearable_feats stays zero
                mask_indicator[i, p] = True
            else:
                input_ids[i, p] = WEARABLE_MARKER_ID
                wearable_feats[i, p] = item["wearable_feats"][d]

        # EOS just past last wearable day
        eos_p = 1 + n_days_pretrain
        input_ids[i, eos_p] = EOS_ID
        token_types[i, eos_p] = TYPE_EOS
        day_index[i, eos_p] = n_days_pretrain
        attn_mask[i, eos_p] = True

    return {
        "input_ids": input_ids,
        "token_types": token_types,
        "day_index": day_index,
        "wearable_feats": wearable_feats,
        "wearable_target": wearable_target,
        "attn_mask": attn_mask,
        "mask_indicator": mask_indicator,
    }


def make_loader(shards_glob: str, batch_size: int, shuffle: bool, num_workers: int = 4) -> DataLoader:
    ds = WearableMaskedDataset(
        shards_glob,
        n_days_pretrain=HP["n_days_pretrain"],
        n_wearable_features=HP["n_wearable_features"],
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=lambda b: collate_masked(
            b, HP["max_seq_len"], HP["n_days_pretrain"],
            HP["n_wearable_features"], HP["mask_rate"],
        ),
        pin_memory=True, persistent_workers=(num_workers > 0),
    )


# ============================================================
# CELL 3 — model: PhoneFMV2 backbone + regression head
# ============================================================

class PhoneFMPretrainV3(nn.Module):
    """v2 backbone (RoPE + BatchNorm input + 6-layer transformer) + linear
    regression head that outputs an 11-dim wearable vector per position.

    Crucially uses the SAME parameter names as PhoneFMV2 for the backbone
    layers so the state_dict (after stripping the pretrain_head) can be
    loaded directly into the v3 supervised model with strict=False.
    """

    def __init__(self, cfg: PhoneFMV2Config):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.type_emb = nn.Embedding(cfg.n_token_types, cfg.d_model)

        self.wearable_input_norm = nn.BatchNorm1d(cfg.n_wearable_features)
        self.wearable_proj = nn.Sequential(
            nn.Linear(cfg.n_wearable_features, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # confounders intentionally NOT used during pretraining; placeholder
        # kept so the state_dict layout stays compatible with v3 supervised.
        self.confounder_input_norm = nn.BatchNorm1d(cfg.n_confounders)
        self.confounder_proj = nn.Sequential(
            nn.Linear(cfg.n_confounders, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        self.rope = RotaryPositionalEmbedding(d_head=cfg.d_model // cfg.n_heads, max_positions=256)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)

        # Pretraining head — discarded for supervised fine-tune
        self.pretrain_head = nn.Linear(cfg.d_model, cfg.n_wearable_features)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_types: torch.Tensor,
        day_index: torch.Tensor,
        wearable_feats: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.tok_emb(input_ids) + self.type_emb(token_types)

        wearable_mask = (token_types == TYPE_WEARABLE).unsqueeze(-1)
        B, L, F_ = wearable_feats.shape
        wearable_norm = self.wearable_input_norm(
            wearable_feats.reshape(B * L, F_)
        ).reshape(B, L, F_)
        wearable_h = self.wearable_proj(wearable_norm)
        h = torch.where(wearable_mask, wearable_h + self.type_emb(token_types), h)

        cos, sin = self.rope(day_index)
        for blk in self.blocks:
            h = blk(h, cos, sin, key_padding_mask=attn_mask)
        h = self.ln_final(h)
        return self.pretrain_head(h)  # (B, L, n_wearable_features)


# ============================================================
# CELL 4 — train loop
# ============================================================

def main() -> None:
    cfg = PhoneFMV2Config(
        d_model=HP["d_model"], n_layers=HP["n_layers"], n_heads=HP["n_heads"],
        d_ff=HP["d_ff"], dropout=HP["dropout"], max_seq_len=HP["max_seq_len"],
        n_wearable_features=HP["n_wearable_features"], n_token_types=HP["n_token_types"],
        vocab_size=HP["vocab_size"],
    )
    model = PhoneFMPretrainV3(cfg).to(DEVICE)
    print(f"model: {model.num_params() / 1e6:.1f}M params", flush=True)

    print("loading train/val...", flush=True)
    train_loader = make_loader(str(DATA_DIR / "train_*.parquet"),
                               batch_size=HP["batch_size"], shuffle=True, num_workers=4)
    val_loader = make_loader(str(DATA_DIR / "val_*.parquet"),
                             batch_size=HP["batch_size"], shuffle=False, num_workers=2)

    # AdamW with no-decay on bias/LayerNorm/BN
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or n.endswith(".bias")) else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": HP["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=HP["lr"], betas=(0.9, 0.95),
    )

    total_steps = HP["epochs"] * math.ceil(len(train_loader) / HP["grad_accum"])
    def lr_at(step: int) -> float:
        if step < HP["warmup_steps"]:
            return HP["lr"] * step / max(1, HP["warmup_steps"])
        progress = (step - HP["warmup_steps"]) / max(1, total_steps - HP["warmup_steps"])
        return HP["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    print(f"total_optimiser_steps={total_steps:,}", flush=True)

    from torch.cuda.amp import autocast as cuda_autocast  # noqa: E402

    @torch.no_grad()
    def evaluate(loader) -> float:
        model.eval()
        total_sq = 0.0; total_n = 0
        for batch in loader:
            pred = model(
                batch["input_ids"].to(DEVICE),
                batch["token_types"].to(DEVICE),
                batch["day_index"].to(DEVICE),
                batch["wearable_feats"].to(DEVICE),
                batch["attn_mask"].to(DEVICE),
            )
            tgt = batch["wearable_target"].to(DEVICE)
            mask = batch["mask_indicator"].to(DEVICE).unsqueeze(-1)
            sq = ((pred - tgt) ** 2 * mask).sum().item()
            n = mask.sum().item() * tgt.shape[-1]
            total_sq += sq; total_n += n
        model.train()
        return total_sq / max(1, total_n)

    metrics_log: list[dict] = []
    best_val = float("inf")
    step = 0
    t0 = time.time()
    model.train()

    for epoch in range(HP["epochs"]):
        for i, batch in enumerate(train_loader):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with cuda_autocast(enabled=(DEVICE == "cuda"), dtype=torch.bfloat16):
                pred = model(
                    batch["input_ids"].to(DEVICE),
                    batch["token_types"].to(DEVICE),
                    batch["day_index"].to(DEVICE),
                    batch["wearable_feats"].to(DEVICE),
                    batch["attn_mask"].to(DEVICE),
                )
                tgt = batch["wearable_target"].to(DEVICE)
                mask = batch["mask_indicator"].to(DEVICE).unsqueeze(-1).float()
                # MSE averaged over masked positions only
                loss = ((pred - tgt) ** 2 * mask).sum() / mask.sum().clamp(min=1.0) / HP["grad_accum"]

            loss.backward()
            if (i + 1) % HP["grad_accum"] == 0:
                nn.utils.clip_grad_norm_(model.parameters(), HP["grad_clip"])
                opt.step()
                opt.zero_grad(set_to_none=True)
                if step % 50 == 0:
                    el = (time.time() - t0) / 60
                    print(f"e{epoch} step {step}/{total_steps}  "
                          f"loss={loss.item() * HP['grad_accum']:.4f}  "
                          f"lr={lr_at(step):.2e}  elapsed={el:.1f}min",
                          flush=True)
                step += 1

        # end-of-epoch eval
        val_mse = evaluate(val_loader)
        print(f"=== epoch {epoch}: val_masked_MSE={val_mse:.6f} ===", flush=True)
        metrics_log.append({"epoch": epoch, "step": step, "val_masked_MSE": val_mse})

        if val_mse < best_val:
            best_val = val_mse
            # Save full state for resume + backbone-only for fine-tune init
            torch.save(model.state_dict(), OUT_DIR / "best.pt")
            backbone_state = {
                k: v for k, v in model.state_dict().items()
                if not k.startswith("pretrain_head.")
            }
            torch.save(backbone_state, OUT_DIR / "backbone_only.pt")
            with open(OUT_DIR / "config.json", "w") as f:
                json.dump({"config": cfg.__dict__, "hp": HP, "val_mse": val_mse}, f, indent=2)
            print(f"  saved best.pt + backbone_only.pt  (val_mse={best_val:.6f})", flush=True)

        with open(OUT_DIR / "metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=2)

    print("pretraining complete", flush=True)


if __name__ == "__main__":
    main()
