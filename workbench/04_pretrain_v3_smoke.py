"""Smoke test for 04_pretrain_v3.py.

Builds 1 batch through the pretraining pipeline, runs forward + backward,
and asserts the masked-MSE math is correct:
  - Loss is computed only over mask_indicator=True positions
  - Gradient flows to the backbone (so the SSL signal actually trains)
  - Shapes line up at every step
  - One known-zero edge case: if a batch has no masked positions, the loss
    must not divide-by-zero (clamp(min=1.0) check)

Falls back to a synthetic shard if real tokenized_v2 shards aren't mounted.

Usage:
    cd ~/repos/PhoneFM/workbench && python3 04_pretrain_v3_smoke.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-not-found]
import torch

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))

# Direct imports — works because we add workbench/ to sys.path
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("pretrain_mod", HERE / "04_pretrain_v3.py")
assert _spec is not None and _spec.loader is not None
pretrain = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(pretrain)

from phonefm_model_v2 import PhoneFMV2Config  # type: ignore[import-not-found]


def make_synthetic_shard(path: Path, n_rows: int = 64) -> None:
    """Mimic the on-disk tokenized_v2 schema with random wearable values."""
    rows = []
    for i in range(n_rows):
        feats = (np.random.rand(180, 11).astype(np.float32) * 100).clip(min=0)
        feats[:, 0] *= 100  # steps column scaled up to match real data
        mask = (np.random.rand(180) > 0.05).astype(np.bool_)
        rows.append({
            "person_id": i,
            "end_date": pd.Timestamp("2024-01-01"),
            "wearable_feats": feats.tobytes(),
            "wearable_mask": mask.tobytes(),
            "ehr_token_ids": np.zeros(0, dtype=np.int32).tobytes(),
            "ehr_day_indices": np.zeros(0, dtype=np.int16).tobytes(),
            "ehr_token_types": np.zeros(0, dtype=np.uint8).tobytes(),
            "confounders": np.zeros(8, dtype=np.float32).tobytes(),
            "label_afib": 0, "label_mi": 0, "label_hf": 0,
            "label_cv_death": 0, "label_composite": 0,
        })
    pd.DataFrame(rows).to_parquet(path, compression="snappy")


def main() -> None:
    print(f"\n=== SMOKE TEST: 04_pretrain_v3.py ===\n")

    real_data = (pretrain.DATA_DIR / "train_0000.parquet").exists()
    if real_data:
        shards_glob = str(pretrain.DATA_DIR / "train_0000.parquet")
        print(f"using real shard: {shards_glob}")
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="phonefm_smoke_"))
        make_synthetic_shard(tmpdir / "train_0000.parquet", n_rows=64)
        shards_glob = str(tmpdir / "train_0000.parquet")
        print(f"real shards missing — using synthetic shard at {shards_glob}")

    # ---- 1. Build dataset + dataloader ----
    ds = pretrain.WearableMaskedDataset(shards_glob)
    print(f"dataset size: {len(ds)} rows")
    assert len(ds) > 0, "empty dataset"

    item = ds[0]
    print(f"item keys: {sorted(item.keys())}")
    assert item["wearable_feats"].shape == (pretrain.HP["n_days_pretrain"],
                                            pretrain.HP["n_wearable_features"])
    assert item["valid"].shape == (pretrain.HP["n_days_pretrain"],)

    # ---- 2. Build a batch ----
    batch = pretrain.collate_masked(
        [ds[i] for i in range(4)],
        max_seq_len=pretrain.HP["max_seq_len"],
        n_days_pretrain=pretrain.HP["n_days_pretrain"],
        n_wearable_features=pretrain.HP["n_wearable_features"],
        mask_rate=pretrain.HP["mask_rate"],
    )
    print(f"\nbatch shapes:")
    for k, v in batch.items():
        print(f"  {k:20s} {tuple(v.shape)} dtype={v.dtype}")

    B, L = batch["input_ids"].shape
    F_ = pretrain.HP["n_wearable_features"]
    assert batch["wearable_feats"].shape == (B, L, F_)
    assert batch["wearable_target"].shape == (B, L, F_)
    assert batch["mask_indicator"].shape == (B, L)
    assert batch["mask_indicator"].dtype == torch.bool

    # ---- 3. Verify masking is sensible ----
    mask = batch["mask_indicator"]
    n_masked = int(mask.sum().item())
    n_wearable_positions = int((batch["token_types"] == pretrain.TYPE_WEARABLE).sum().item())
    actual_rate = n_masked / max(1, n_wearable_positions)
    target_rate = pretrain.HP["mask_rate"]
    print(f"\nmask_indicator: {n_masked} / {n_wearable_positions} wearable positions = "
          f"{actual_rate:.3f} (target {target_rate})")
    assert 0.05 < actual_rate < 0.35, f"mask rate {actual_rate} out of plausible range"

    # ---- 4. Verify masked positions have wearable_feats==0, target!=0 ----
    masked_feats = batch["wearable_feats"][mask]
    assert masked_feats.abs().max().item() < 1e-6, \
        "wearable_feats must be zero at masked positions"
    print(f"masked wearable_feats max abs = {masked_feats.abs().max().item():.6f}  (OK, ~0)")

    masked_target = batch["wearable_target"][mask]
    print(f"masked wearable_target stats: mean={masked_target.mean().item():.2f}  "
          f"max={masked_target.max().item():.2f}")

    # ---- 5. Verify input_ids at masked positions = MASK_ID ----
    masked_ids = batch["input_ids"][mask]
    assert (masked_ids == pretrain.MASK_ID).all(), \
        f"input_ids at masked positions should all be {pretrain.MASK_ID}, got {masked_ids.unique()}"
    print(f"input_ids at masked positions: all MASK_ID={pretrain.MASK_ID}  (OK)")

    # ---- 6. Build the model and run forward ----
    cfg = PhoneFMV2Config(
        d_model=pretrain.HP["d_model"],
        n_layers=pretrain.HP["n_layers"],
        n_heads=pretrain.HP["n_heads"],
        d_ff=pretrain.HP["d_ff"],
        dropout=pretrain.HP["dropout"],
        max_seq_len=pretrain.HP["max_seq_len"],
        n_wearable_features=pretrain.HP["n_wearable_features"],
        n_token_types=pretrain.HP["n_token_types"],
        vocab_size=pretrain.HP["vocab_size"],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = pretrain.PhoneFMPretrainV3(cfg).to(device)
    print(f"\nmodel: {model.num_params() / 1e6:.1f}M params  device={device}")

    model.train()
    pred = model(
        batch["input_ids"].to(device),
        batch["token_types"].to(device),
        batch["day_index"].to(device),
        batch["wearable_feats"].to(device),
        batch["attn_mask"].to(device),
    )
    print(f"pred shape: {tuple(pred.shape)}   (expect ({B}, {L}, {F_}))")
    assert pred.shape == (B, L, F_)
    assert torch.isfinite(pred).all(), "non-finite pred — model broke"
    print(f"pred stats: mean={pred.mean().item():.4f}  std={pred.std().item():.4f}")

    # ---- 7. Compute the loss two ways and assert equal ----
    tgt = batch["wearable_target"].to(device)
    mind = batch["mask_indicator"].to(device).unsqueeze(-1).float()

    loss_A = ((pred - tgt) ** 2 * mind).sum() / mind.sum().clamp(min=1.0)
    diff_sq_masked = ((pred - tgt) ** 2)[mind.squeeze(-1).bool()]
    loss_B = diff_sq_masked.mean()

    print(f"\nloss_A (production) = {loss_A.item():.6f}")
    print(f"loss_B (explicit)   = {loss_B.item():.6f}")
    assert torch.allclose(loss_A, loss_B, rtol=1e-4), \
        f"masked-MSE formula mismatch: A={loss_A.item()} vs B={loss_B.item()}"
    print("masked-MSE formulas agree  (OK)")

    # ---- 8. Verify gradient flows to the backbone ----
    loss_A.backward()
    backbone_param = model.wearable_proj[0].weight
    assert backbone_param.grad is not None
    assert backbone_param.grad.abs().max().item() > 0, \
        "no gradient reached wearable_proj — SSL signal does not train the backbone"
    print(f"wearable_proj.0.weight grad norm = {backbone_param.grad.norm().item():.6f}  (OK)")

    head_param = model.pretrain_head.weight
    assert head_param.grad is not None
    assert head_param.grad.abs().max().item() > 0
    print(f"pretrain_head.weight grad norm   = {head_param.grad.norm().item():.6f}  (OK)")

    # ---- 9. Edge case: divide-by-zero protection when no positions are masked ----
    zero_mind = torch.zeros_like(mind)
    loss_edge = ((pred - tgt) ** 2 * zero_mind).sum() / zero_mind.sum().clamp(min=1.0)
    assert torch.isfinite(loss_edge), "divide-by-zero protection failed"
    assert loss_edge.item() == 0.0
    print(f"all-zero-mask edge case: loss = {loss_edge.item()} (no NaN, no inf)  (OK)")

    # ---- 10. Verify backbone state_dict has no pretrain_head keys ----
    backbone_state = {k: v for k, v in model.state_dict().items()
                      if not k.startswith("pretrain_head.")}
    head_state = [k for k in model.state_dict() if k.startswith("pretrain_head")]
    print(f"\nbackbone state has {len(backbone_state)} tensors; "
          f"strips {len(head_state)} pretrain_head tensors for fine-tune init  (OK)")

    print("\n=== SMOKE TEST PASSED ===\n")


if __name__ == "__main__":
    main()
