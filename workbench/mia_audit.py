"""
mia_audit.py (v2) — Loss-based membership-inference audit for the PhoneFM v3
checkpoint, supporting the AoU/Verily Controlled-Tier egress request.

Design (v2, post-adversarial-review):

  PRIMARY metric — single-head MIA on the headline endpoint (cv_composite_30d):
    per-window loss = BCE-with-logits(logit, label, pos_weight=pw) for the
    cv_composite_30d head. No mask-pattern confound, no multi-head weighting
    drift, clean interpretation.

  SECONDARY metric — multi-head un-normalized weighted-sum MIA:
    per-window loss = sum_h [head_weight_h * BCE_h * mask_h]  over active heads
    (heads with n_pos >= 50 in train, matching 05_train_v3.py). No per-window
    denominator (training's batch-level 1/mask_sum_h cannot be reconstructed
    at audit time). Reported alongside the primary for diagnostic transparency.

  Sampling: PERSONS-first, then up to K windows per person. Avoids the
  window-count bias where heavy-window persons dominate the member arm but
  not the smaller non-member arms.

  Reproducibility: num_workers=0, deterministic sampling plan computed up
  front from dataset.index, Subset loader with shuffle=False.

  Loss replication: bf16 autocast matching training (05_train_v3.py:290).
  Logits up-cast to fp32 before BCE to avoid bf16 underflow at low losses.

  Output: numbers only, no verdict strings — the human picks thresholds.
  Train-vs-val computed but reported as a diagnostic (best.pt was selected
  on val via early-stopping → train-val gap conflates memorization with
  checkpoint-selection bias; it is NOT a clean non-member control).

Run (GPU pod):  cd ~/repos/PhoneFM/workbench && python3 -u mia_audit.py
"""
from __future__ import annotations

import contextlib
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))
from phonefm_dataset_v3 import (  # noqa: E402
    PhoneFMV3Dataset,
    collate_v3,
    report_label_distribution,
)
from phonefm_model_v3 import PhoneFMV3, PhoneFMV3Config  # noqa: E402

DATA_DIR = Path("/home/jupyter/workspace/phonefm-data/tokenized_v3")
MODEL_DIR = Path("/home/jupyter/workspace/phonefm-data/phonefm_v3")
OUT_PATH = MODEL_DIR / "mia_audit.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 20260614
BATCH_SIZE = 128
MIN_POS_FOR_TRAINING = 50
BOOTSTRAP_N = 2000
CI_LEVEL = 0.95
PRIMARY_HEAD = "cv_composite_30d"

N_PERSONS_PER_GROUP = 1300  # bound by smaller arm; test ~1390, val ~1356
K_WINDOWS_PER_PERSON = 8    # ~10,400 windows per group, balanced across groups

TRAIN_GLOB = str(DATA_DIR / "train_*.parquet")
TEST_GLOB = str(DATA_DIR / "test_*.parquet")
VAL_GLOB = str(DATA_DIR / "val_*.parquet")


def build_weights(head_names: list[str]) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Reconstruct POS_WEIGHTS and HEAD_WEIGHTS from train stats."""
    stats = report_label_distribution(TRAIN_GLOB)
    pos_w = {n: stats[n]["pos_weight_for_bce"] for n in head_names}
    head_w = {
        n: (1.0 if stats[n]["n_pos"] >= MIN_POS_FOR_TRAINING else 0.0)
        for n in head_names
    }
    active = [n for n, w in head_w.items() if w > 0.0]
    if not active:
        raise RuntimeError(
            f"No active heads after MIN_POS_FOR_TRAINING={MIN_POS_FOR_TRAINING}. "
            f"Audit cannot proceed."
        )
    if PRIMARY_HEAD not in active:
        raise RuntimeError(
            f"PRIMARY_HEAD={PRIMARY_HEAD!r} is not in active heads: {active}. "
            f"Audit cannot proceed."
        )
    return pos_w, head_w, active


def build_per_person_index(dataset: PhoneFMV3Dataset) -> dict[int, list[int]]:
    """Map person_id -> list of global dataset indices (one Python pass over dataset.index)."""
    shard_pids = [frame["person_id"].to_numpy() for frame in dataset.frames]
    per_person: dict[int, list[int]] = {}
    for gi, (si, ri) in enumerate(dataset.index):
        pid = int(shard_pids[si][ri])
        per_person.setdefault(pid, []).append(gi)
    return per_person


def sample_persons_and_rows(
    dataset: PhoneFMV3Dataset,
    n_persons: int,
    k_windows: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Sample n_persons persons, then ≤k_windows windows per person.

    Returns (row_indices, person_id_per_row) — both length len(row_indices).
    """
    per_person = build_per_person_index(dataset)
    all_pids = sorted(per_person.keys())
    n_take = min(n_persons, len(all_pids))
    chosen_pids = rng.choice(np.array(all_pids), size=n_take, replace=False)
    row_indices: list[int] = []
    pid_per_row: list[int] = []
    for pid in chosen_pids:
        pid_i = int(pid)
        rows = per_person[pid_i]
        if len(rows) > k_windows:
            picked = rng.choice(np.array(rows), size=k_windows, replace=False)
        else:
            picked = np.array(rows)
        for r in picked:
            row_indices.append(int(r))
            pid_per_row.append(pid_i)
    return row_indices, pid_per_row


@torch.no_grad()
def collect_losses(
    model: PhoneFMV3,
    dataset: PhoneFMV3Dataset,
    row_indices: list[int],
    pid_per_row: list[int],
    cfg: PhoneFMV3Config,
    pos_weights: dict[str, float],
    head_weights: dict[str, float],
    active: list[str],
) -> dict:
    """Run forward pass over a pre-selected subset; return per-window losses + pids."""
    subset = Subset(dataset, row_indices)
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_v3(b, max_seq_len=cfg.max_seq_len),
        pin_memory=(DEVICE == "cuda"),
    )

    primary_losses: list[np.ndarray] = []
    multihead_losses: list[np.ndarray] = []
    pids_seen: list[np.ndarray] = []

    pw_primary = torch.tensor(pos_weights[PRIMARY_HEAD], device=DEVICE, dtype=torch.float32)

    autocast_ctx: contextlib.AbstractContextManager = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)  # type: ignore[attr-defined]
        if DEVICE == "cuda" else contextlib.nullcontext()
    )

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        token_types = batch["token_types"].to(DEVICE)
        day_index = batch["day_index"].to(DEVICE)
        wearable = batch["wearable_feats"].to(DEVICE)
        confounders = batch["confounders"].to(DEVICE)
        attn_mask = batch["attn_mask"].to(DEVICE)
        with autocast_ctx:
            logits = model(input_ids, token_types, day_index, wearable, confounders, attn_mask)

        # Primary: single-head BCE for cv_composite_30d (fp32)
        logit_p = logits[PRIMARY_HEAD].float().reshape(-1)
        label_p = batch["labels"][PRIMARY_HEAD].to(DEVICE).float().reshape(-1)
        mask_p = batch["masks"][PRIMARY_HEAD].to(DEVICE).float().reshape(-1)
        bce_p = F.binary_cross_entropy_with_logits(
            logit_p, label_p, pos_weight=pw_primary, reduction="none"
        )
        loss_p = torch.where(
            mask_p > 0,
            bce_p,
            torch.full_like(bce_p, float("nan")),
        ).cpu().numpy()

        # Secondary: sum_h [hw * bce * mask] over active heads (no per-window denom)
        B = input_ids.shape[0]
        mh = torch.zeros(B, device=DEVICE, dtype=torch.float32)
        any_active = torch.zeros(B, device=DEVICE, dtype=torch.float32)
        for name in active:
            label_h = batch["labels"][name].to(DEVICE).float().reshape(-1)
            mask_h = batch["masks"][name].to(DEVICE).float().reshape(-1)
            pw_h = torch.tensor(pos_weights[name], device=DEVICE, dtype=torch.float32)
            bce_h = F.binary_cross_entropy_with_logits(
                logits[name].float().reshape(-1), label_h, pos_weight=pw_h, reduction="none"
            )
            hw = head_weights[name]
            mh += hw * bce_h * mask_h
            any_active += hw * mask_h
        loss_mh = torch.where(
            any_active > 0, mh, torch.full_like(mh, float("nan"))
        ).cpu().numpy()

        primary_losses.append(loss_p)
        multihead_losses.append(loss_mh)
        pids_seen.append(batch["person_id"].numpy())

    primary_arr = np.concatenate(primary_losses)
    multihead_arr = np.concatenate(multihead_losses)
    pid_arr = np.concatenate(pids_seen)
    expected_pid = np.array(pid_per_row, dtype=pid_arr.dtype)
    if not np.array_equal(pid_arr, expected_pid):
        raise RuntimeError("person_id order from loader does not match sampling plan")

    return {"primary": primary_arr, "multihead": multihead_arr, "pid": pid_arr}


def auroc_lower_is_member(member: np.ndarray, nonmember: np.ndarray) -> float:
    y = np.concatenate([np.ones(len(member)), np.zeros(len(nonmember))])
    s = np.concatenate([-member, -nonmember])
    return float(roc_auc_score(y, s))


def per_person_means(loss: np.ndarray, pid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-person mean, ignoring NaN windows. Returns (means, unique_pids)."""
    df = pd.DataFrame({"pid": pid, "loss": loss}).dropna()
    g = df.groupby("pid", sort=True)["loss"].mean()
    return g.to_numpy(), g.index.to_numpy()


def person_bootstrap_ci(
    member_loss: np.ndarray, member_pid: np.ndarray,
    nonmember_loss: np.ndarray, nonmember_pid: np.ndarray,
    n_boot: int, ci: float, seed: int,
) -> dict:
    """Person-cluster bootstrap CI on (-loss -> member) AUROC."""
    rng = np.random.default_rng(seed)
    m_means, _ = per_person_means(member_loss, member_pid)
    n_means, _ = per_person_means(nonmember_loss, nonmember_pid)
    if len(m_means) == 0 or len(n_means) == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_persons_member": int(len(m_means)),
                "n_persons_nonmember": int(len(n_means)),
                "n_boot": n_boot}
    point = auroc_lower_is_member(m_means, n_means)
    boots = np.empty(n_boot, dtype=np.float64)
    Pm, Pn = len(m_means), len(n_means)
    for b in range(n_boot):
        rm = rng.integers(0, Pm, size=Pm)
        rn = rng.integers(0, Pn, size=Pn)
        boots[b] = auroc_lower_is_member(m_means[rm], n_means[rn])
    alpha = (1 - ci) / 2
    return {
        "point": point,
        "lo": float(np.quantile(boots, alpha)),
        "hi": float(np.quantile(boots, 1 - alpha)),
        "n_persons_member": Pm,
        "n_persons_nonmember": Pn,
        "n_boot": n_boot,
    }


def summarize_arm(name: str, data: dict) -> dict:
    p_means, _ = per_person_means(data["primary"], data["pid"])
    mh_means, _ = per_person_means(data["multihead"], data["pid"])
    return {
        "name": name,
        "n_windows": int(len(data["pid"])),
        "n_unique_persons": int(len(np.unique(data["pid"]))),
        "n_primary_valid_windows": int(np.sum(~np.isnan(data["primary"]))),
        "n_multihead_valid_windows": int(np.sum(~np.isnan(data["multihead"]))),
        "mean_primary_loss_window": float(np.nanmean(data["primary"])),
        "mean_primary_loss_person": float(np.mean(p_means)) if len(p_means) else float("nan"),
        "mean_multihead_loss_window": float(np.nanmean(data["multihead"])),
        "mean_multihead_loss_person": float(np.mean(mh_means)) if len(mh_means) else float("nan"),
    }


def main():
    print(f"device={DEVICE}  out={OUT_PATH}", flush=True)
    print(f"seed={SEED}  n_persons/group={N_PERSONS_PER_GROUP}  k_windows/person={K_WINDOWS_PER_PERSON}",
          flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with open(MODEL_DIR / "config.json") as f:
        saved = json.load(f)
    cfg_d = saved["config"].copy()
    cfg_d["head_specs"] = tuple(tuple(s) for s in cfg_d["head_specs"])
    cfg = PhoneFMV3Config(**cfg_d)
    model = PhoneFMV3(cfg).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_DIR / "best.pt", map_location=DEVICE), strict=True
    )
    model.eval()
    print(f"model loaded: {model.num_params() / 1e6:.1f}M params  heads={len(model.head_names)}",
          flush=True)

    pos_w, head_w, active = build_weights(model.head_names)
    print(f"active heads ({len(active)}): {active}", flush=True)

    # Sampling plans (deterministic given SEED). Per-arm seeds so arm order
    # is interchangeable (rerun-robust if the loop is ever reordered).
    # Smallest arms first to keep peak RAM down on T4 pod.
    arm_seeds = {"test": SEED + 10, "val": SEED + 20, "train": SEED + 30}
    arms = {}
    for arm_name, glob in [("test", TEST_GLOB), ("val", VAL_GLOB), ("train", TRAIN_GLOB)]:
        print(f"\n[{arm_name}] building dataset + sampling plan (seed={arm_seeds[arm_name]})...",
              flush=True)
        ds = PhoneFMV3Dataset(
            glob, max_seq_len=cfg.max_seq_len, n_confounders=cfg.n_confounders
        )
        plan_rng = np.random.default_rng(arm_seeds[arm_name])
        rows, pid_row = sample_persons_and_rows(
            ds, N_PERSONS_PER_GROUP, K_WINDOWS_PER_PERSON, plan_rng
        )
        print(f"[{arm_name}] sampled {len(set(pid_row))} persons -> {len(rows)} windows; "
              f"running forward pass...", flush=True)
        data = collect_losses(model, ds, rows, pid_row, cfg, pos_w, head_w, active)
        arms[arm_name] = data
        print(f"[{arm_name}] done: "
              f"primary valid={np.sum(~np.isnan(data['primary']))} / {len(data['primary'])}, "
              f"multihead_score valid={np.sum(~np.isnan(data['multihead']))} / {len(data['multihead'])}",
              flush=True)
        del ds
        gc.collect()

    print(f"\nComputing person-cluster bootstrap CIs (n_boot={BOOTSTRAP_N})...", flush=True)
    results = {
        "metric_primary_train_vs_test": person_bootstrap_ci(
            arms["train"]["primary"], arms["train"]["pid"],
            arms["test"]["primary"], arms["test"]["pid"],
            BOOTSTRAP_N, CI_LEVEL, SEED,
        ),
        "metric_primary_train_vs_val_DIAGNOSTIC": person_bootstrap_ci(
            arms["train"]["primary"], arms["train"]["pid"],
            arms["val"]["primary"], arms["val"]["pid"],
            BOOTSTRAP_N, CI_LEVEL, SEED + 1,
        ),
        "metric_multihead_train_vs_test": person_bootstrap_ci(
            arms["train"]["multihead"], arms["train"]["pid"],
            arms["test"]["multihead"], arms["test"]["pid"],
            BOOTSTRAP_N, CI_LEVEL, SEED + 2,
        ),
        "metric_multihead_train_vs_val_DIAGNOSTIC": person_bootstrap_ci(
            arms["train"]["multihead"], arms["train"]["pid"],
            arms["val"]["multihead"], arms["val"]["pid"],
            BOOTSTRAP_N, CI_LEVEL, SEED + 3,
        ),
    }

    arm_summaries = {n: summarize_arm(n, arms[n]) for n in ("train", "test", "val")}

    print("\n" + "=" * 78)
    print("MIA AUDIT RESULTS  (numbers only — human picks the verdict threshold)")
    print("=" * 78)
    for arm in ("train", "test", "val"):
        s = arm_summaries[arm]
        print(f"\n[{arm}]  n_windows={s['n_windows']}  n_persons={s['n_unique_persons']}")
        print(f"  primary  (cv_composite_30d single-head BCE):")
        print(f"    per-window mean loss = {s['mean_primary_loss_window']:.4f}")
        print(f"    per-person mean loss = {s['mean_primary_loss_person']:.4f}")
        print(f"  multihead (sum_h hw * bce * mask, no per-window denom):")
        print(f"    per-window mean loss = {s['mean_multihead_loss_window']:.4f}")
        print(f"    per-person mean loss = {s['mean_multihead_loss_person']:.4f}")

    print(f"\nPerson-level MIA AUROC (-loss -> member), {int(CI_LEVEL * 100)}% bootstrap CI:")
    for k, r in results.items():
        tag = "  [DIAGNOSTIC — best.pt selected on val]" if "DIAGNOSTIC" in k else ""
        print(f"  {k}: point={r['point']:.4f}  CI=[{r['lo']:.4f}, {r['hi']:.4f}]"
              f"  (n_persons m={r['n_persons_member']} n={r['n_persons_nonmember']}){tag}")

    out = {
        "version": "v2",
        "seed": SEED,
        "n_persons_per_group_target": N_PERSONS_PER_GROUP,
        "k_windows_per_person": K_WINDOWS_PER_PERSON,
        "batch_size": BATCH_SIZE,
        "min_pos_for_training": MIN_POS_FOR_TRAINING,
        "primary_head": PRIMARY_HEAD,
        "active_heads": active,
        "dropped_heads": [h for h, w in head_w.items() if w == 0.0],
        "pos_weights": pos_w,
        "head_weights": head_w,
        "methodology": {
            "primary_loss_formula": "BCE(logit, label, pos_weight=pw) for cv_composite_30d head only",
            "multihead_score_formula": "sum_h [head_weight_h * BCE_h * mask_h] over active heads (no per-window denominator). NOTE: this is NOT a reconstruction of the training loss — training's per-head batch-level normalizer (1/mask_sum_h) cannot be reconstructed at audit time. Reported as a complementary diagnostic score, not as 'the' training loss.",
            "person_aggregation": "mean across that person's sampled windows, ignoring NaN windows",
            "sampling": f"sample N_PERSONS_PER_GROUP={N_PERSONS_PER_GROUP} persons per arm without replacement, then ≤K_WINDOWS_PER_PERSON={K_WINDOWS_PER_PERSON} windows per person. EXPECTED per-arm window-count imbalance: train participants have many more windows per person than test/val, so test/val may collect fewer than {N_PERSONS_PER_GROUP * K_WINDOWS_PER_PERSON} windows when many of their persons have <{K_WINDOWS_PER_PERSON} windows; see arms.<name>.n_windows for actual counts.",
            "bootstrap": f"person-cluster, n_boot={BOOTSTRAP_N}, two-sided {int(CI_LEVEL*100)}% CI, sampling persons within each arm separately",
            "non_member_arms": {
                "test": "primary — never seen during training or checkpoint selection",
                "val_DIAGNOSTIC": "secondary — best.pt was selected on val via early-stopping; train-vs-val gap conflates memorization with checkpoint-selection bias and is reported as diagnostic only, NOT as a verdict-eligible signal",
            },
            "loss_replication": "bf16 autocast on forward pass matching 05_train_v3.py; logits up-cast to fp32 before BCE",
            "reproducibility": f"per-arm seeds derived from base SEED={SEED}: test={SEED+10}, val={SEED+20}, train={SEED+30}; loader uses num_workers=0, shuffle=False, deterministic sampling plan via Subset",
        },
        "arms": arm_summaries,
        "results": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
