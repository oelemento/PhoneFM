"""PhoneFM Core ML conversion — best.pt → cardio_fm_v1.mlpackage.

Runs inside Workbench AFTER 06_eval_mia.py passes the export criteria. The
converted artifact ships as part of the All-of-Us model-export request so the
reviewer can audit both the PyTorch state_dict and the on-device-deployable
form together.

Key design choices for ANE compatibility:
  - Fixed input shapes (1, MAX_LEN). Dynamic axes are slower on Apple Neural
    Engine and tank fp16 conversion stability.
  - Manual softmax(QK/√d)V replacing `F.scaled_dot_product_attention` at trace
    time. SDPA → Core ML still has rough edges with bool masks; explicit
    attention converts cleanly and runs on ANE.
  - Float16 weight quantization (not full activation quant) — preserves
    ~all accuracy and halves disk size.
  - Inputs typed as int32 (not bool) since Core ML's bool support is patchy.

Outputs:
  /tmp/phonefm_v1/cardio_fm_v1.mlpackage
  /tmp/phonefm_v1/coreml_parity_report.json   ← max |ΔLogit| vs PyTorch
"""

# ============================================================
# CELL 1 — imports + paths
# ============================================================
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    linear_quantize_weights,
)

from phonefm_model import PhoneFM, PhoneFMConfig, _rotate_half

RUN_DIR = Path("/tmp/phonefm_v1")
MAX_LEN = 4096
DEVICE  = "cpu"  # trace + convert on CPU — Core ML doesn't care about origin device


# ============================================================
# CELL 2 — load trained model
# ============================================================
cfg_dict = json.load(open(RUN_DIR / "config.json"))["config"]
cfg = PhoneFMConfig(**cfg_dict)
model = PhoneFM(cfg).to(DEVICE)
state = torch.load(RUN_DIR / "best.pt", map_location=DEVICE)
model.load_state_dict(state)
model.eval()
print(f"loaded PhoneFM: {model.num_params()/1e6:.1f}M params,  vocab_size={cfg.vocab_size}")


# ============================================================
# CELL 3 — ANE-friendly attention substitute
# ============================================================
def _manual_attention(q, k, v, attn_mask_bool):
    """Drop-in replacement for F.scaled_dot_product_attention(q, k, v, attn_mask).
       q, k, v: [B, H, L, D];  attn_mask_bool: [B, 1, 1, L]  (True = keep)
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L, L]
    # Add -inf where attn_mask_bool is False (key dim masking only — query-side
    # padding is handled by the post-pool mask multiply).
    add = torch.zeros_like(scores)
    add = add.masked_fill(~attn_mask_bool, float("-inf"))
    scores = scores + add
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def _patched_block_forward(self, x, cos, sin, key_padding_mask=None):
    """Mirror of PhoneFM Block.forward but using _manual_attention."""
    B, L, D = x.shape
    h = self.ln1(x)
    q = self.q_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
    # Rotary embedding (re-uses model's apply_rope path)
    cos_h = cos.unsqueeze(1); sin_h = sin.unsqueeze(1)
    q = (q * cos_h) + (_rotate_half(q) * sin_h)
    k = (k * cos_h) + (_rotate_half(k) * sin_h)

    attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
    out = _manual_attention(q, k, v, attn_mask)
    out = out.transpose(1, 2).reshape(B, L, D)
    x = x + self.o_proj(out)             # drop dropout (eval mode, p=0)
    x = x + self.mlp(self.ln2(x))
    return x


# Apply the patch to all blocks BEFORE tracing
for blk in model.blocks:
    blk.forward = _patched_block_forward.__get__(blk, type(blk))


# ============================================================
# CELL 4 — exportable wrapper (int32 mask in, float logit out)
# ============================================================
class PhoneFMExport(nn.Module):
    """Wrap PhoneFM so the Core ML interface uses int32 inputs only."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids: torch.Tensor,
                      positions: torch.Tensor,
                      attn_mask_i: torch.Tensor):
        # attn_mask_i: int32 [B, L], 1 = real token, 0 = pad
        attn_mask = attn_mask_i > 0
        return self.inner(input_ids, positions, attn_mask)


wrapper = PhoneFMExport(model).eval()


# ============================================================
# CELL 5 — JIT trace
# ============================================================
B = 1
ids_ex  = torch.randint(low=0, high=cfg.vocab_size, size=(B, MAX_LEN), dtype=torch.int32)
pos_ex  = torch.randint(low=0, high=32,             size=(B, MAX_LEN), dtype=torch.int32)
mask_ex = torch.cat([torch.ones(B, 2000, dtype=torch.int32),
                     torch.zeros(B, MAX_LEN - 2000, dtype=torch.int32)], dim=1)

with torch.no_grad():
    pt_logit = wrapper(ids_ex.long(), pos_ex.long(), mask_ex).item()
print(f"PyTorch reference logit on example input: {pt_logit:.6f}")

# Trace needs long inputs for embeddings; cast inside the wrapper for Core ML
class PhoneFMExportTraceable(nn.Module):
    def __init__(self, inner): super().__init__(); self.inner = inner
    def forward(self, input_ids, positions, attn_mask_i):
        return self.inner(input_ids.long(), positions.long(), attn_mask_i)

traceable = PhoneFMExportTraceable(wrapper).eval()
with torch.no_grad():
    traced = torch.jit.trace(traceable, (ids_ex, pos_ex, mask_ex), strict=False)
print("JIT trace OK")


# ============================================================
# CELL 6 — convert to Core ML mlprogram (iOS 18 target)
# ============================================================
ml_inputs = [
    ct.TensorType(name="input_ids",  shape=(B, MAX_LEN), dtype=np.int32),
    ct.TensorType(name="positions",  shape=(B, MAX_LEN), dtype=np.int32),
    ct.TensorType(name="attn_mask",  shape=(B, MAX_LEN), dtype=np.int32),
]
ml_outputs = [ct.TensorType(name="risk_logit")]

mlmodel = ct.convert(
    traced,
    inputs=ml_inputs,
    outputs=ml_outputs,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,    # ANE-friendly
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.ALL,
)
print("Core ML conversion OK")


# ============================================================
# CELL 7 — fp16 weight quantization (extra ~2x compression)
# ============================================================
q_cfg = OptimizationConfig(
    global_config=OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=512)
)
try:
    mlmodel_q = linear_quantize_weights(mlmodel, config=q_cfg)
    print("Weight quantization OK")
except Exception as e:
    print(f"Quantization skipped (non-fatal): {e}")
    mlmodel_q = mlmodel


# ============================================================
# CELL 8 — metadata + save
# ============================================================
mlmodel_q.short_description = (
    "PhoneFM cardio-FM v1 — 30-day cardio event risk from 30-day window of "
    "wearable + EHR tokens. Trained on All of Us Fitbit + EHR cohort. "
    "On-device inference only; no participant data embedded."
)
mlmodel_q.author = "PhoneFM team (Elemento / Hodjat / WCM EIPM)"
mlmodel_q.license = "Research preview — not a medical device"
mlmodel_q.version = "1.0"
mlmodel_q.input_description["input_ids"]  = f"Token IDs, int32, length {MAX_LEN}. Pad with 0 (<PAD>)."
mlmodel_q.input_description["positions"]  = "Calendar-day index per token, int32, [0, 31]."
mlmodel_q.input_description["attn_mask"]  = "1 = real token, 0 = padding. int32."
mlmodel_q.output_description["risk_logit"] = "Raw logit. Apply sigmoid to get 30-day cardio event probability."

OUT_PKG = RUN_DIR / "cardio_fm_v1.mlpackage"
mlmodel_q.save(str(OUT_PKG))
print(f"Saved {OUT_PKG}  ({sum(p.stat().st_size for p in OUT_PKG.rglob('*'))/1e6:.1f} MB)")


# ============================================================
# CELL 9 — parity check: Core ML vs PyTorch
# ============================================================
# Compare predictions on N random inputs. Should match within fp16 tolerance.
N_PROBES = 20
deltas = []
for i in range(N_PROBES):
    rng = np.random.RandomState(1000 + i)
    ids  = rng.randint(low=0, high=cfg.vocab_size, size=(B, MAX_LEN)).astype(np.int32)
    pos  = rng.randint(low=0, high=32,             size=(B, MAX_LEN)).astype(np.int32)
    n_real = rng.randint(low=512, high=MAX_LEN)
    mask = np.concatenate([np.ones((B, n_real)), np.zeros((B, MAX_LEN - n_real))], axis=1).astype(np.int32)

    cm_out = mlmodel_q.predict({
        "input_ids": ids, "positions": pos, "attn_mask": mask,
    })["risk_logit"]
    cm_logit = float(np.array(cm_out).reshape(-1)[0])

    with torch.no_grad():
        pt_logit_i = wrapper(torch.from_numpy(ids).long(),
                             torch.from_numpy(pos).long(),
                             torch.from_numpy(mask)).item()
    deltas.append(abs(cm_logit - pt_logit_i))
    if i < 3:
        print(f"  probe {i}: pt={pt_logit_i:+.4f}  cm={cm_logit:+.4f}  Δ={deltas[-1]:.4f}")

parity = {
    "n_probes": N_PROBES,
    "max_abs_delta": float(max(deltas)),
    "mean_abs_delta": float(np.mean(deltas)),
    "p95_abs_delta": float(np.percentile(deltas, 95)),
    "tolerance": 0.05,
    "passes": float(max(deltas)) < 0.05,
}
json.dump(parity, open(RUN_DIR / "coreml_parity_report.json", "w"), indent=2)
print(f"\nParity: max Δ = {parity['max_abs_delta']:.4f}  "
      f"(tol = {parity['tolerance']})  →  {'PASS' if parity['passes'] else 'FAIL'}")
if not parity["passes"]:
    print("  Investigate before shipping: rerun with compute_precision=FLOAT32 to localize the drift.")


# ============================================================
# CELL 10 — sync to workspace bucket (so model-export request can attach it)
# ============================================================
import os
from google.cloud import storage
BUCKET = os.environ["WORKSPACE_BUCKET"]
gcs = storage.Client(); bkt = gcs.bucket(BUCKET.replace("gs://", ""))

# .mlpackage is a directory — upload recursively
def upload_dir(local_dir: Path, remote_prefix: str):
    for p in local_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir)
            bkt.blob(f"{remote_prefix}/{rel}").upload_from_filename(str(p))

upload_dir(OUT_PKG, f"phonefm/checkpoints/v1/cardio_fm_v1.mlpackage")
bkt.blob("phonefm/checkpoints/v1/coreml_parity_report.json").upload_from_filename(
    str(RUN_DIR / "coreml_parity_report.json"))
print("Synced. Attach cardio_fm_v1.mlpackage + coreml_parity_report.json to the export request.")
