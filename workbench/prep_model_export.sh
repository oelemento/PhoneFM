#!/usr/bin/env bash
# prep_model_export.sh — stage the MINIMAL model artifact for an AoU / Verily
# Controlled-Tier egress request.
#
# Copies ONLY the weights + config (nothing data-derived) into a clean export/
# folder and prints sizes, sha256, and a parameter manifest, so the egress
# request can document exactly what is leaving: model parameters only, no
# participant-level data.
#
# Run on the pod:  bash ~/repos/PhoneFM/workbench/prep_model_export.sh
set -euo pipefail

MODEL_DIR="${1:-/home/jupyter/workspace/phonefm-data/phonefm_v3}"
EXPORT_DIR="${2:-$HOME/export/phonefm_v3}"

echo ">> source model dir: $MODEL_DIR"
echo ">> clean export dir: $EXPORT_DIR"
rm -rf "$EXPORT_DIR"
mkdir -p "$EXPORT_DIR"

# --- ONLY these leave. No cohort, no tokenized parquet, no predictions. -------
cp "$MODEL_DIR/best.pt"     "$EXPORT_DIR/"
cp "$MODEL_DIR/config.json" "$EXPORT_DIR/"

echo
echo "=== files staged for export ==="
ls -la "$EXPORT_DIR"

echo
echo "=== sha256 (attach to the egress request) ==="
( cd "$EXPORT_DIR" && sha256sum ./* | tee SHA256SUMS.txt )

echo
echo "=== parameter manifest (evidence: weights only, no data/optimizer state) ==="
python3 - "$EXPORT_DIR/best.pt" <<'PY'
import sys, collections, torch
ckpt = torch.load(sys.argv[1], map_location="cpu")
# best.pt is loaded elsewhere via model.load_state_dict(torch.load(best.pt)),
# i.e. it should be a bare state_dict (no optimizer/RNG/epoch).
if not isinstance(ckpt, dict):
    print("WARNING: checkpoint is not a dict — inspect manually before export")
    sys.exit(0)
SUSPECT = ("optimizer", "optimizer_state_dict", "epoch", "scaler",
           "rng_state", "amp", "scheduler")
suspect = [k for k in ckpt if k in SUSPECT]
tensors = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
n_par = sum(v.numel() for v in tensors.values())
dtypes = collections.Counter(str(v.dtype) for v in tensors.values())
print(f"tensor entries : {len(tensors)}")
print(f"total parameters: {n_par:,}  ({n_par/1e6:.2f}M)")
print(f"dtypes         : {dict(dtypes)}")
nonten = [k for k, v in ckpt.items() if not torch.is_tensor(v)]
print(f"non-tensor keys: {nonten if nonten else 'none'}")
if suspect:
    print(f"!! TRAINING STATE PRESENT {suspect} — strip to a weights-only file before export")
else:
    print("OK: parameter tensors only (no optimizer / RNG / data)")
PY

echo
echo "staged at: $EXPORT_DIR"
echo "next: attach $EXPORT_DIR/SHA256SUMS.txt + the manifest above to the AoU/Verily egress request."
echo "(this only STAGES inside the perimeter — the actual egress is the reviewed step.)"
