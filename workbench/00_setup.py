"""PhoneFM environment setup — run ONCE per Workbench environment.

Workbench's default Python 3.10 kernel ships with most things we need, but
PyTorch and coremltools are usually too old. Pin everything here so the
training run is reproducible.

After running this cell, restart the kernel before running 01–07.
"""

# ============================================================
# CELL 1 — pinned pip installs
# ============================================================
import subprocess, sys

PINS = [
    # Core ML conversion (iOS 18 target requires >= 7.2)
    "coremltools>=7.2,<8.0",

    # PyTorch — bf16 + stable SDPA require >= 2.2
    "torch>=2.2,<2.6",

    # ML utilities
    "scikit-learn>=1.3",
    "pyarrow>=14.0",

    # Data plumbing (usually present, pinning so Workbench upgrades don't break us)
    "pandas>=2.0",
    "numpy>=1.24",

    # Google Cloud (usually present in Workbench)
    "google-cloud-bigquery>=3.13",
    "google-cloud-storage>=2.13",
]

print("Installing pinned dependencies...")
cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"] + PINS
subprocess.check_call(cmd)
print("Done.\n")


# ============================================================
# CELL 2 — version + capability sanity check
# ============================================================
import importlib

def show(mod_name, attr="__version__"):
    try:
        m = importlib.import_module(mod_name)
        print(f"  {mod_name:30s} {getattr(m, attr, '?')}")
    except Exception as e:
        print(f"  {mod_name:30s} MISSING ({e})")

print("=== package versions ===")
for m in ["torch", "coremltools", "sklearn", "numpy", "pandas",
          "pyarrow", "google.cloud.bigquery", "google.cloud.storage"]:
    show(m)

import torch
print("\n=== compute ===")
print(f"  cuda available    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  cuda device       : {torch.cuda.get_device_name(0)}")
    print(f"  cuda capability   : {torch.cuda.get_device_capability(0)}")
    print(f"  bf16 supported    : {torch.cuda.is_bf16_supported()}")
    print(f"  total memory (GB) : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")
else:
    print("  WARNING: no CUDA. Training will be unusably slow on CPU.")


# ============================================================
# CELL 3 — Workbench environment vars
# ============================================================
import os

required = ["WORKSPACE_CDR", "WORKSPACE_BUCKET"]
print("\n=== workspace env ===")
for k in required:
    v = os.environ.get(k)
    status = "OK" if v else "MISSING"
    print(f"  {k:25s} {status}  {v or ''}")

missing = [k for k in required if not os.environ.get(k)]
if missing:
    raise RuntimeError(
        f"Missing required Workbench env vars: {missing}. "
        "These are normally injected automatically — try restarting the environment."
    )

print("\nSetup OK. Restart the kernel, then run 01_cohort_extraction.py.")
