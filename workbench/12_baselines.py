"""
12_baselines.py — non-FM baselines for the multi-endpoint cardiovascular model,
one-pipeline-comparable with PhoneFM v3 on the held-out test set. Answers the
reviewer question "does the 13.3M-parameter FM beat simple classifiers on
hand-features / tokens / raw signals?"

Three standard wearable-cardio feature representations (no FM), + demographics:
  HAND  — hand-engineered wearable summary stats (per-stream daily mean/SD/min/max
          /quartiles + coverage over the 180-day window). Apple-Heart-Study /
          Fitbit-cardio benchmark style.
  TOK   — bag-of-tokens over the EHR vocabulary (ids >= 4). LogReg sees the FULL
          sparse 6007-dim vocab; HistGB/MLP see the top-2048 (train doc-freq)
          dense, for memory — so no informative rare dx code is dropped from the
          linear baseline.
  RAW7  — last-7-days raw wearable signals (7 x 11) flattened.

Three classifiers per representation: Logistic Regression, Hist Gradient Boosting
(sklearn's XGBoost-equivalent; no PyPI install — the CT perimeter blocks it), MLP.

FAIRNESS / honesty (per adversarial review):
  - Config (rep x model) is SELECTED ON THE VAL split, and only the selected
    config's TEST AUROC is reported as the headline baseline (no test-set
    max-over-9 cherry-pick). The full grid of test AUROCs is also saved.
  - Bootstrap CI matches the FM eval exactly (n_boot, >=5-positive guard, seed).
  - Baselines use fixed light hyperparameters with val-based config selection;
    the reported FM advantage is therefore approximately an UPPER BOUND.
  - Negative-control heads are flagged; AUROC ~0.5 there is the correct outcome
    and FM-minus-baseline deltas are not interpreted for them.

CPU-only. Features extracted once from the tokenised parquet (stays inside the CT
perimeter) and cached to .npz for cheap re-runs.

In:  ~/workspace/phonefm-data/tokenized_v3/{train,val,test}_*.parquet, .../phonefm_v3/config.json
Out: ~/workspace/phonefm-data/phonefm_v3/baselines.json
     ~/workspace/phonefm-data/phonefm_v3/baselines_features_{train,val,test}.npz  (cache, perimeter-internal)
Run (CPU pod):  cd ~/repos/PhoneFM/workbench && python3 -u 12_baselines.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))
from phonefm_dataset_v3 import make_loader_v3
from phonefm_model_v3 import PhoneFMV3Config

# ===========================================================================
DATA_DIR = Path("/home/jupyter/workspace/phonefm-data/tokenized_v3")
MODEL_DIR = Path("/home/jupyter/workspace/phonefm-data/phonefm_v3")
OUT_PATH = MODEL_DIR / "baselines.json"

SEED = 20260612
BOOT_SEED = 20260609          # matches the FM eval (06_eval_v3_test.py) for parity
BATCH_SIZE = 256
NUM_WORKERS = 4
VOCAB_SIZE = 6007
FIRST_CONTENT_ID = 4          # ids 0..3 are PAD/CLS/EOS/wearable-marker
TOPK_TOKENS = 2048            # dense bag-of-tokens kept for HistGB/MLP (by train doc-freq)
N_WEAR_FEATS = 11
N_WEAR_DAYS = 180
N_BOOTSTRAP = 1000            # match FM eval
MIN_BOOT_POS = 5             # match FM eval per-resample guard
CI = 0.95

HEADS = (
    "mortality_30d", "mortality_180d", "mortality_365d",
    "t2d_180d", "t2d_365d", "dep_180d", "dep_365d",
    "cv_composite_30d", "cv_composite_180d", "cv_composite_365d",
    "skin_neoplasm_365d", "refractive_errors_365d", "dental_caries_365d",
)
NEG_CONTROL = frozenset({"skin_neoplasm_365d", "refractive_errors_365d", "dental_caries_365d"})
CI_HEADS = ("cv_composite_30d", "mortality_365d", "t2d_365d", "dep_365d")
REPS = ("hand", "tok", "raw7")
MODELS = ("logreg", "histgb", "mlp")


# ===========================================================================
# Feature extraction (loader with shuffle=False -> windows == what the FM sees)
# ===========================================================================
def _wear_summary(w: np.ndarray) -> np.ndarray:
    """w: (B, 180, 11). Per-feature stats over PRESENT days + coverage -> (B, 78)."""
    present = (np.abs(w).sum(axis=2) > 0)                  # (B, 180)
    wn = np.where(present[:, :, None], w, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)    # all-nan slices
        feats = [np.nanmean(wn, 1), np.nanstd(wn, 1), np.nanmin(wn, 1), np.nanmax(wn, 1),
                 np.nanpercentile(wn, 25, axis=1), np.nanpercentile(wn, 50, axis=1),
                 np.nanpercentile(wn, 75, axis=1)]
    out = np.concatenate(feats + [present.mean(1, keepdims=True)], axis=1)
    return np.nan_to_num(out, nan=0.0)


def extract(split: str, n_confounders: int, max_seq_len: int) -> dict:
    cache = MODEL_DIR / f"baselines_features_{split}.npz"
    meta = [VOCAB_SIZE, FIRST_CONTENT_ID, N_WEAR_DAYS, n_confounders, max_seq_len]
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        fresh = ("meta" in z and list(z["meta"]) == meta
                 and "heads" in z and list(z["heads"]) == list(HEADS))
        if fresh:
            tok = sparse.csr_matrix((z["tok_data"], z["tok_indices"], z["tok_indptr"]),
                                    shape=tuple(z["tok_shape"]))
            d = {"X_hand": z["X_hand"], "X_raw7": z["X_raw7"], "C": z["C"], "tok": tok,
                 "pid": z["pid"], "Y": {h: z[f"Y__{h}"] for h in HEADS},
                 "M": {h: z[f"M__{h}"] for h in HEADS}}
            print(f"  [{split}] loaded cache {cache.name}: {len(d['pid'])} windows", flush=True)
            return d
        print(f"  [{split}] cache {cache.name} stale (params/heads changed) — re-extracting", flush=True)

    torch.manual_seed(SEED)
    loader = make_loader_v3(str(DATA_DIR / f"{split}_*.parquet"), batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS, shuffle=False, max_seq_len=max_seq_len,
                            n_confounders=n_confounders)
    print(f"  [{split}] {len(loader)} batches, {len(loader.dataset)} windows", flush=True)
    X_hand, X_raw7, C, pid = [], [], [], []
    Y = {h: [] for h in HEADS}; M = {h: [] for h in HEADS}
    tok_rows, tok_cols = [], []
    row_off = 0
    for i, batch in enumerate(loader):
        ids = batch["input_ids"].numpy()
        w = batch["wearable_feats"].numpy()[:, 1:1 + N_WEAR_DAYS, :]   # days 1..180
        B = ids.shape[0]
        X_hand.append(_wear_summary(w))
        X_raw7.append(w[:, -7:, :].reshape(B, 7 * N_WEAR_FEATS))
        C.append(batch["confounders"].numpy())
        pid.append(batch["person_id"].numpy())
        for h in HEADS:
            Y[h].append(batch["labels"][h].numpy()); M[h].append(batch["masks"][h].numpy())
        r, c = np.where(ids >= FIRST_CONTENT_ID)
        tok_rows.append(r + row_off); tok_cols.append(ids[r, c]); row_off += B
        if i % 50 == 0:
            print(f"    batch {i}/{len(loader)}", flush=True)
    N = row_off
    cols = np.concatenate(tok_cols)
    tok = sparse.coo_matrix((np.ones(len(cols), np.float32),
                             (np.concatenate(tok_rows), cols)), shape=(N, VOCAB_SIZE)).tocsr()
    tok.data[:] = 1.0    # multi-hot presence (collapse duplicate counts)
    d = {"X_hand": np.concatenate(X_hand).astype(np.float32),
         "X_raw7": np.concatenate(X_raw7).astype(np.float32),
         "C": np.concatenate(C).astype(np.float32), "tok": tok,
         "Y": {h: np.concatenate(Y[h]).astype(np.int64) for h in HEADS},
         "M": {h: np.concatenate(M[h]).astype(bool) for h in HEADS},
         "pid": np.concatenate(pid)}
    np.savez(cache, X_hand=d["X_hand"], X_raw7=d["X_raw7"], C=d["C"], pid=d["pid"],
             tok_data=tok.data, tok_indices=tok.indices, tok_indptr=tok.indptr,
             tok_shape=np.array(tok.shape), meta=np.array(meta), heads=np.array(list(HEADS)),
             **{f"Y__{h}": d["Y"][h] for h in HEADS}, **{f"M__{h}": d["M"][h] for h in HEADS})
    print(f"  [{split}] cached -> {cache.name}", flush=True)
    return d


# ===========================================================================
# Eval helpers (mirror the FM eval exactly)
# ===========================================================================
def auroc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def cluster_ci(y: np.ndarray, s: np.ndarray, pid: np.ndarray) -> dict:
    """Person-cluster bootstrap matching 06_eval_v3_test.py: n_boot, >=MIN_BOOT_POS
    positives (and negatives) per resample, fixed BOOT_SEED."""
    order = np.argsort(pid, kind="stable")
    uniq, starts = np.unique(pid[order], return_index=True)
    groups = np.split(order, starts[1:])
    rng = np.random.RandomState(BOOT_SEED)
    vals = np.full(N_BOOTSTRAP, np.nan)
    for b in range(N_BOOTSTRAP):
        sel = np.concatenate([groups[j] for j in rng.randint(0, len(uniq), len(uniq))])
        yb = y[sel]
        if yb.sum() < MIN_BOOT_POS or (len(yb) - yb.sum()) < MIN_BOOT_POS:
            continue
        vals[b] = roc_auc_score(yb, s[sel])
    n = int(np.count_nonzero(~np.isnan(vals)))
    lo = float(np.nanpercentile(vals, (1 - CI) / 2 * 100)) if n >= 100 else float("nan")
    hi = float(np.nanpercentile(vals, (1 + CI) / 2 * 100)) if n >= 100 else float("nan")
    return {"lo": lo, "hi": hi, "n_bootstrap_valid": n}


def new_model(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                                  solver="lbfgs")
    if name == "histgb":
        return HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.08,
                                              l2_regularization=1.0, random_state=SEED)
    # MLP: early_stopping OFF — on rare heads a positive-free internal val slice
    # would otherwise stop it at a near-chance model (adversarial review).
    return MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=150,
                         early_stopping=False, random_state=SEED)


# ===========================================================================
def main() -> None:
    with open(MODEL_DIR / "config.json") as f:
        saved = json.load(f)
    cfg = PhoneFMV3Config(**{**saved["config"],
                             "head_specs": tuple(tuple(s) for s in saved["config"]["head_specs"])})

    splits = {s: extract(s, cfg.n_confounders, cfg.max_seq_len) for s in ("train", "val", "test")}
    tr, va, te = splits["train"], splits["val"], splits["test"]

    # top-K tokens by TRAIN document-frequency (presence), drop zero-freq, never specials
    dfreq = np.asarray(tr["tok"].sum(axis=0)).ravel()
    dfreq[:FIRST_CONTENT_ID] = -1
    topk = np.argsort(dfreq)[::-1][:TOPK_TOKENS]
    topk = np.sort(topk[dfreq[topk] > 0])
    print(f"bag-of-tokens: dense top-{len(topk)} of {VOCAB_SIZE} (min train doc-freq "
          f"{dfreq[topk].min():.0f}); LogReg uses only train-nonzero token cols (mathematically equivalent)", flush=True)

    sc_hand = StandardScaler().fit(np.hstack([tr["X_hand"], tr["C"]]))
    sc_raw7 = StandardScaler().fit(np.hstack([tr["X_raw7"], tr["C"]]))
    sc_conf = StandardScaler().fit(tr["C"])

    def build(d: dict) -> dict:
        conf_s = sc_conf.transform(d["C"]).astype(np.float32)
        return {
            "hand": sc_hand.transform(np.hstack([d["X_hand"], d["C"]])).astype(np.float32),
            "raw7": sc_raw7.transform(np.hstack([d["X_raw7"], d["C"]])).astype(np.float32),
            "tok_dense": np.hstack([d["tok"][:, topk].toarray(), conf_s]).astype(np.float32),
            "tok_sparse": sparse.hstack([d["tok"][:, topk],
                                         sparse.csr_matrix(conf_s)]).tocsr(),
        }
    Mtr, Mva, Mte = build(tr), build(va), build(te)

    def matrix(Mx: dict, rep: str, model: str):
        if rep == "tok":
            return Mx["tok_sparse"] if model == "logreg" else Mx["tok_dense"]
        return Mx[rep]

    results = {"config": {"seed": SEED, "boot_seed": BOOT_SEED, "n_bootstrap": N_BOOTSTRAP,
                          "topk_tokens": int(len(topk)),
                          "n": {s: int(len(splits[s]["pid"])) for s in splits},
                          "heads": list(HEADS), "ci_heads": list(CI_HEADS),
                          "neg_control_heads": sorted(NEG_CONTROL),
                          "reps": list(REPS), "models": list(MODELS),
                          "selection": "rep x model chosen by VAL AUROC; TEST AUROC reported",
                          "note": "baselines use fixed light hyperparameters; FM-minus-baseline "
                                  "is approximately an UPPER BOUND on the FM advantage. Bag-of-"
                                  "tokens: LogReg=full 6007 sparse, HistGB/MLP=top-2048 dense."},
               "heads": {}}

    print(f"\n{'head':<22} {'sel rep/model':<16} {'VAL':>7} {'TEST':>7}   (full grid TEST AUROC)", flush=True)
    for h in HEADS:
        mtr, mva, mte = tr["M"][h], va["M"][h], te["M"][h]
        ytr, yva, yte = tr["Y"][h][mtr], va["Y"][h][mva], te["Y"][h][mte]
        pid_te = te["pid"][mte]
        rec = {"n_test_pos": int(yte.sum()), "n_test": int(mte.sum()),
               "negative_control": h in NEG_CONTROL, "grid": {}}
        if ytr.sum() < 10 or yte.sum() < MIN_BOOT_POS or len(np.unique(ytr)) < 2:
            rec["skipped"] = True
            results["heads"][h] = rec
            print(f"{h:<22} (skipped: n_pos_tr={int(ytr.sum())}, n_pos_te={int(yte.sum())})", flush=True)
            continue
        best = None   # (val_auroc, rep, model, test_auroc, test_score)
        for rep in REPS:
            for model in MODELS:
                clf = new_model(model)
                clf.fit(matrix(Mtr, rep, model)[mtr], ytr)
                va_a = auroc(yva, clf.predict_proba(matrix(Mva, rep, model)[mva])[:, 1])
                te_s = clf.predict_proba(matrix(Mte, rep, model)[mte])[:, 1]
                te_a = auroc(yte, te_s)
                rec["grid"].setdefault(rep, {})[model] = te_a
                if not np.isnan(va_a) and (best is None or va_a > best[0]):
                    best = (va_a, rep, model, te_a, te_s)
        if best is not None:
            _, rep, model, te_a, te_s = best
            rec["selected"] = {"rep": rep, "model": model,
                               "val_auroc": best[0], "test_auroc": te_a}
            if h in CI_HEADS:
                rec["selected"]["test_ci"] = cluster_ci(yte, te_s, pid_te)
            grid_str = "  ".join(f"{r[:4]}:{max(v for v in rec['grid'][r].values() if not np.isnan(v)):.3f}"
                                 for r in REPS if rec["grid"].get(r))
            print(f"{h:<22} {rep+'/'+model:<16} {best[0]:>7.4f} {te_a:>7.4f}   {grid_str}", flush=True)
        results["heads"][h] = rec

    # FM comparison (best-effort load of the FM eval json)
    fm = None
    for cand in ("eval_test.json", "eval_v3_test.json", "test_eval.json"):
        if (MODEL_DIR / cand).exists():
            fm = json.load(open(MODEL_DIR / cand)); results["fm_eval_source"] = cand; break
    print("\n=== FM vs val-selected baseline (TEST AUROC) ===", flush=True)
    print("(negative controls: AUROC~0.5 is the CORRECT outcome; deltas not interpreted)", flush=True)
    for h in HEADS:
        rec = results["heads"].get(h, {})
        if "selected" not in rec:
            continue
        bl = rec["selected"]["test_auroc"]
        fm_a = _fm_auroc(fm, h)
        if h in NEG_CONTROL:
            print(f"  {h:<22} baseline={bl:.4f}  FM={_fmt(fm_a)}   [negative control]", flush=True)
        else:
            delta = (fm_a - bl) if fm_a is not None else None
            print(f"  {h:<22} baseline={bl:.4f}  FM={_fmt(fm_a)}   FM-baseline="
                  f"{f'{delta:+.4f}' if delta is not None else 'n/a'}", flush=True)
        if fm_a is not None:
            rec["fm_auroc"] = fm_a

    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_san(x) for x in o]
        if isinstance(o, float) and np.isnan(o):
            return None
        return o
    with open(OUT_PATH, "w") as f:
        json.dump(_san(results), f, indent=2)
    print(f"\nsaved {OUT_PATH}", flush=True)


def _fmt(x):
    return f"{x:.4f}" if x is not None else "n/a  "


def _fm_auroc(fm, head):
    if not fm:
        return None
    for container in (fm, fm.get("heads", {}), fm.get("auroc", {})):
        if isinstance(container, dict) and head in container:
            v = container[head]
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, dict):
                for kk in ("auroc", "auroc_test", "test_auroc"):
                    if isinstance(v.get(kk), (int, float)):
                        return float(v[kk])
    return None


if __name__ == "__main__":
    main()
