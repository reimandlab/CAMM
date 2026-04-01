#!/usr/bin/env python3
"""
Train **six fully independent tree models** (SNV/INDEL × 1mb/100kb/10kb) using a
simple **random 80/20 split**.

Key differences vs the earlier "base at 10kb" script
-----------------------------------------------------
- No shared "base" table that merges all six targets together.
- Each (task, scale) model is built from ONLY:
    - CA/RT features at that same scale
    - mutation target at that same scale
- No split_level / blocked splitting.
- No agg_fn (no cross-scale aggregation is needed).
- No "drop missing targets" logic and no "clip negatives to 0" logic.
  (We *assert* there are no missing/negative values; if there are, we raise.)

Outputs
-------
  <outdir>/baseline_metrics.tsv
  <outdir>/baseline_summary.json
  <outdir>/eval_{task}_{scale}_y_true.npy
  <outdir>/eval_{task}_{scale}_y_pred.npy

Optional (if --save_models):
  <outdir>/models/rf_snv_1mb.joblib   (or xgb_*.json)
  <outdir>/models/features_snv_1mb.txt

Example
-------
python run_baseline_tree_models_randomsplit_independent.py \
  --ca_1mb CA_RT_1MB.tsv \
  --ca_100kb CA_RT_100KB.tsv \
  --ca_10kb CA_RT_10KB.tsv \
  --snv_1mb HMF_snv_1MB.csv \
  --snv_100kb HMF_snv_100KB.csv \
  --snv_10kb HMF_snv_10KB.csv \
  --indel_1mb HMF_indel_1MB.csv \
  --indel_100kb HMF_indel_100KB.csv \
  --indel_10kb HMF_indel_10KB.csv \
  --ctype esophagus \
  --model rf \
  --val_frac 0.2 \
  --seed 42 \
  --outdir results/esophagus_rf_random_independent \
  --save_models
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import joblib  # type: ignore
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False


SCALES = ("1mb", "100kb", "10kb")
TASKS = ("snv", "indel")


def _canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    return df


def _read_ca(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="infer", low_memory=False)


def _read_mut(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="infer", low_memory=False)


def _normalize_chr_start(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["chr"] = df["chr"].astype(str)
    df["start"] = pd.to_numeric(df["start"], errors="raise").astype("int64")
    df["end"] = pd.to_numeric(df["end"], errors="raise").astype("int64")
    return df


def _select_ct(df_mut: pd.DataFrame, ctype: str) -> pd.DataFrame:
    c = ctype.lower()
    if c not in df_mut.columns:
        raise KeyError(
            f"Cancer type '{ctype}' not in mutation table columns. "
            f"Example columns: {df_mut.columns.tolist()[:15]}"
        )
    out = df_mut[["chr", "start","end", c]].rename(columns={c: "y"})
    return out

    
def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("chr", "start", "end")]


def _assert_no_missing_or_negative(X: np.ndarray, y: np.ndarray, tag: str) -> None:
    if not np.isfinite(X).all():
        bad = int((~np.isfinite(X)).sum())
        raise ValueError(f"[{tag}] Found {bad} non-finite values in X (NaN/inf).")

    if not np.isfinite(y).all():
        bad = int((~np.isfinite(y)).sum())
        raise ValueError(f"[{tag}] Found {bad} non-finite values in y (NaN/inf).")

    if (y < 0).any():
        bad = int((y < 0).sum())
        mn = float(y.min())
        raise ValueError(f"[{tag}] Found {bad} negative y values (min={mn}).")


def _zscore_inplace(X: np.ndarray) -> None:
    # assumes no NaN/inf
    m = X.mean(axis=0, keepdims=True)
    s = X.std(axis=0, keepdims=True)
    s[s < 1e-12] = 1.0
    X -= m
    X /= s


def _seed_for_model(base_seed: int, task: str, scale: str) -> int:
    # stable across runs (avoid Python's randomized hash)
    h = hashlib.md5(f"{task}:{scale}".encode("utf-8")).hexdigest()
    off = int(h[:8], 16)  # 32-bit offset
    return int((base_seed + off) % (2**32 - 1))


def random_row_split(n_rows: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if n_rows < 2:
        raise ValueError(f"Need at least 2 rows to split; got n={n_rows}")
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0,1); got {val_frac}")

    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows)
    rng.shuffle(idx)
    n_val = int(np.round(n_rows * val_frac))
    n_val = max(1, min(n_rows - 1, n_val))

    val_idx = idx[:n_val]
    is_val = np.zeros(n_rows, dtype=bool)
    is_val[val_idx] = True
    is_tr = ~is_val
    return is_tr, is_val


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pm": float(np.mean(y_pred)),
        "ps": float(np.std(y_pred)),
        "n": int(len(y_true)),
    }


def fit_predict(model_name: str, X_tr, y_tr, X_val, xgb_objective: str) -> tuple[np.ndarray, object]:
    if model_name == "rf":
        model = RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=0,
            oob_score=False,
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        return pred, model

    if model_name == "xgb":
        if not _HAS_XGB:
            raise RuntimeError("xgboost is not installed in this environment.")
        model = xgb.XGBRegressor(
            n_estimators=800,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective=xgb_objective,
            tree_method="hist",
            n_jobs=-1,
            random_state=0,
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        return pred, model

    raise ValueError(model_name)


def _model_filename(model_kind: str, task: str, scale: str) -> str:
    if model_kind == "rf":
        return f"rf_{task}_{scale}.joblib"
    if model_kind == "xgb":
        return f"xgb_{task}_{scale}.json"
    return f"model_{task}_{scale}.bin"


def save_model(model: object, model_kind: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if model_kind == "rf":
        if not _HAS_JOBLIB:
            raise RuntimeError("joblib not installed; cannot save RF model.")
        joblib.dump(model, out_path)
        return

    if model_kind == "xgb":
        # XGBoost has a native serializer
        model.save_model(str(out_path))  # type: ignore[attr-defined]
        return

    raise ValueError(model_kind)


def load_all_inputs(args) -> tuple[dict[str, pd.DataFrame], dict[tuple[str, str], pd.DataFrame]]:
    """Load CA (per scale) and mutation targets (per task,scale)."""
    ca = {}
    for scale, p in [("1mb", args.ca_1mb), ("100kb", args.ca_100kb), ("10kb", args.ca_10kb)]:
        df = _canon_cols(_read_ca(Path(p)))
        if not {"chr", "start"}.issubset(df.columns):
            raise KeyError(f"CA_{scale} must contain 'chr' and 'start'.")
        df = _normalize_chr_start(df)
        ca[scale] = df

    mut = {}
    for task, scale, p in [
        ("snv", "1mb", args.snv_1mb),
        ("snv", "100kb", args.snv_100kb),
        ("snv", "10kb", args.snv_10kb),
        ("indel", "1mb", args.indel_1mb),
        ("indel", "100kb", args.indel_100kb),
        ("indel", "10kb", args.indel_10kb),
    ]:
        df = _canon_cols(_read_mut(Path(p)))
        if not {"chr", "start"}.issubset(df.columns):
            raise KeyError(f"{task.upper()}_{scale} must contain 'chr' and 'start'.")
        df = _normalize_chr_start(df)
        df = _select_ct(df, args.ctype)
        mut[(task, scale)] = df

    return ca, mut


def main() -> None:
    ap = argparse.ArgumentParser()

    # inputs
    ap.add_argument("--ca_1mb", required=True)
    ap.add_argument("--ca_100kb", required=True)
    ap.add_argument("--ca_10kb", required=True)

    ap.add_argument("--snv_1mb", required=True)
    ap.add_argument("--snv_100kb", required=True)
    ap.add_argument("--snv_10kb", required=True)

    ap.add_argument("--indel_1mb", required=True)
    ap.add_argument("--indel_100kb", required=True)
    ap.add_argument("--indel_10kb", required=True)

    ap.add_argument("--ctype", required=True)

    # preprocessing
    ap.add_argument("--zscore", action="store_true", help="Z-score features within each model (train+val combined).")

    # model
    ap.add_argument("--model", choices=["rf", "xgb"], default="rf")
    ap.add_argument(
        "--xgb_objective",
        choices=["reg:squarederror", "count:poisson"],
        default="reg:squarederror",
    )

    # split
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # output
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--save_models", action="store_true")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    models_dir = outdir / "models"
    if args.save_models:
        models_dir.mkdir(parents=True, exist_ok=True)

    # Load everything once
    ca, mut = load_all_inputs(args)

    rows = []
    summary = {
        "ctype": args.ctype,
        "model": args.model,
        "split": "random_row",
        "val_frac": args.val_frac,
        "seed": args.seed,
        "metrics": {"snv": {}, "indel": {}},
    }

    for task in TASKS:
        for scale in SCALES:
            tag = f"{task}:{scale}"

            ca_df = ca[scale]
            y_df = mut[(task, scale)]

            # Merge (left join to avoid accidental dropping; then assert no missing)
            merged = ca_df.merge(y_df, on=["chr", "start"], how="left", validate="one_to_one")

            feat_cols = _feature_cols(ca_df)
            if len(feat_cols) == 0:
                raise ValueError(f"[{tag}] No feature columns found in CA_{scale} (columns: {ca_df.columns.tolist()[:20]}).")

            X = merged[feat_cols].to_numpy(np.float32, copy=False)
            y = merged["y"].to_numpy(np.float32, copy=False)

            _assert_no_missing_or_negative(X, y, tag)

            if args.zscore:
                _zscore_inplace(X)

            model_seed = _seed_for_model(args.seed, task, scale)
            is_tr, is_val = random_row_split(len(y), val_frac=args.val_frac, seed=model_seed)

            X_tr, X_val = X[is_tr], X[is_val]
            y_tr, y_val = y[is_tr].astype(np.float64), y[is_val].astype(np.float64)

            y_pred_val, model = fit_predict(args.model, X_tr, y_tr, X_val, xgb_objective=args.xgb_objective)
            y_pred_val = y_pred_val.astype(np.float64)

            m = metrics(y_val, y_pred_val)

            rows.append(
                {
                    "ctype": args.ctype,
                    "task": task,
                    "scale": scale,
                    "model": args.model,
                    "split": "random_row",
                    "val_frac": args.val_frac,
                    "seed": args.seed,
                    "model_seed": model_seed,
                    "n_total": int(len(y)),
                    **m,
                }
            )
            summary["metrics"][task][scale] = {k: m[k] for k in ["mae", "mse", "r2", "pm", "ps", "n"]}

            np.save(outdir / f"eval_{task}_{scale}_y_true.npy", y_val)
            np.save(outdir / f"eval_{task}_{scale}_y_pred.npy", y_pred_val)

            if args.save_models:
                model_path = models_dir / _model_filename(args.model, task, scale)
                save_model(model, args.model, model_path)
                (models_dir / f"features_{task}_{scale}.txt").write_text("\n".join(map(str, feat_cols)) + "\n")

            print(
                f"[{tag}] n_total={len(y)}  val_n={m['n']}  R2={m['r2']:.4f}  seed={model_seed}"
            )

    dfm = pd.DataFrame(rows)
    dfm.to_csv(outdir / "baseline_metrics.tsv", sep="\t", index=False)

    # Convenient primary score (mean SNV R² over scales)
    try:
        prim = float(np.mean([summary["metrics"]["snv"][s]["r2"] for s in SCALES]))
    except Exception:
        prim = float("nan")
    summary["primary_task"] = "snv"
    summary["primary_scale"] = "all_mean"
    summary["primary_score"] = prim

    with open(outdir / "baseline_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[OK] Wrote: {outdir/'baseline_metrics.tsv'} and {outdir/'baseline_summary.json'}")
    print(f"Primary (SNV all_mean R²): {prim:.4f}")
    if args.save_models:
        print(f"[OK] Saved models under: {models_dir}")


if __name__ == "__main__":
    main()
