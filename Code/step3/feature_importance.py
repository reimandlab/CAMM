#!/usr/bin/env python3
"""
Feature importance for hierarchical multi-task model,
focused on SNV at 10kb, using *all* 10kb windows that have SNV counts.

This script:
  - Builds a full 10kb window grid from CA/RT (tcga_atac_with_repliseq.*.tsv)
  - Joins SNV/INDEL counts at all scales but does NOT drop rows for missing
    1MB / 100KB targets.
  - Uses the subset of windows with all 6 targets present only to:
        * approximate the training scaling (z-score) statistics
        * compute mu_caps (p99.9 * 2) per task/scale
    but predictions and importance are evaluated on the full set of
    10kb SNV windows.
  - Loads best_model.pt from a final_version best model directory.
  - Computes:
        * Baseline R² for SNV 10kb over all available windows
        * Permutation importance for 10kb features (CA/RT at 10kb)
          w.r.t. SNV 10kb R²
        * Optional SHAP-style attributions for SNV 10kb (Captum Shapley).

Outputs under:
  <model_outdir>/importance_10kb_allwindows/

  - snv_10kb_baseline.tsv
      chr, start, y_true_snv_10kb, y_pred_snv_10kb
  - permutation_importance_10kb.tsv
  - permutation_importance_10kb_significant.tsv
  - permutation_importance_10kb_nonsignificant.tsv
  - shap_summary_10kb.tsv          (if captum installed)
  - shap_checks_10kb.tsv
  - shap_dependency_10kb.tsv
  - shap_long_10kb.tsv.gz          (optional)
  - shap_beeswarm_10kb.tsv.gz      (optional)

NOTE:
  Runtime for permutation importance can be large.
  You can subsample windows with --max_windows_for_perm
  and/or reduce --permutation_repeats.
"""

from __future__ import annotations
import argparse, re, gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

import importlib.util

def load_training_module(model_outdir: Path):
    local = model_outdir / "run_model_hier_multi.py"
    if local.exists():
        spec = importlib.util.spec_from_file_location("train_mod", str(local))
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        print(f"[Import] Using model code from: {local}")
        return mod

    import Code.step1.run_model_hier_multi as mod
    print("[Import] Using model code from PYTHONPATH: run_model_hier_multi.py")
    return mod


# Optional Captum for SHAP
try:
    from captum.attr import ShapleyValueSampling
    _HAS_CAPTUM = True
except Exception:
    _HAS_CAPTUM = False


# ---------------------- Helpers ----------------------


def _feat_cols(df: pd.DataFrame) -> List[str]:
    """Feature columns are everything except chr/start and derived coordinate columns."""
    return [c for c in df.columns if c not in ("chr", "start", "start_100kb", "start_1mb")]


def _pos_clip(a: np.ndarray) -> np.ndarray:
    """Clip to non-negative and replace NaN/inf by 0."""
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(a, 0.0, None)


def _zscore_with_mask(A: np.ndarray, mask: np.ndarray) -> None:
    """
    In-place z-scoring for A, using mean/std estimated only on rows where mask is True
    (approximate training normalization).
    """
    A_masked = A[mask]
    m = np.nanmean(A_masked, axis=0, keepdims=True)
    s = np.nanstd(A_masked, axis=0, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    A -= m
    A /= s


def _spearman_from_numpy(a: np.ndarray, b: np.ndarray) -> float:
    # Spearman via rank + Pearson
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------- Data prep ----------------------


def prepare_all_windows_10kb(
    ca_1mb: Path,
    ca_100kb: Path,
    ca_10kb: Path,
    snv_1mb: Path,
    snv_100kb: Path,
    snv_10kb: Path,
    indel_1mb: Path,
    indel_100kb: Path,
    indel_10kb: Path,
    ctype: str,
    zscore: bool = True,
    feature_clip: Optional[float] = None,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,  # X1, X2, X3
    np.ndarray, np.ndarray,             # Y_snv_10kb_all, Y_indel_10kb_all
    pd.DataFrame,                       # coords (chr,start)
    List[str], List[str], List[str],    # C1, C2, C3
    Dict[str, Dict[str, float]],        # mu_caps
]:
    """
    Build full 10kb window grid and features, plus targets at all scales.

    We:
      - Start from CA 10kb grid (tcga_atac_with_repliseq.10kb.tsv).
      - Attach 100kb and 1MB CA by coordinate.
      - Attach SNV and INDEL at all scales.
      - DO NOT drop rows where some targets are missing.
      - Use the subset with all six targets present only for z-score stats
        and mu_caps computation.

    Returns:
      X1, X2, X3        : feature matrices for 1MB, 100KB, 10KB
      Y_snv_10kb_all    : SNV 10kb targets for all windows (NaN if missing)
      Y_indel_10kb_all  : INDEL 10kb targets for all windows (NaN if missing)
      coords            : DataFrame with chr, start
      C1, C2, C3        : feature names per scale
      mu_caps           : dict[task][scale] of caps (p99.9*2, min 10)
    """
    # --- Read CA ---
    ca1 = _canon_cols(_read_ca(ca_1mb))
    ca2 = _canon_cols(_read_ca(ca_100kb))
    ca3 = _canon_cols(_read_ca(ca_10kb))

    # --- Read mutation tables ---
    s1 = _canon_cols(_read_mut(snv_1mb))
    s2 = _canon_cols(_read_mut(snv_100kb))
    s3 = _canon_cols(_read_mut(snv_10kb))
    i1 = _canon_cols(_read_mut(indel_1mb))
    i2 = _canon_cols(_read_mut(indel_100kb))
    i3 = _canon_cols(_read_mut(indel_10kb))

    for df, name in [
        (ca1, "CA_1MB"), (ca2, "CA_100KB"), (ca3, "CA_10KB"),
        (s1, "SNV_1MB"), (s2, "SNV_100KB"), (s3, "SNV_10KB"),
        (i1, "INDEL_1MB"), (i2, "INDEL_100KB"), (i3, "INDEL_10KB"),
    ]:
        if not {"chr", "start"}.issubset(df.columns):
            raise KeyError(f"{name} must contain 'chr' and 'start' columns")

    # --- Base grid from 10kb CA ---
    base = ca3[["chr", "start"]].copy()
    base["start"] = pd.to_numeric(base["start"], errors="coerce").fillna(0).astype("int64")
    base["start_100kb"] = ((base["start"] - 1) // 100_000) * 100_000 + 1
    base["start_1mb"]   = ((base["start"] - 1) // 1_000_000) * 1_000_000 + 1

    # --- Attach CA features ---
    C1 = _feat_cols(ca1)
    C2 = _feat_cols(ca2)
    C3 = _feat_cols(ca3)

    base = base.merge(
        ca1.rename(columns={"start": "start_1mb"}),
        on=["chr", "start_1mb"],
        how="left",
        suffixes=("", "_ca1"),
    )
    base = base.merge(
        ca2.rename(columns={"start": "start_100kb"}),
        on=["chr", "start_100kb"],
        how="left",
        suffixes=("", "_ca2"),
    )
    base = base.merge(
        ca3,
        on=["chr", "start"],
        how="left",
        suffixes=("", "_ca3"),
    )

    #--- Attach mutation targets at all scales ---
    y_snv_1 = _select_ct(s1, ctype).rename(columns={"y": "y_snv_1mb"})
    y_snv_2 = _select_ct(s2, ctype).rename(columns={"y": "y_snv_100kb"})
    y_snv_3 = _select_ct(s3, ctype).rename(columns={"y": "y_snv_10kb"})
    y_ind_1 = _select_ct(i1, ctype).rename(columns={"y": "y_ind_1mb"})
    y_ind_2 = _select_ct(i2, ctype).rename(columns={"y": "y_ind_100kb"})
    y_ind_3 = _select_ct(i3, ctype).rename(columns={"y": "y_ind_10kb"})

    base = base.merge(
        y_snv_1.rename(columns={"start": "start_1mb"}),
        on=["chr", "start_1mb"],
        how="left",
    )
    base = base.merge(
        y_snv_2.rename(columns={"start": "start_100kb"}),
        on=["chr", "start_100kb"],
        how="left",
    )
    base = base.merge(y_snv_3, on=["chr", "start"], how="left")

    base = base.merge(
        y_ind_1.rename(columns={"start": "start_1mb"}),
        on=["chr", "start_1mb"],
        how="left",
    )
    base = base.merge(
        y_ind_2.rename(columns={"start": "start_100kb"}),
        on=["chr", "start_100kb"],
        how="left",
    )
    base = base.merge(y_ind_3, on=["chr", "start"], how="left")

    print(f"Total 10kb windows in base grid: {len(base):,}")

    # --- Training-mask: windows with all 6 targets present (approx training subset) ---
    need = ["y_snv_1mb", "y_snv_100kb", "y_snv_10kb",
            "y_ind_1mb", "y_ind_100kb", "y_ind_10kb"]
    mask_all = base[need].notna().all(axis=1).to_numpy()
    n_all = int(mask_all.sum())
    frac = n_all / len(base) if len(base) > 0 else 0.0
    print(f"Windows with ALL six targets present (training-like subset): "
          f"{n_all}/{len(base)} ({frac:.4f})")

    # --- Feature matrices ---
    X1 = base[C1].to_numpy(np.float32, copy=False)
    X2 = base[C2].to_numpy(np.float32, copy=False)
    X3 = base[C3].to_numpy(np.float32, copy=False)

    _sanitize_inplace(X1, fill=0.0)
    _sanitize_inplace(X2, fill=0.0)
    _sanitize_inplace(X3, fill=0.0)

    # z-score using training-like subset, then clip
    if zscore:
        print("Applying z-score normalization (stats from all-target subset)...")
        _zscore_with_mask(X1, mask_all)
        _zscore_with_mask(X2, mask_all)
        _zscore_with_mask(X3, mask_all)
    if feature_clip is not None:
        print(f"Applying feature clipping +/-{feature_clip}...")
        _sanitize_inplace(X1, clip_abs=feature_clip)
        _sanitize_inplace(X2, clip_abs=feature_clip)
        _sanitize_inplace(X3, clip_abs=feature_clip)

    # --- Targets (10kb) ---
    Y_snv_10kb_all = base["y_snv_10kb"].to_numpy(np.float32, copy=False)
    Y_indel_10kb_all = base["y_ind_10kb"].to_numpy(np.float32, copy=False)

    # Sanity: we expect essentially no missing 10kb SNV/INDEL (as breast already showed)
    missing_snv = int(np.isnan(Y_snv_10kb_all).sum())
    missing_ind = int(np.isnan(Y_indel_10kb_all).sum())
    print(f"Missing SNV 10kb targets: {missing_snv}")
    print(f"Missing INDEL 10kb targets: {missing_ind}")

    # --- mu_caps from training-like subset (approx training logic) ---
    Ys1_all = _pos_clip(base["y_snv_1mb"].to_numpy(np.float32, copy=False))
    Ys2_all = _pos_clip(base["y_snv_100kb"].to_numpy(np.float32, copy=False))
    Ys3_all = _pos_clip(Y_snv_10kb_all.copy())
    Yi1_all = _pos_clip(base["y_ind_1mb"].to_numpy(np.float32, copy=False))
    Yi2_all = _pos_clip(base["y_ind_100kb"].to_numpy(np.float32, copy=False))
    Yi3_all = _pos_clip(Y_indel_10kb_all.copy())

    mu_caps = {
        "snv": {
            "1mb": float(max(10.0, np.percentile(Ys1_all[mask_all], 99.9) * 2.0)) if mask_all.any() else 1e6,
            "100kb": float(max(10.0, np.percentile(Ys2_all[mask_all], 99.9) * 2.0)) if mask_all.any() else 1e6,
            "10kb": float(max(10.0, np.percentile(Ys3_all[mask_all], 99.9) * 2.0)) if mask_all.any() else 1e6,
        },
        "indel": {
            "1mb": float(max(10.0, np.percentile(Yi1_all[mask_all], 99.9) * 2.0)) if mask_all.any() else 1e6,
            "100kb": float(max(10.0, np.percentile(Yi2_all[mask_all], 99.9) * 2.0)) if mask_all.any() else 1e6,
            "10kb": float(max(10.0, np.percentile(Yi3_all[mask_all], 99.9) * 2.0)) if mask_all.any() else 1e6,
        },
    }
    print("mu_caps p99.9*2 (SNV):", {s: round(mu_caps["snv"][s], 3) for s in SCALES})
    print("mu_caps p99.9*2 (INDEL):", {s: round(mu_caps["indel"][s], 3) for s in SCALES})

    coords = base[["chr", "start"]].copy()

    return (
        X1,
        X2,
        X3,
        Y_snv_10kb_all,
        Y_indel_10kb_all,
        coords,
        C1,
        C2,
        C3,
        mu_caps,
    )


# ---------------------- Prediction ----------------------


@torch.no_grad()
def predict_snv_10kb(
    model: "HierMulti",
    X1: np.ndarray,
    X2: np.ndarray,
    X3: np.ndarray,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Predict SNV μ at 10kb for all windows."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    n = X3.shape[0]
    preds = []
    for i in range(0, n, batch_size):
        x1 = torch.from_numpy(X1[i : i + batch_size]).float().to(device)
        x2 = torch.from_numpy(X2[i : i + batch_size]).float().to(device)
        x3 = torch.from_numpy(X3[i : i + batch_size]).float().to(device)
        mu, _ = model(x1, x2, x3)
        preds.append(mu["snv"]["10kb"].detach().cpu())
    return torch.cat(preds).numpy().astype(np.float64)


# ---------------------- Permutation importance (10kb features) ----------------------


def permutation_importance_10kb(
    model: "HierMulti",
    X1: np.ndarray,
    X2: np.ndarray,
    X3: np.ndarray,
    Y_snv_10kb: np.ndarray,
    cols_10kb: List[str],
    repeats: int = 1000,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
    feature_regex: Optional[str] = None,
    seed: int = 42,
    max_windows_for_perm: Optional[int] = None,
) -> pd.DataFrame:
    """
    Permute 10kb features one-by-one and measure ΔR² for SNV 10kb.

    ΔR² = R²_baseline - R²_permuted (positive => feature helps).

    We may subsample windows for permutation to control runtime.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mask out NaN targets (should be none, but keep safe)
    mask_obs = ~np.isnan(Y_snv_10kb)
    X1_use = X1[mask_obs]
    X2_use = X2[mask_obs]
    X3_use = X3[mask_obs]
    y_use = Y_snv_10kb[mask_obs].astype(np.float64)

    # Optional downsample windows for permutation
    if max_windows_for_perm is not None and max_windows_for_perm < len(y_use):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y_use), size=max_windows_for_perm, replace=False)
        X1_use = X1_use[idx]
        X2_use = X2_use[idx]
        X3_use = X3_use[idx]
        y_use = y_use[idx]
        print(f"[Permutation] Subsampled windows for permutation: "
              f"{len(y_use)}/{mask_obs.sum()}")

    # Baseline R2 (SNV 10kb)
    preds_base = predict_snv_10kb(model, X1_use, X2_use, X3_use,
                                  batch_size=batch_size, device=device)
    baseline_r2 = float(r2_score(y_use, preds_base))
    print(f"[Permutation] Baseline R² (SNV 10kb) on used windows: {baseline_r2:.4f}")

    # Feature subset (regex)
    mask_feat = np.ones(len(cols_10kb), dtype=bool)
    if feature_regex:
        pat = re.compile(feature_regex, re.IGNORECASE)
        mask_feat = np.array([bool(pat.search(c)) for c in cols_10kb], dtype=bool)
    feat_idxs = np.where(mask_feat)[0]
    print(f"[Permutation] Features to test at 10kb: {len(feat_idxs)}/{len(cols_10kb)}")

    rng = np.random.default_rng(seed)
    records = []

    for j in feat_idxs:
        deltas = []
        for _ in range(repeats):
            X3p = X3_use.copy()
            X3p[:, j] = rng.permutation(X3p[:, j])

            preds_perm = predict_snv_10kb(model, X1_use, X2_use, X3p,
                                          batch_size=batch_size, device=device)
            r2_perm = float(r2_score(y_use, preds_perm))
            deltas.append(baseline_r2 - r2_perm)

        deltas = np.array(deltas, dtype=np.float64)
        d_mean = float(np.mean(deltas))
        d_std = float(np.std(deltas, ddof=1)) if repeats > 1 else 0.0

        R = repeats
        p_greater = (1 + np.sum(deltas <= 0)) / (R + 1)
        p_less    = (1 + np.sum(deltas >= 0)) / (R + 1)
        p_two     = min(1.0, 2.0 * min(p_greater, p_less))

        records.append({
            "scale": "10kb",
            "feature": cols_10kb[j],
            "baseline_r2_snv_10kb": baseline_r2,
            "delta_r2_mean": d_mean,
            "delta_r2_std": d_std,
            "repeats": repeats,
            "p_one_sided_greater": float(p_greater),
            "p_one_sided_less": float(p_less),
            "p_two_sided": float(p_two),
        })

    df = pd.DataFrame.from_records(records).sort_values(
        ["p_one_sided_greater", "delta_r2_mean"],
        ascending=[True, False],
    )
    return df


# ---------------------- SHAP (Captum) for SNV 10kb ----------------------


def captum_shap_snv_10kb(
    model: "HierMulti",
    X1: np.ndarray,
    X2: np.ndarray,
    X3: np.ndarray,
    Y_snv_10kb: np.ndarray,
    cols_10kb: List[str],
    background_n: int = 200,
    sample_n: int = 200,
    coalitions: int = 512,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    beeswarm_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, List[str]]:
    """
    SHAP-style attributions for SNV 10kb (10kb features only).
    """
    if not _HAS_CAPTUM:
        raise RuntimeError("captum is not installed; cannot run SHAP.")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    mask_obs = ~np.isnan(Y_snv_10kb)
    X1_use = X1[mask_obs]
    X2_use = X2[mask_obs]
    X3_use = X3[mask_obs]
    y_use = Y_snv_10kb[mask_obs]

    n = X3_use.shape[0]
    rng = np.random.default_rng(0)
    bg_idx = rng.choice(n, size=min(background_n, n), replace=False)
    sm_idx = rng.choice(n, size=min(sample_n, n), replace=False)

    X1_t = torch.from_numpy(X1_use[sm_idx]).float().to(device)
    X2_t = torch.from_numpy(X2_use[sm_idx]).float().to(device)
    X3_t = torch.from_numpy(X3_use[sm_idx]).float().to(device)

    b1 = torch.from_numpy(X1_use[bg_idx].mean(axis=0).astype(np.float32)).to(device).unsqueeze(0)
    b2 = torch.from_numpy(X2_use[bg_idx].mean(axis=0).astype(np.float32)).to(device).unsqueeze(0)
    b3 = torch.from_numpy(X3_use[bg_idx].mean(axis=0).astype(np.float32)).to(device).unsqueeze(0)

    def forward_func(x1, x2, x3):
        mu, _ = model(x1, x2, x3)
        return mu["snv"]["10kb"]  # [B]

    explainer = ShapleyValueSampling(forward_func)

    attributions = explainer.attribute(
        inputs=(X1_t, X2_t, X3_t),
        baselines=(b1, b2, b3),
        n_samples=coalitions,
    )
    # We only care about 10kb branch attributions here
    a1, a2, a3 = attributions
    A_scale = a3.detach().cpu().numpy()
    V_scale = X3_use[sm_idx].astype(np.float64, copy=False)

    # summary per feature
    mean_abs = np.mean(np.abs(A_scale), axis=0)
    mean_signed = np.mean(A_scale, axis=0)
    df_sum = pd.DataFrame({
        "scale": "10kb",
        "feature": cols_10kb,
        "mean_abs_attr": mean_abs,
        "mean_attr": mean_signed,
    }).sort_values("mean_abs_attr", ascending=False)

    # dependence
    dep_rows = []
    for j, name in enumerate(cols_10kb):
        v = V_scale[:, j]
        a = A_scale[:, j]
        if np.all(v == v[0]) or np.all(a == a[0]):
            pearson_r = np.nan
            spearman_rho = np.nan
            slope = np.nan
            intercept = np.nan
        else:
            pearson_r = float(np.corrcoef(v, a)[0, 1])
            spearman_rho = _spearman_from_numpy(v, a)
            slope, intercept = np.polyfit(v, a, 1)

        q20 = float(np.quantile(v, 0.20))
        q80 = float(np.quantile(v, 0.80))
        low = a[v <= q20]
        high = a[v >= q80]
        mean_low = float(np.mean(low)) if low.size else np.nan
        mean_high = float(np.mean(high)) if high.size else np.nan
        delta = (mean_high - mean_low) if (np.isfinite(mean_low) and np.isfinite(mean_high)) else np.nan
        pct_pos_low = float(np.mean(low > 0)) if low.size else np.nan
        pct_pos_high = float(np.mean(high > 0)) if high.size else np.nan

        dep_rows.append({
            "scale": "10kb",
            "feature": name,
            "pearson_r_value_vs_attr": pearson_r,
            "spearman_rho_value_vs_attr": spearman_rho,
            "slope_attr_vs_value": slope,
            "intercept": intercept,
            "q20_value": q20,
            "q80_value": q80,
            "mean_attr_low20": mean_low,
            "mean_attr_high80": mean_high,
            "delta_attr_high_minus_low": delta,
            "pct_pos_attr_low20": pct_pos_low,
            "pct_pos_attr_high80": pct_pos_high,
            "n_low": int((v <= q20).sum()),
            "n_high": int((v >= q80).sum()),
        })
    df_dep = pd.DataFrame(dep_rows).sort_values(
        ["delta_attr_high_minus_low", "spearman_rho_value_vs_attr"],
        ascending=[False, False],
    )

    # Optional beeswarm long file
    if beeswarm_path is not None:
        beeswarm_path = Path(beeswarm_path)
        with gzip.open(beeswarm_path, "wt") as gz:
            gz.write("row\tfeature\tvalue\tattr\n")
            n_rows, n_feats = A_scale.shape
            for i in range(n_rows):
                ai = A_scale[i]
                vi = V_scale[i]
                for j, name in enumerate(cols_10kb):
                    gz.write(f"{i}\t{name}\t{vi[j]:.8g}\t{ai[j]:.8g}\n")

    # quick checks
    with torch.no_grad():
        y_true = torch.from_numpy(y_use[sm_idx])
        y_pred = forward_func(X1_t, X2_t, X3_t).detach().cpu()
    df_checks = pd.DataFrame({
        "scale": "10kb",
        "y_true": y_true.numpy().astype(np.float64),
        "y_pred": y_pred.numpy().astype(np.float64),
    })

    return df_sum, df_checks, df_dep, A_scale, cols_10kb


# ---------------------- Main ----------------------


def main():
    ap = argparse.ArgumentParser()
    # data
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

    # model
    ap.add_argument(
        "--model_outdir",
        required=True,
        help="Directory containing best_model.pt for this cancer type "
             "(e.g., final_version/best_models/<ctype>_best).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=1024)

    # permutation settings
    ap.add_argument("--permutation_repeats", type=int, default=1000)
    ap.add_argument(
        "--feature_regex",
        type=str,
        default=None,
        help="Regex to select 10kb features (e.g. '(ca|acc|rt)'); empty=all.",
    )
    ap.add_argument("--p_threshold", type=float, default=0.001)
    ap.add_argument(
        "--p_use",
        choices=["p_one_sided_greater", "p_two_sided"],
        default="p_one_sided_greater",
    )
    ap.add_argument(
        "--max_windows_for_perm",
        type=int,
        default=None,
        help="Optional cap on number of windows used for permutation importance "
             "(subsample; helps control runtime).",
    )

    # SHAP
    ap.add_argument("--shap_background", type=int, default=200)
    ap.add_argument("--shap_samples", type=int, default=200)
    ap.add_argument("--shap_coalitions", type=int, default=512)
    ap.add_argument(
        "--save_shap_long",
        action="store_true",
        help="Also save shap_long_10kb.tsv.gz and shap_beeswarm_10kb.tsv.gz.",
    )

    # preprocessing
    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--feature_clip", type=float, default=None)

    args = ap.parse_args()

    train_mod = load_training_module(Path(args.model_outdir))

    global HierMulti, SCALES, _canon_cols, _read_ca, _read_mut, _select_ct, _sanitize_inplace

    HierMulti = train_mod.HierMulti
    SCALES = getattr(train_mod, "SCALES", ["1mb", "100kb", "10kb"])

    _canon_cols = train_mod._canon_cols
    _read_ca = train_mod._read_ca
    _read_mut = train_mod._read_mut
    _select_ct = train_mod._select_ct
    _sanitize_inplace = train_mod._sanitize_inplace


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.model_outdir) / "SHAP_10kb"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[Output dir] {outdir}")

    print(f"[Preprocessing] zscore={args.zscore} feature_clip={args.feature_clip}")

    # ---------- Prepare data ----------
    (
        X1,
        X2,
        X3,
        Y_snv_10kb,
        Y_indel_10kb,
        coords,
        C1,
        C2,
        C3,
        mu_caps,
    ) = prepare_all_windows_10kb(
        Path(args.ca_1mb),
        Path(args.ca_100kb),
        Path(args.ca_10kb),
        Path(args.snv_1mb),
        Path(args.snv_100kb),
        Path(args.snv_10kb),
        Path(args.indel_1mb),
        Path(args.indel_100kb),
        Path(args.indel_10kb),
        ctype=args.ctype,
        zscore=args.zscore,
        feature_clip=args.feature_clip,
    )

    # ---------- Build model + load best weights ----------
    model = HierMulti(
        X1.shape[1],
        X2.shape[1],
        X3.shape[1],
        branch_dim=256,
        hidden=256,
        dropout=0.3,
        loss_type="poisson",
        mu_caps=mu_caps,
        fix_theta=None,
    ).to(device)

    state_path = Path(args.model_outdir) / "best_model.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"best_model.pt not found in {args.model_outdir}")
    print(f"[Model] Loading state from: {state_path}")
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ---------- Baseline predictions (all windows) ----------
    preds = predict_snv_10kb(model, X1, X2, X3,
                             batch_size=args.batch_size, device=device)
    mask_obs = ~np.isnan(Y_snv_10kb)
    baseline_r2_all = float(r2_score(Y_snv_10kb[mask_obs].astype(np.float64),
                                     preds[mask_obs]))
    print(f"[Baseline] R² (SNV 10kb) on ALL obs windows: {baseline_r2_all:.4f}")

    baseline_path = outdir / "snv_10kb_baseline.tsv"
    df_base = coords.copy()
    df_base["y_true_snv_10kb"] = Y_snv_10kb.astype(np.float64)
    df_base["y_pred_snv_10kb"] = preds
    df_base.to_csv(baseline_path, sep="\t", index=False)
    print(f"[Baseline] Saved obs/pred table -> {baseline_path}")

    # ---------- Permutation importance (10kb features) ----------
    print("[Permutation] Starting permutation importance at 10kb features...")
    df_perm = permutation_importance_10kb(
        model,
        X1,
        X2,
        X3,
        Y_snv_10kb,
        C3,
        repeats=args.permutation_repeats,
        batch_size=args.batch_size,
        device=device,
        feature_regex=args.feature_regex,
        seed=args.seed,
        max_windows_for_perm=args.max_windows_for_perm,
    )
    perm_path = outdir / "permutation_importance_10kb.tsv"
    df_perm.to_csv(perm_path, sep="\t", index=False)
    print(f"[Permutation] Saved full table -> {perm_path}")

    # significant / non-significant splits
    pcol = args.p_use
    if pcol not in df_perm.columns:
        raise ValueError(
            f"P-value column '{pcol}' not found in permutation output columns: "
            f"{df_perm.columns.tolist()}"
        )
    sig_mask = df_perm[pcol] <= args.p_threshold
    sig_path = outdir / "permutation_importance_10kb_significant.tsv"
    nonsig_path = outdir / "permutation_importance_10kb_nonsignificant.tsv"
    df_perm[sig_mask].to_csv(sig_path, sep="\t", index=False)
    df_perm[~sig_mask].to_csv(nonsig_path, sep="\t", index=False)
    print(f"[Permutation] significant (<= {args.p_threshold}, {pcol}) -> {sig_path}")
    print(f"[Permutation] non-significant (> {args.p_threshold}, {pcol}) -> {nonsig_path}")

    # ---------- SHAP ----------
    print("[SHAP] SNV 10kb (10kb features only)...")
    if _HAS_CAPTUM:
        beeswarm_path = None
        if args.save_shap_long:
            beeswarm_path = outdir / "shap_beeswarm_10kb.tsv.gz"
        df_shap_sum, df_checks, df_dep, A_scale, cols_shap = captum_shap_snv_10kb(
            model,
            X1,
            X2,
            X3,
            Y_snv_10kb,
            C3,
            background_n=args.shap_background,
            sample_n=args.shap_samples,
            coalitions=args.shap_coalitions,
            batch_size=256,
            device=device,
            beeswarm_path=beeswarm_path,
        )
        shap_sum_path = outdir / "shap_summary_10kb.tsv"
        shap_checks_path = outdir / "shap_checks_10kb.tsv"
        shap_dep_path = outdir / "shap_dependency_10kb.tsv"
        df_shap_sum.to_csv(shap_sum_path, sep="\t", index=False)
        df_checks.to_csv(shap_checks_path, sep="\t", index=False)
        df_dep.to_csv(shap_dep_path, sep="\t", index=False)
        print(f"[SHAP] summary    -> {shap_sum_path}")
        print(f"[SHAP] checks     -> {shap_checks_path}")
        print(f"[SHAP] dependency -> {shap_dep_path}")

        if args.save_shap_long:
            long_path = outdir / "shap_long_10kb.tsv.gz"
            with gzip.open(long_path, "wt") as gz:
                gz.write("row\tfeature\tattr\n")
                for i in range(A_scale.shape[0]):
                    for j, name in enumerate(cols_shap):
                        gz.write(f"{i}\t{name}\t{A_scale[i, j]:.8g}\n")
            print(f"[SHAP] long matrix -> {long_path}")
    else:
        print("[SHAP] captum not installed; skipping SHAP. pip install captum to enable.")

    print("[Done] Importance results in:", outdir)


if __name__ == "__main__":
    main()
