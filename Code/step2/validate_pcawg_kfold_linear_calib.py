#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate frozen HMF models on PCAWG counts with RANDOM 5-fold CV and
compare calibration methods:

Methods reported per (task,scale):
  - global_ols            : slope-only   (W * yhat)
  - global_poisson        : slope-only via Poisson deviance (same closed-form W)
  - perchr_ols            : slope-only, per chromosome
  - perchr_poisson        : slope-only, per chromosome (same closed-form W)
  - global_ols_intercept  : ycal = a + b * yhat  (OLS)
  - perchr_ols_intercept  : ycal = a_c + b_c * yhat (OLS per chromosome)

Outputs in --outdir/<ctype>/:
  - calibration_comparison.<ctype>.json         (all folds, per (task,scale))
  - calibration_comparison.summary.<ctype>.json (aggregate_mean / aggregate_sd)
  - pred_counts_<ctype>_fold<k>_<task>_<scale>.tsv.gz (if --save_preds)
"""

import argparse, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr


SCALES = ["1mb", "100kb", "10kb"]
TASKS  = ["snv", "indel"]


# ---------------- I/O & utilities ----------------

def _canon_cols(df):
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    if "chr" in df.columns:
        df["chr"] = df["chr"].astype(str).str.replace("^chr", "", regex=True)
    if "start" in df.columns:
        df["start"] = pd.to_numeric(df["start"], errors="coerce").astype("Int64")
    return df

def _read_table(path):
    return pd.read_csv(path, sep="\t", compression="infer", low_memory=False)

def _read_csv(path):
    return pd.read_csv(path, compression="infer", low_memory=False)

def _select_ct(df, ctype):
    c = ctype.lower()
    if c not in df.columns:
        raise KeyError("Cancer type '%s' not found in: %s ..." % (ctype, list(df.columns)[:12]))
    return df[["chr","start", c]].rename(columns={c: "y"})

def _feat_cols(df):
    return [c for c in df.columns if c not in {"chr","start","start_100kb","start_1mb"}]

def _add_bin_keys(df):
    df = df.copy()
    s = df["start"].astype("int64")
    df["start_100kb"] = ((s - 1) // 100000) * 100000 + 1
    df["start_1mb"]   = ((s - 1) // 1000000) * 1000000 + 1
    return df

def _sanitize_inplace(A, fill=0.0, clip_abs=None):
    np.nan_to_num(A, copy=False, nan=fill, posinf=fill, neginf=fill)
    if clip_abs is not None:
        np.clip(A, -clip_abs, clip_abs, out=A)

def _zscore_inplace(A):
    m = np.nanmean(A, axis=0, keepdims=True)
    s = np.nanstd(A, axis=0, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    A -= m; A /= s

def _agg_mean(df, by_cols):
    feats = _feat_cols(df)
    return df.groupby(by_cols, as_index=False)[feats].mean()

def _broadcast(parent_df, child_df_with_keys, parent_key):
    feats = _feat_cols(parent_df)
    src = parent_df.rename(columns={"start": parent_key})[["chr", parent_key] + feats]
    return child_df_with_keys.merge(src, on=["chr", parent_key], how="left")


# ---------------- Model (inference only) ----------------

class GatingLayer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lin = nn.Linear(input_size, input_size)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)
    def forward(self, x):
        g = torch.sigmoid(self.lin(x))
        return x * g, g

class Branch(nn.Module):
    def __init__(self, input_size, out_dim=256, dropout=0.3):
        super().__init__()
        self.gate = GatingLayer(input_size)
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        gx, g = self.gate(x)
        e = self.mlp(gx)
        return e, g

class HierMulti(nn.Module):
    def __init__(self, f1, f2, f3, branch_dim=256, hidden=256, dropout=0.3):
        super().__init__()
        self.b1 = Branch(f1, out_dim=branch_dim, dropout=dropout)
        self.b2 = Branch(f2, out_dim=branch_dim, dropout=dropout)
        self.b3 = Branch(f3, out_dim=branch_dim, dropout=dropout)
        self.shared1 = nn.Sequential(nn.Linear(branch_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout))
        self.shared2 = nn.Sequential(nn.Linear(branch_dim + hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout))
        self.shared3_snv   = nn.Sequential(nn.Linear(branch_dim + hidden + hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout))
        self.shared3_indel = nn.Sequential(nn.Linear(branch_dim + hidden + hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout))
        def head(): return nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())
        self.mu_1mb_snv   = head(); self.mu_1mb_indel   = head()
        self.mu_100kb_snv = head(); self.mu_100kb_indel = head()
        self.mu_10kb_snv  = head(); self.mu_10kb_indel  = head()
    def forward(self, x1,x2,x3):
        e1,_ = self.b1(x1); h1 = self.shared1(e1)
        e2,_ = self.b2(x2); h2 = self.shared2(torch.cat([e2,h1], dim=1))
        e3,_ = self.b3(x3)
        h3_snv   = self.shared3_snv(torch.cat([e3,h1,h2], dim=1))
        h3_indel = self.shared3_indel(torch.cat([e3,h1,h2], dim=1))
        def clamp(mu): return torch.clamp(mu.squeeze(-1), 1e-6, 1e12)
        mu = {
            "snv":   {"1mb": clamp(self.mu_1mb_snv(h1)),
                      "100kb":clamp(self.mu_100kb_snv(h2)),
                      "10kb": clamp(self.mu_10kb_snv(h3_snv))},
            "indel": {"1mb": clamp(self.mu_1mb_indel(h1)),
                      "100kb":clamp(self.mu_100kb_indel(h2)),
                      "10kb": clamp(self.mu_10kb_indel(h3_indel))},
        }
        return mu


# ---------------- Preparation per scale ----------------

def build_inputs_for_scale(scale, ca1, ca2, ca3, pc_snv, pc_ind, ctype,
                           restrict_autosomes, zscore, feature_clip):
    ca1 = _canon_cols(ca1); ca2 = _canon_cols(ca2); ca3 = _canon_cols(ca3)
    pc_snv = _select_ct(_canon_cols(pc_snv), ctype)
    pc_ind = _select_ct(_canon_cols(pc_ind), ctype)

    # base rows (same scale)
    if scale == "1mb": base = ca1[["chr","start"]].copy()
    elif scale == "100kb": base = ca2[["chr","start"]].copy()
    elif scale == "10kb": base = ca3[["chr","start"]].copy()
    else: raise ValueError(scale)
    base = _add_bin_keys(base)

    if restrict_autosomes:
        base = base[base["chr"].isin([str(i) for i in range(1,23)])].reset_index(drop=True)

    # y from PCAWG counts, keep zeros
    y_snv = base.merge(pc_snv, on=["chr","start"], how="left")["y"].fillna(0.0).to_numpy(np.float32)
    y_ind = base.merge(pc_ind, on=["chr","start"], how="left")["y"].fillna(0.0).to_numpy(np.float32)

    ca1k = _add_bin_keys(ca1); ca2k = _add_bin_keys(ca2); ca3k = _add_bin_keys(ca3)
    basek = _add_bin_keys(base)

    if scale == "1mb":
        X1_df = base.merge(ca1, on=["chr","start"], how="left")
        X2_df = base.merge(_agg_mean(ca2k, ["chr","start_1mb"]).rename(columns={"start_1mb":"start"}), on=["chr","start"], how="left")
        X3_df = base.merge(_agg_mean(ca3k, ["chr","start_1mb"]).rename(columns={"start_1mb":"start"}), on=["chr","start"], how="left")
    elif scale == "100kb":
        X2_df = base.merge(ca2, on=["chr","start"], how="left")
        X1_df = _broadcast(ca1, basek, parent_key="start_1mb")
        X3_df = base.merge(_agg_mean(ca3k, ["chr","start_100kb"]).rename(columns={"start_100kb":"start"}), on=["chr","start"], how="left")
    else:
        X3_df = base.merge(ca3, on=["chr","start"], how="left")
        X2_df = _broadcast(ca2, basek, parent_key="start_100kb")
        X1_df = _broadcast(ca1, basek, parent_key="start_1mb")

    def toX(df):
        X = df[_feat_cols(df)].to_numpy(np.float32, copy=False)
        _sanitize_inplace(X, fill=0.0, clip_abs=feature_clip)
        return X

    X1 = toX(X1_df); X2 = toX(X2_df); X3 = toX(X3_df)
    if zscore:
        _zscore_inplace(X1); _zscore_inplace(X2); _zscore_inplace(X3)

    coords = base[["chr","start"]].copy()
    return (X1,X2,X3, y_snv, y_ind, coords)


# ---------------- Metrics & calibration helpers ----------------

def poisson_deviance(y, lam, eps=1e-12):
    y = np.asarray(y, dtype=float)
    lam = np.asarray(lam, dtype=float)
    lam = np.clip(lam, eps, None)
    term = np.where(y > 0, y * np.log(y / lam), 0.0)
    return float(2.0 * np.sum(term - (y - lam)))

def deviance_explained(y, yhat):
    mu_null = np.full_like(np.asarray(y, float), float(np.mean(y)))
    Dm = poisson_deviance(y, yhat)
    D0 = poisson_deviance(y, mu_null)
    if not np.isfinite(Dm) or not np.isfinite(D0) or D0 <= 0:
        return float('nan')
    return float(1.0 - Dm / D0)

def r2_within_chr(y, yhat, chrs):
    y = np.asarray(y); yhat = np.asarray(yhat); chrs = np.asarray(chrs)
    vals = []
    for c in np.unique(chrs):
        idx = (chrs == c)
        if idx.sum() < 3:  # needs variance
            continue
        try:
            vals.append(r2_score(y[idx], yhat[idx]))
        except Exception:
            vals.append(np.nan)
    if not vals: return float('nan')
    return float(np.nanmean(vals))

def spearman_safe(y, yhat):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(spearmanr(y, yhat, nan_policy='omit').statistic)
    except Exception:
        return float('nan')

def compute_W(y_tr, p_tr):
    den = float(np.sum(p_tr))
    return float(np.sum(y_tr) / den) if den > 0 else 1.0

def fit_ols_intercept(y_tr, p_tr):
    # Solve [a, b] by least squares for y ≈ a + b p
    X = np.column_stack([np.ones_like(p_tr, dtype=float), p_tr.astype(float)])
    y = y_tr.astype(float)
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a = float(beta[0]); b = float(beta[1])
    except Exception:
        a = float(np.mean(y)); b = 0.0
    return a, b

def apply_clip_pos(arr, eps=1e-6):
    arr = np.asarray(arr, dtype=float)
    np.clip(arr, eps, None, out=arr)
    return arr

def metrics_block(y_true, y_pred, chrs):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    r2 = r2_score(y_true, y_pred) if np.var(y_true) > 0 else float('nan')
    r2w = r2_within_chr(y_true, y_pred, chrs)
    rho = spearman_safe(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    dev = poisson_deviance(y_true, y_pred)
    dev_null = poisson_deviance(y_true, np.full_like(y_true, float(np.mean(y_true))))
    dev_exp = 1.0 - (dev / dev_null) if np.isfinite(dev) and np.isfinite(dev_null) and dev_null > 0 else float('nan')
    return {
        "R2": float(r2),
        "R2_within_chr": float(r2w),
        "SpearmanR": float(rho),
        "MAE": float(mae),
        "MSE": float(mse),
        "PoissonDeviance": float(dev),
        "PoissonDevianceNull": float(dev_null),
        "DevianceExplained": float(dev_exp),
        "N": int(len(y_true)),
    }


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    # CA inputs
    ap.add_argument('--ca_1mb', required=True)
    ap.add_argument('--ca_100kb', required=True)
    ap.add_argument('--ca_10kb', required=True)
    # PCAWG counts
    ap.add_argument('--pcawg_snv_1mb', required=True)
    ap.add_argument('--pcawg_snv_100kb', required=True)
    ap.add_argument('--pcawg_snv_10kb', required=True)
    ap.add_argument('--pcawg_ind_1mb', required=True)
    ap.add_argument('--pcawg_ind_100kb', required=True)
    ap.add_argument('--pcawg_ind_10kb', required=True)
    # Core
    ap.add_argument('--ctype', required=True)
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--k_folds', type=int, default=5)
    ap.add_argument('--restrict_autosomes', action='store_true')
    ap.add_argument('--zscore', action='store_true')
    ap.add_argument('--feature_clip', type=float, default=None)
    ap.add_argument('--save_preds', action='store_true')
    ap.add_argument('--random_seed', type=int, default=42)
    args = ap.parse_args()

    outroot = Path(args.outdir); outroot.mkdir(parents=True, exist_ok=True)

    # Load CA & PCAWG
    ca1 = _read_table(Path(args.ca_1mb))
    ca2 = _read_table(Path(args.ca_100kb))
    ca3 = _read_table(Path(args.ca_10kb))

    s1  = _read_csv(Path(args.pcawg_snv_1mb))
    s2  = _read_csv(Path(args.pcawg_snv_100kb))
    s3  = _read_csv(Path(args.pcawg_snv_10kb))
    i1  = _read_csv(Path(args.pcawg_ind_1mb))
    i2  = _read_csv(Path(args.pcawg_ind_100kb))
    i3  = _read_csv(Path(args.pcawg_ind_10kb))

    # containers
    results = []        # per-fold entries (like the snippet you shared)
    agg_vals = {}       # key -> list of metric dicts to average

    for scale, (pc_snv, pc_ind) in {
        "1mb":   (s1, i1),
        "100kb": (s2, i2),
        "10kb":  (s3, i3),
    }.items():
        # build inputs for this scale
        X1,X2,X3, Ys, Yi, coords = build_inputs_for_scale(
            scale, ca1, ca2, ca3, pc_snv, pc_ind, args.ctype,
            args.restrict_autosomes, args.zscore, args.feature_clip
        )
        chrs = coords["chr"].astype(str).to_numpy()
        starts = coords["start"].astype(int).to_numpy()

        # model & weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        f1,f2,f3 = X1.shape[1], X2.shape[1], X3.shape[1]
        model = HierMulti(f1,f2,f3, branch_dim=256, hidden=256, dropout=0.3).to(device).eval()
        pt = Path(args.model_dir) / 'best_model.pt'
        if not pt.exists():
            raise FileNotFoundError("Missing weights: %s" % str(pt))
        state = torch.load(pt, map_location=device)
        model_state = model.state_dict()
        filtered = {k:v for k,v in state.items() if (k in model_state and tuple(v.shape)==tuple(model_state[k].shape))}
        model.load_state_dict(filtered, strict=False)

        # predict all rows (raw counts)
        with torch.no_grad():
            x1 = torch.from_numpy(X1).float().to(device)
            x2 = torch.from_numpy(X2).float().to(device)
            x3 = torch.from_numpy(X3).float().to(device)
            mu = model(x1,x2,x3)
        P_raw = {
            "snv":   mu["snv"][scale].cpu().numpy(),
            "indel": mu["indel"][scale].cpu().numpy(),
        }
        Y_true = {"snv": Ys, "indel": Yi}

        # random KFold
        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_seed)
        idx_all = np.arange(len(chrs))

        for fold_i, (tr_idx, te_idx) in enumerate(kf.split(idx_all), start=1):
            fold_tag = "fold_%d" % fold_i

            for task in TASKS:
                y_tr = Y_true[task][tr_idx]; y_te = Y_true[task][te_idx]
                p_tr = P_raw[task][tr_idx];  p_te = P_raw[task][te_idx]
                chr_tr = chrs[tr_idx];       chr_te = chrs[te_idx]

                # ----- methods -----
                methods = {}

                # 1) global_ols (slope-only)
                W = compute_W(y_tr, p_tr)
                yhat = apply_clip_pos(W * p_te)
                methods["global_ols"] = {"W": float(W), "metrics": metrics_block(y_te, yhat, chr_te)}

                # 2) global_poisson (slope-only, same closed-form W for identity link)
                Wp = compute_W(y_tr, p_tr)
                yhat_p = apply_clip_pos(Wp * p_te)
                methods["global_poisson"] = {"W": float(Wp), "metrics": metrics_block(y_te, yhat_p, chr_te)}

                # 3) perchr_ols (slope-only per chromosome)
                Wc = {}
                yhat_pc = np.empty_like(y_te, dtype=float)
                for c in np.unique(chr_te):
                    idx_tr_c = (chr_tr == c)
                    idx_te_c = (chr_te == c)
                    if np.any(idx_tr_c) and np.sum(p_tr[idx_tr_c]) > 0:
                        Wc_c = compute_W(y_tr[idx_tr_c], p_tr[idx_tr_c])
                    else:
                        Wc_c = compute_W(y_tr, p_tr)
                    Wc[str(c)] = float(Wc_c)
                    yhat_pc[idx_te_c] = Wc_c * p_te[idx_te_c]
                yhat_pc = apply_clip_pos(yhat_pc)
                methods["perchr_ols"] = {"W": Wc, "metrics": metrics_block(y_te, yhat_pc, chr_te)}

                # 4) perchr_poisson (slope-only per chromosome; same Wc)
                methods["perchr_poisson"] = {"W": Wc, "metrics": metrics_block(y_te, yhat_pc, chr_te)}

                # 5) global_ols_intercept
                a, b = fit_ols_intercept(y_tr, p_tr)
                yhat_gib = apply_clip_pos(a + b * p_te)
                methods["global_ols_intercept"] = {"a": float(a), "b": float(b), "metrics": metrics_block(y_te, yhat_gib, chr_te)}

                # 6) perchr_ols_intercept
                ABc = {}
                yhat_cib = np.empty_like(y_te, dtype=float)
                # fallback to global a,b if a chromosome has very few training points
                for c in np.unique(chr_te):
                    idx_tr_c = (chr_tr == c)
                    idx_te_c = (chr_te == c)
                    if np.sum(idx_tr_c) >= 3:
                        a_c, b_c = fit_ols_intercept(y_tr[idx_tr_c], p_tr[idx_tr_c])
                    else:
                        a_c, b_c = a, b
                    ABc[str(c)] = {"a": float(a_c), "b": float(b_c)}
                    yhat_cib[idx_te_c] = a_c + b_c * p_te[idx_te_c]
                yhat_cib = apply_clip_pos(yhat_cib)
                methods["perchr_ols_intercept"] = {"ab_per_chr": ABc, "metrics": metrics_block(y_te, yhat_cib, chr_te)}

                # save preds (optional): raw + the two intercept methods (to keep file size reasonable)
                if args.save_preds:
                    dfp = pd.DataFrame({
                        "chr": chr_te,
                        "start": starts[te_idx],
                        "fold": fold_i, "ctype": args.ctype, "scale": scale, "task": task,
                        "y_true": y_te,
                        "y_pred_raw": p_te,
                        "y_pred_global_ols_intercept": yhat_gib,
                        "y_pred_perchr_ols_intercept": yhat_cib
                    })
                    out_gz = outroot / ("pred_counts_%s_fold%d_%s_%s.tsv.gz" % (args.ctype, fold_i, task, scale))
                    dfp.to_csv(out_gz, sep="\t", index=False, compression="gzip")

                # store fold entry
                entry = {
                    "ctype": args.ctype,
                    "fold": fold_i,
                    "task": task,
                    "scale": scale,
                    "methods": methods
                }
                results.append(entry)

                # also push metrics to aggregate buckets
                for mname, mobj in methods.items():
                    key = "%s|%s|%s|%s" % (args.ctype, task, scale, mname)
                    agg_vals.setdefault(key, []).append(mobj["metrics"])

    # ----- save per-fold file -----
    out_all = outroot / ("calibration_comparison.%s.json" % args.ctype)
    with open(out_all, "w") as fh:
        json.dump({"results": results}, fh, indent=2)

    # ----- aggregate mean/sd across folds -----
    def agg_stats(arr_of_dicts):
        # arr_of_dicts: list of metric dicts
        keys = list(arr_of_dicts[0].keys())
        mean_d = {}; sd_d = {}
        for k in keys:
            vals = [float(x.get(k, np.nan)) for x in arr_of_dicts]
            a = np.array(vals, float)
            mean_d[k] = float(np.nanmean(a))
            sd_d[k]   = float(np.nanstd(a, ddof=1))
        return mean_d, sd_d

    aggregate_mean = {}
    aggregate_sd   = {}
    for key, arr in agg_vals.items():
        if len(arr) == 0: continue
        m, s = agg_stats(arr)
        aggregate_mean[key] = m
        aggregate_sd[key]   = s

    out_summary = outroot / ("calibration_comparison.summary.%s.json" % args.ctype)
    with open(out_summary, "w") as fh:
        json.dump({"aggregate_mean": aggregate_mean, "aggregate_sd": aggregate_sd}, fh, indent=2)

    print("[done] Wrote:")
    print("  - %s" % str(out_all))
    print("  - %s" % str(out_summary))


if __name__ == "__main__":
    main()
