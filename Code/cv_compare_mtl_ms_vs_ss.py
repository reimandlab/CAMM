#!/usr/bin/env python3
"""
cv_compare_mtl_ms_vs_ss.py

Blocked, repeated K-fold cross-validation comparing:
  - MTL-MS: multi-task, multi-scale (hierarchical DL)
  - MTL-SS: multi-task, single-scale (ablation; one model per scale)

Prevents spatial leakage by grouping windows into 1Mb blocks; folds are formed on groups.

IMPORTANT UPDATE (multi-scale evaluation):
  - Training/early-stopping for MTL-MS still uses --target_scale (default: 10kb) as the
    validation metric scale (for LR scheduler + early stopping).
  - Final evaluation is performed for ALL scales: 1mb, 100kb, 10kb in a single run.
  - Test metrics at each scale are computed using scale-appropriate grouping:
        1mb   -> block-level grouping by chr:start_1mb
        100kb -> block-level grouping by chr:start_100kb
        10kb  -> window-level grouping by chr:start (identity)

IMPORTANT UPDATE (missing targets / "value drop" fix):
  - Many mutation-window CSVs omit zero-count windows. After merge, those become NaN.
  - By default (--missing_targets zero), missing targets are filled with 0 so each scale's
    performance is based on ALL windows (per CA 10kb windows), not only rows where every
    target was present.
  - You can restore old behavior with: --missing_targets drop

IMPORTANT UPDATE (no leakage scaling):
  - Optional per-fold feature standardization (--zscore_in_fold) uses training-split stats
    within each fold (applied to train/val/test).

Outputs:
  <outdir>/cv_folds.tsv
  <outdir>/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============ CONSTANTS ============
SCALES = ["1mb", "100kb", "10kb"]
TASKS = ["snv", "indel"]


# ============ I/O & PREP ============

def _canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    return df


def _read_ca(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="infer", low_memory=False)


def _read_mut(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="infer", low_memory=False)


def _select_ct(df_mut: pd.DataFrame, ctype: str) -> pd.DataFrame:
    c = ctype.lower()
    if c not in df_mut.columns:
        raise KeyError(f"Cancer type '{ctype}' not found in {list(df_mut.columns)[:12]}...")
    out = df_mut[["chr", "start", c]].rename(columns={c: "y"}).copy()
    # normalize keys to reduce silent merge-misses
    out["chr"] = out["chr"].astype(str)
    out["start"] = pd.to_numeric(out["start"], errors="coerce").fillna(0).astype("int64")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    return out


def _sanitize_inplace(A: np.ndarray, fill: float = 0.0, clip_abs: float | None = None) -> None:
    np.nan_to_num(A, copy=False, nan=fill, posinf=fill, neginf=fill)
    if clip_abs is not None:
        np.clip(A, -clip_abs, clip_abs, out=A)


def _feat_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("chr", "start")]


def load_base_and_arrays(
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
    feature_clip: float | None = None,
    missing_targets: str = "zero",  # "zero" (recommended) or "drop" (old behavior)
):
    # CA
    ca1 = _canon_cols(_read_ca(ca_1mb))
    ca2 = _canon_cols(_read_ca(ca_100kb))
    ca3 = _canon_cols(_read_ca(ca_10kb))

    # Targets
    s1 = _canon_cols(_read_mut(snv_1mb))
    s2 = _canon_cols(_read_mut(snv_100kb))
    s3 = _canon_cols(_read_mut(snv_10kb))
    i1 = _canon_cols(_read_mut(indel_1mb))
    i2 = _canon_cols(_read_mut(indel_100kb))
    i3 = _canon_cols(_read_mut(indel_10kb))

    for df, name in [
        (ca1, "CA_1MB"),
        (ca2, "CA_100KB"),
        (ca3, "CA_10KB"),
        (s1, "SNV_1MB"),
        (s2, "SNV_100KB"),
        (s3, "SNV_10KB"),
        (i1, "INDEL_1MB"),
        (i2, "INDEL_100KB"),
        (i3, "INDEL_10KB"),
    ]:
        if not {"chr", "start"}.issubset(df.columns):
            raise KeyError(f"{name} must contain 'chr' and 'start'")

    # Base at 10kb resolution
    base = ca3[["chr", "start"]].copy()
    base["chr"] = base["chr"].astype(str)
    base["start"] = pd.to_numeric(base["start"], errors="coerce").fillna(0).astype("int64")

    base["start_100kb"] = ((base["start"] - 1) // 100_000) * 100_000 + 1
    base["start_1mb"] = ((base["start"] - 1) // 1_000_000) * 1_000_000 + 1

    # Merge features
    X1_cols = _feat_cols(ca1)
    base = base.merge(
        ca1.rename(columns={"start": "start_1mb", "chr": "chr"}),
        on=["chr", "start_1mb"],
        how="left",
    )
    X2_cols = _feat_cols(ca2)
    base = base.merge(
        ca2.rename(columns={"start": "start_100kb", "chr": "chr"}),
        on=["chr", "start_100kb"],
        how="left",
    )
    X3_cols = _feat_cols(ca3)
    base = base.merge(ca3, on=["chr", "start"], how="left")

    # Merge targets
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

    need = [
        "y_snv_1mb",
        "y_snv_100kb",
        "y_snv_10kb",
        "y_ind_1mb",
        "y_ind_100kb",
        "y_ind_10kb",
    ]
    for c in need:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    before = len(base)
    nan_counts = base[need].isna().sum()

    if missing_targets == "zero":
        total_nan = int(nan_counts.sum())
        if total_nan > 0:
            msg = ", ".join([f"{k}={int(v)}" for k, v in nan_counts.items() if int(v) > 0])
            print(f"[prep] NaNs after target merge (will fill with 0): {msg}")
            worst = float((nan_counts / max(len(base), 1)).max())
            if worst > 0.50:
                print(
                    "[prep][WARN] >50% missing in at least one target column after merge. "
                    "This may indicate chr naming ('1' vs 'chr1') or 0/1-based coordinate mismatch."
                )
        base[need] = base[need].fillna(0.0)
        base = base.reset_index(drop=True)
        print(f"[prep] kept {len(base)}/{before} rows; filled missing targets with 0.")
    elif missing_targets == "drop":
        base = base.dropna(subset=need).reset_index(drop=True)
        print(f"[prep] kept {len(base)}/{before} rows with all targets present.")
    else:
        raise ValueError(f"--missing_targets must be 'zero' or 'drop', got: {missing_targets}")

    # numpy arrays
    X1 = base[X1_cols].to_numpy(np.float32, copy=False)
    X2 = base[X2_cols].to_numpy(np.float32, copy=False)
    X3 = base[X3_cols].to_numpy(np.float32, copy=False)

    def pos_clip(a: np.ndarray) -> np.ndarray:
        # counts should be >=0; also remove inf/nan
        return np.clip(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0), 0, None)

    Ys1 = pos_clip(base["y_snv_1mb"].to_numpy(np.float32, copy=False))
    Ys2 = pos_clip(base["y_snv_100kb"].to_numpy(np.float32, copy=False))
    Ys3 = pos_clip(base["y_snv_10kb"].to_numpy(np.float32, copy=False))
    Yi1 = pos_clip(base["y_ind_1mb"].to_numpy(np.float32, copy=False))
    Yi2 = pos_clip(base["y_ind_100kb"].to_numpy(np.float32, copy=False))
    Yi3 = pos_clip(base["y_ind_10kb"].to_numpy(np.float32, copy=False))

    for X in (X1, X2, X3):
        _sanitize_inplace(X, fill=0.0, clip_abs=feature_clip)

    # Group ids for CV + evaluation
    base["group_1mb"] = base["chr"].astype(str) + ":" + base["start_1mb"].astype(str)
    base["group_100kb"] = base["chr"].astype(str) + ":" + base["start_100kb"].astype(str)
    base["group_10kb"] = base["chr"].astype(str) + ":" + base["start"].astype(str)

    arrays = (X1, X2, X3, Ys1, Ys2, Ys3, Yi1, Yi2, Yi3)
    cols = (X1_cols, X2_cols, X3_cols)
    return base, arrays, cols


# ============ DATASETS & GROUPED CV ============

class SNVIndelDataset(Dataset):
    def __init__(self, X1, X2, X3, Ys1, Ys2, Ys3, Yi1, Yi2, Yi3):
        n = len(Ys3)
        for arr in [X1, X2, X3, Ys1, Ys2, Ys3, Yi1, Yi2, Yi3]:
            assert len(arr) == n
        self.x1 = torch.from_numpy(X1).float()
        self.x2 = torch.from_numpy(X2).float()
        self.x3 = torch.from_numpy(X3).float()
        self.ys1 = torch.from_numpy(Ys1).float()
        self.ys2 = torch.from_numpy(Ys2).float()
        self.ys3 = torch.from_numpy(Ys3).float()
        self.yi1 = torch.from_numpy(Yi1).float()
        self.yi2 = torch.from_numpy(Yi2).float()
        self.yi3 = torch.from_numpy(Yi3).float()

    def __len__(self):
        return len(self.ys3)

    def __getitem__(self, i):
        return (
            self.x1[i],
            self.x2[i],
            self.x3[i],
            self.ys1[i],
            self.ys2[i],
            self.ys3[i],
            self.yi1[i],
            self.yi2[i],
            self.yi3[i],
        )


def make_blocked_folds(group_labels: np.ndarray, K: int, repeats: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns a list of (train_idx, test_idx) where folds are formed on GROUPS (blocks), not windows.
    Repeat K-fold "repeats" times with different shuffles.
    """
    rng = np.random.RandomState(seed)
    groups = np.unique(group_labels)
    folds_all: List[Tuple[np.ndarray, np.ndarray]] = []

    for r in range(repeats):
        rng.shuffle(groups)
        group_folds = np.array_split(groups, K)
        for k in range(K):
            te_groups = group_folds[k]
            te_mask = np.isin(group_labels, te_groups)
            te_idx = np.where(te_mask)[0]
            tr_idx = np.where(~te_mask)[0]
            folds_all.append((tr_idx, te_idx))
    return folds_all


# ============ MODELS ============

class GatingLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        return self.do(self.bn(F.relu(self.fc(x))))


class Branch(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.gate = GatingLayer(in_dim, out_dim, dropout=dropout)

    def forward(self, x):
        e = self.gate(x)
        return e, e


class HierMulti_MS(nn.Module):
    """MTL-MS: multi-task, multi-scale (hierarchical)."""

    def __init__(
        self,
        f1: int,
        f2: int,
        f3: int,
        branch_dim: int = 256,
        hidden: int = 256,
        dropout: float = 0.3,
        mu_caps: Dict[str, Dict[str, float]] | None = None,
    ):
        super().__init__()
        self.mu_caps = mu_caps or {t: {s: 1e6 for s in SCALES} for t in TASKS}

        self.b1 = Branch(f1, out_dim=branch_dim, dropout=dropout)
        self.b2 = Branch(f2, out_dim=branch_dim, dropout=dropout)
        self.b3 = Branch(f3, out_dim=branch_dim, dropout=dropout)

        self.shared1 = nn.Sequential(
            nn.Linear(branch_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout)
        )
        self.shared2 = nn.Sequential(
            nn.Linear(branch_dim + hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
        )

        self.shared3_snv = nn.Sequential(
            nn.Linear(branch_dim + hidden + hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
        )
        self.shared3_indel = nn.Sequential(
            nn.Linear(branch_dim + hidden + hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
        )

        def head():
            return nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())

        self.mu_1mb_snv = head()
        self.mu_1mb_indel = head()
        self.mu_100kb_snv = head()
        self.mu_100kb_indel = head()
        self.mu_10kb_snv = head()
        self.mu_10kb_indel = head()

        # learned task weights (uncertainty weighting style)
        self.logsigma_snv = nn.Parameter(torch.tensor(0.0))
        self.logsigma_indel = nn.Parameter(torch.tensor(0.0))

    def forward(self, x1, x2, x3):
        e1, _ = self.b1(x1)
        h1 = self.shared1(e1)
        e2, _ = self.b2(x2)
        h2 = self.shared2(torch.cat([e2, h1], dim=1))
        e3, _ = self.b3(x3)
        h3_snv = self.shared3_snv(torch.cat([e3, h1, h2], dim=1))
        h3_indel = self.shared3_indel(torch.cat([e3, h1, h2], dim=1))

        def clamp(mu, task, scale):
            return torch.clamp(mu.squeeze(-1), 1e-6, self.mu_caps[task][scale])

        mu = {
            "snv": {
                "1mb": clamp(self.mu_1mb_snv(h1), "snv", "1mb"),
                "100kb": clamp(self.mu_100kb_snv(h2), "snv", "100kb"),
                "10kb": clamp(self.mu_10kb_snv(h3_snv), "snv", "10kb"),
            },
            "indel": {
                "1mb": clamp(self.mu_1mb_indel(h1), "indel", "1mb"),
                "100kb": clamp(self.mu_100kb_indel(h2), "indel", "100kb"),
                "10kb": clamp(self.mu_10kb_indel(h3_indel), "indel", "10kb"),
            },
        }
        return mu


class MTL_SingleScale(nn.Module):
    """MTL-SS: multi-task, single-scale (one scale -> shared tower -> two heads)."""

    def __init__(self, f: int, hidden: int = 256, dropout: float = 0.3, cap_snv: float = 1e6, cap_indel: float = 1e6):
        super().__init__()
        self.branch = Branch(f, out_dim=hidden, dropout=dropout)
        self.shared = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout))
        self.mu_snv = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())
        self.mu_ind = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())
        self.cap_snv, self.cap_ind = cap_snv, cap_indel
        self.logsigma_snv = nn.Parameter(torch.tensor(0.0))
        self.logsigma_indel = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        e, _ = self.branch(x)
        h = self.shared(e)
        mu_s = torch.clamp(self.mu_snv(h).squeeze(-1), 1e-6, self.cap_snv)
        mu_i = torch.clamp(self.mu_ind(h).squeeze(-1), 1e-6, self.cap_ind)
        return {"snv": mu_s, "indel": mu_i}


# ============ LOSS, METRICS, UTILS ============

def poisson_nll(mu, y):
    return nn.PoissonNLLLoss(log_input=False, full=True, reduction="mean")(mu, y)


def mse_loss(mu, y):
    return F.mse_loss(mu, y)


def r2_score_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.inf)


def mae_np(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mse_np(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def aggregate_by_group(y: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"g": groups, "y": y})
    ag = df.groupby("g", sort=False)["y"].mean().values
    return ag, np.unique(groups)


def inv_softplus_stable(y: float, eps: float = 1e-8) -> float:
    y = float(max(y, eps))
    return float(np.log(np.expm1(y)))


def init_mu_biases_ms(model: HierMulti_MS, y_means: Dict[str, Dict[str, float]], mu_caps: Dict[str, Dict[str, float]]) -> None:
    with torch.no_grad():
        head = {
            ("snv", "1mb"): model.mu_1mb_snv[0],
            ("snv", "100kb"): model.mu_100kb_snv[0],
            ("snv", "10kb"): model.mu_10kb_snv[0],
            ("indel", "1mb"): model.mu_1mb_indel[0],
            ("indel", "100kb"): model.mu_100kb_indel[0],
            ("indel", "10kb"): model.mu_10kb_indel[0],
        }
        for t in TASKS:
            for s in SCALES:
                cap = mu_caps[t][s]
                target = float(min(y_means[t][s], 0.8 * cap))
                head[(t, s)].bias.data.fill_(inv_softplus_stable(target))


def init_caps(y_tr: Dict[str, np.ndarray]) -> Dict[str, float]:
    # cap = max(10, p99.9 * 2)
    cap: Dict[str, float] = {}
    for t in ["snv", "indel"]:
        arr = np.asarray(y_tr[t], dtype=float)
        cap[t] = float(max(10.0, np.percentile(arr, 99.9) * 2.0))
    return cap


def fit_std(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = np.nanmean(X, axis=0, keepdims=True)
    s = np.nanstd(X, axis=0, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    return m, s


def apply_std(X: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
    return (X - m) / s


@torch.no_grad()
def predict_mtl_ms(model: HierMulti_MS, loader: DataLoader, device, target_scale: str) -> Dict[str, np.ndarray]:
    """Predictions for ONE target scale (used for early stopping)."""
    model.eval()
    P = {"snv": [], "indel": []}
    for b in loader:
        x1, x2, x3, *_ = b
        x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)
        mu = model(x1, x2, x3)
        P["snv"].append(mu["snv"][target_scale].detach().cpu().numpy())
        P["indel"].append(mu["indel"][target_scale].detach().cpu().numpy())
    return {"snv": np.concatenate(P["snv"]), "indel": np.concatenate(P["indel"])}


@torch.no_grad()
def predict_mtl_ms_allscales(model: HierMulti_MS, loader: DataLoader, device) -> Dict[str, Dict[str, np.ndarray]]:
    """Predictions for ALL scales (used for final evaluation)."""
    model.eval()
    P = {t: {s: [] for s in SCALES} for t in TASKS}
    for b in loader:
        x1, x2, x3, *_ = b
        x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)
        mu = model(x1, x2, x3)
        for t in TASKS:
            for s in SCALES:
                P[t][s].append(mu[t][s].detach().cpu().numpy())
    return {t: {s: np.concatenate(P[t][s]) for s in SCALES} for t in TASKS}


@torch.no_grad()
def predict_mtl_ss(model: MTL_SingleScale, loader: DataLoader, device) -> Dict[str, np.ndarray]:
    model.eval()
    P = {"snv": [], "indel": []}
    for b in loader:
        # loader yields (x, y_snv, y_ind) for TensorDataset
        x = b[0] if isinstance(b, (list, tuple)) else b
        x = x.to(device)
        mu = model(x)
        P["snv"].append(mu["snv"].detach().cpu().numpy())
        P["indel"].append(mu["indel"].detach().cpu().numpy())
    return {"snv": np.concatenate(P["snv"]), "indel": np.concatenate(P["indel"])}


# ============ TRAIN / EVAL (with inner val + scheduler) ============

def _make_train_val_split(tr_idx: np.ndarray, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(tr_idx)
    n_val = max(1, int(round(val_frac * len(tr_idx))))
    val_idx = perm[:n_val]
    tr2_idx = perm[n_val:]
    return tr2_idx, val_idx


def train_one_fold_MTL_MS(
    X1,
    X2,
    X3,
    Ys,
    Yi,
    tr_idx,
    te_idx,
    target_scale: str,
    batch_size=256,
    epochs=240,
    patience=40,
    lr=1e-3,
    dropout=0.2,
    device=None,
    loss_type="poisson",
    val_frac: float = 0.1,
    seed: int = 42,
    zscore_in_fold: bool = True,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr2_idx, va_idx = _make_train_val_split(np.asarray(tr_idx), val_frac, seed)

    # per-fold scaling to avoid leakage
    X1_tr, X1_va, X1_te = X1[tr2_idx], X1[va_idx], X1[te_idx]
    X2_tr, X2_va, X2_te = X2[tr2_idx], X2[va_idx], X2[te_idx]
    X3_tr, X3_va, X3_te = X3[tr2_idx], X3[va_idx], X3[te_idx]
    if zscore_in_fold:
        m1, s1 = fit_std(X1_tr)
        m2, s2 = fit_std(X2_tr)
        m3, s3 = fit_std(X3_tr)
        X1_tr, X1_va, X1_te = apply_std(X1_tr, m1, s1), apply_std(X1_va, m1, s1), apply_std(X1_te, m1, s1)
        X2_tr, X2_va, X2_te = apply_std(X2_tr, m2, s2), apply_std(X2_va, m2, s2), apply_std(X2_te, m2, s2)
        X3_tr, X3_va, X3_te = apply_std(X3_tr, m3, s3), apply_std(X3_va, m3, s3), apply_std(X3_te, m3, s3)

    ds_tr = SNVIndelDataset(
        X1_tr,
        X2_tr,
        X3_tr,
        Ys["1mb"][tr2_idx],
        Ys["100kb"][tr2_idx],
        Ys["10kb"][tr2_idx],
        Yi["1mb"][tr2_idx],
        Yi["100kb"][tr2_idx],
        Yi["10kb"][tr2_idx],
    )
    ds_va = SNVIndelDataset(
        X1_va,
        X2_va,
        X3_va,
        Ys["1mb"][va_idx],
        Ys["100kb"][va_idx],
        Ys["10kb"][va_idx],
        Yi["1mb"][va_idx],
        Yi["100kb"][va_idx],
        Yi["10kb"][va_idx],
    )
    ds_te = SNVIndelDataset(
        X1_te,
        X2_te,
        X3_te,
        Ys["1mb"][te_idx],
        Ys["100kb"][te_idx],
        Ys["10kb"][te_idx],
        Yi["1mb"][te_idx],
        Yi["100kb"][te_idx],
        Yi["10kb"][te_idx],
    )

    tr_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    va_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False, pin_memory=True)
    te_loader = DataLoader(ds_te, batch_size=batch_size, shuffle=False, pin_memory=True)

    # caps + bias init from train portion
    y_tr = {
        "snv": {"1mb": Ys["1mb"][tr2_idx], "100kb": Ys["100kb"][tr2_idx], "10kb": Ys["10kb"][tr2_idx]},
        "indel": {"1mb": Yi["1mb"][tr2_idx], "100kb": Yi["100kb"][tr2_idx], "10kb": Yi["10kb"][tr2_idx]},
    }
    mu_caps = {t: {s: float(max(10.0, np.percentile(y_tr[t][s], 99.9) * 2.0)) for s in SCALES} for t in TASKS}
    y_means = {t: {s: float(np.mean(y_tr[t][s])) for s in SCALES} for t in TASKS}

    model = HierMulti_MS(X1.shape[1], X2.shape[1], X3.shape[1], dropout=dropout, mu_caps=mu_caps).to(device)
    init_mu_biases_ms(model, y_means, mu_caps)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=5, min_lr=1e-6, verbose=False
    )

    def _loss_one(mu, y):
        return poisson_nll(mu, y) if loss_type == "poisson" else mse_loss(mu, y)

    best = -1e9
    no_imp = 0
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        for b in tr_loader:
            x1, x2, x3, ys1, ys2, ys3, yi1, yi2, yi3 = [t.to(device) if torch.is_tensor(t) else t for t in b]
            mu = model(x1, x2, x3)
            loss_snv = _loss_one(mu["snv"]["1mb"], ys1) + _loss_one(mu["snv"]["100kb"], ys2) + _loss_one(mu["snv"]["10kb"], ys3)
            loss_ind = _loss_one(mu["indel"]["1mb"], yi1) + _loss_one(mu["indel"]["100kb"], yi2) + _loss_one(mu["indel"]["10kb"], yi3)
            total = (
                torch.exp(-2 * model.logsigma_snv) * loss_snv
                + 2 * model.logsigma_snv
                + torch.exp(-2 * model.logsigma_indel) * loss_ind
                + 2 * model.logsigma_indel
            )
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        # composite early stop metric: average of R2 across tasks on target_scale
        with torch.no_grad():
            P_va = predict_mtl_ms(model, va_loader, device, target_scale)
        r2_s = r2_score_np(Ys[target_scale][va_idx], P_va["snv"])
        r2_i = r2_score_np(Yi[target_scale][va_idx], P_va["indel"])
        r2_va = 0.5 * (r2_s + r2_i)
        scheduler.step(r2_va)

        if r2_va > best:
            best = r2_va
            no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, te_loader


def train_one_fold_MTL_SS(
    X_target,
    Y_snv,
    Y_ind,
    tr_idx,
    te_idx,
    batch_size=256,
    epochs=240,
    patience=40,
    lr=1e-3,
    dropout=0.2,
    device=None,
    loss_type="poisson",
    val_frac: float = 0.1,
    seed: int = 42,
    zscore_in_fold: bool = True,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr2_idx, va_idx = _make_train_val_split(np.asarray(tr_idx), val_frac, seed)

    # per-fold scaling to avoid leakage
    X_tr, X_va, X_te = X_target[tr2_idx], X_target[va_idx], X_target[te_idx]
    if zscore_in_fold:
        m, s = fit_std(X_tr)
        X_tr, X_va, X_te = apply_std(X_tr, m, s), apply_std(X_va, m, s), apply_std(X_te, m, s)

    # caps from train
    caps = init_caps({"snv": Y_snv[tr2_idx], "indel": Y_ind[tr2_idx]})
    model = MTL_SingleScale(
        f=X_target.shape[1], hidden=256, dropout=dropout, cap_snv=caps["snv"], cap_indel=caps["indel"]
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=5, min_lr=1e-6, verbose=False
    )

    def _loss_one(mu, y):
        return poisson_nll(mu, y) if loss_type == "poisson" else mse_loss(mu, y)

    # loaders (use already-split X arrays)
    tr_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(Y_snv[tr2_idx]).float(),
        torch.from_numpy(Y_ind[tr2_idx]).float(),
    )
    va_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_va).float(),
        torch.from_numpy(Y_snv[va_idx]).float(),
        torch.from_numpy(Y_ind[va_idx]).float(),
    )
    te_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_te).float(),
        torch.from_numpy(Y_snv[te_idx]).float(),
        torch.from_numpy(Y_ind[te_idx]).float(),
    )

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    best = -1e9
    no_imp = 0
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        for x, ys, yi in tr_loader:
            x, ys, yi = x.to(device), ys.to(device), yi.to(device)
            mu = model(x)
            loss_snv = _loss_one(mu["snv"], ys)
            loss_ind = _loss_one(mu["indel"], yi)
            total = (
                torch.exp(-2 * model.logsigma_snv) * loss_snv
                + 2 * model.logsigma_snv
                + torch.exp(-2 * model.logsigma_indel) * loss_ind
                + 2 * model.logsigma_indel
            )
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        with torch.no_grad():
            P_va = predict_mtl_ss(model, va_loader, device)
        r2_s = r2_score_np(Y_snv[va_idx], P_va["snv"])
        r2_i = r2_score_np(Y_ind[va_idx], P_va["indel"])
        r2_va = 0.5 * (r2_s + r2_i)
        scheduler.step(r2_va)

        if r2_va > best:
            best = r2_va
            no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, te_loader


# ============ STATS: CIs & P-values ============

def mean_ci_t(values: List[float], alpha=0.05):
    try:
        import scipy.stats as st
        x = np.asarray(values, dtype=float)
        n = len(x)
        m = float(np.mean(x))
        s = float(np.std(x, ddof=1))
        if n < 2 or s == 0.0:
            return m, (m, m)
        h = st.t.ppf(1 - alpha / 2, df=n - 1) * s / math.sqrt(n)
        return m, (m - h, m + h)
    except Exception:
        x = np.asarray(values, dtype=float)
        n = len(x)
        m = float(np.mean(x))
        s = float(np.std(x, ddof=1))
        h = 1.96 * (s / math.sqrt(max(n, 1)))
        return m, (m - h, m + h)


def p_wilcoxon_signed(values: List[float]):
    try:
        import scipy.stats as st
        stat, p = st.wilcoxon(values, zero_method="wilcox", alternative="two-sided", correction=False, mode="auto")
        return float(p)
    except Exception:
        return float("nan")


def p_permutation_signflip(values: List[float], B=5000, seed=0):
    rng = np.random.RandomState(seed)
    x = np.asarray(values, dtype=float)
    t_obs = float(np.mean(x))
    if len(x) == 0:
        return float("nan")
    cnt = 0
    for _ in range(B):
        flips = rng.choice([-1, 1], size=len(x))
        t_b = float(np.mean(x * flips))
        if abs(t_b) >= abs(t_obs):
            cnt += 1
    return (cnt + 1) / (B + 1)


def cohen_d_paired(values: List[float]):
    x = np.asarray(values, dtype=float)
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    return m / (s + 1e-12)


# ============ MAIN ============

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

    # NOTE: --target_scale is now used for EARLY STOPPING / scheduler metric (for MTL-MS).
    # Final evaluation is computed for ALL scales (1mb/100kb/10kb).
    ap.add_argument(
        "--target_scale",
        choices=SCALES,
        default="10kb",
        help="Scale used for validation metric (early stopping + LR scheduler) for MTL-MS. "
             "Final evaluation is computed for ALL scales (1mb/100kb/10kb).",
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    # training
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--loss", choices=["poisson", "mse"], default="poisson")
    ap.add_argument("--val_frac", type=float, default=0.10, help="Inner validation fraction from train for early stopping.")

    # preprocessing
    ap.add_argument("--zscore_in_fold", action="store_true", help="Standardize features using train split stats within each fold.")
    ap.add_argument("--feature_clip", type=float, default=None)
    ap.add_argument(
        "--missing_targets",
        choices=["zero", "drop"],
        default="zero",
        help="How to handle missing mutation targets after merge. 'zero' fills NaN with 0 (recommended if CSVs omit zero windows). "
             "'drop' keeps only rows with all targets present (old behavior).",
    )

    # stats
    ap.add_argument("--perm_B", type=int, default=5000)

    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    # seeds/determinism
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    outdir = Path(args.outdir)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}  ctype={args.ctype}  early_stop_scale={args.target_scale}")

    # ---- load data
    base, arrays, _ = load_base_and_arrays(
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
        feature_clip=args.feature_clip,
        missing_targets=args.missing_targets,
    )
    X1, X2, X3, Ys1, Ys2, Ys3, Yi1, Yi2, Yi3 = arrays
    Ys = {"1mb": Ys1, "100kb": Ys2, "10kb": Ys3}
    Yi = {"1mb": Yi1, "100kb": Yi2, "10kb": Yi3}

    # CV blocking always at 1Mb
    groups_cv = base["group_1mb"].to_numpy()

    # evaluation grouping per scale
    groups_eval_by_scale = {
        "1mb": base["group_1mb"].to_numpy(),
        "100kb": base["group_100kb"].to_numpy(),
        "10kb": base["group_10kb"].to_numpy(),
    }

    folds = make_blocked_folds(groups_cv, K=args.folds, repeats=args.repeats, seed=args.seed)
    print(f"[cv] total folds = {len(folds)} (K={args.folds} x R={args.repeats})")

    rows: List[Dict[str, object]] = []
    deltas: Dict[str, Dict[str, List[float]]] = {t: {s: [] for s in SCALES} for t in TASKS}

    for fold_id, (tr_idx, te_idx) in enumerate(folds, start=1):
        print(f"\n[fold {fold_id}] train={len(tr_idx)} test={len(te_idx)}")

        # --- train/eval MTL-MS once per fold
        ms_model, te_loader_ms = train_one_fold_MTL_MS(
            X1,
            X2,
            X3,
            Ys,
            Yi,
            tr_idx,
            te_idx,
            target_scale=args.target_scale,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            dropout=args.dropout,
            device=device,
            loss_type=args.loss,
            val_frac=args.val_frac,
            seed=args.seed + fold_id,
            zscore_in_fold=args.zscore_in_fold,
        )
        P_ms_all = predict_mtl_ms_allscales(ms_model, te_loader_ms, device)

        # --- train/eval MTL-SS per scale (one model per scale)
        P_ss_all: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in SCALES}

        for scale in SCALES:
            X_target = {"1mb": X1, "100kb": X2, "10kb": X3}[scale]

            ss_model, te_loader_ss = train_one_fold_MTL_SS(
                X_target,
                Ys[scale],
                Yi[scale],
                tr_idx,
                te_idx,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                dropout=args.dropout,
                device=device,
                loss_type=args.loss,
                val_frac=args.val_frac,
                seed=args.seed + 10_000 + fold_id + (0 if scale == "1mb" else 1 if scale == "100kb" else 2),
                zscore_in_fold=args.zscore_in_fold,
            )
            P_ss_all[scale] = predict_mtl_ss(ss_model, te_loader_ss, device)

        # --- evaluation for ALL scales, with scale-appropriate grouping
        for scale in SCALES:
            g_te = groups_eval_by_scale[scale][te_idx]

            # SNV
            y_true_s = Ys[scale][te_idx]
            y_s_grp, grp_ids_s = aggregate_by_group(y_true_s, g_te)
            p_ms_s_grp, _ = aggregate_by_group(P_ms_all["snv"][scale], g_te)
            p_ss_s_grp, _ = aggregate_by_group(P_ss_all[scale]["snv"], g_te)

            r2_ms_s = r2_score_np(y_s_grp, p_ms_s_grp)
            r2_ss_s = r2_score_np(y_s_grp, p_ss_s_grp)
            mae_ms_s = mae_np(y_s_grp, p_ms_s_grp)
            mae_ss_s = mae_np(y_s_grp, p_ss_s_grp)
            mse_ms_s = mse_np(y_s_grp, p_ms_s_grp)
            mse_ss_s = mse_np(y_s_grp, p_ss_s_grp)

            rows.append({"fold": fold_id, "task": "snv", "model": "MTL-MS", "scale": scale, "r2": r2_ms_s, "mae": mae_ms_s, "mse": mse_ms_s, "n_blocks": len(grp_ids_s)})
            rows.append({"fold": fold_id, "task": "snv", "model": "MTL-SS", "scale": scale, "r2": r2_ss_s, "mae": mae_ss_s, "mse": mse_ss_s, "n_blocks": len(grp_ids_s)})

            d_s = r2_ms_s - r2_ss_s
            deltas["snv"][scale].append(d_s)

            # INDEL
            y_true_i = Yi[scale][te_idx]
            y_i_grp, grp_ids_i = aggregate_by_group(y_true_i, g_te)
            p_ms_i_grp, _ = aggregate_by_group(P_ms_all["indel"][scale], g_te)
            p_ss_i_grp, _ = aggregate_by_group(P_ss_all[scale]["indel"], g_te)

            r2_ms_i = r2_score_np(y_i_grp, p_ms_i_grp)
            r2_ss_i = r2_score_np(y_i_grp, p_ss_i_grp)
            mae_ms_i = mae_np(y_i_grp, p_ms_i_grp)
            mae_ss_i = mae_np(y_i_grp, p_ss_i_grp)
            mse_ms_i = mse_np(y_i_grp, p_ms_i_grp)
            mse_ss_i = mse_np(y_i_grp, p_ss_i_grp)

            rows.append({"fold": fold_id, "task": "indel", "model": "MTL-MS", "scale": scale, "r2": r2_ms_i, "mae": mae_ms_i, "mse": mse_ms_i, "n_blocks": len(grp_ids_i)})
            rows.append({"fold": fold_id, "task": "indel", "model": "MTL-SS", "scale": scale, "r2": r2_ss_i, "mae": mae_ss_i, "mse": mse_ss_i, "n_blocks": len(grp_ids_i)})

            d_i = r2_ms_i - r2_ss_i
            deltas["indel"][scale].append(d_i)

            print(
                f"[fold {fold_id}] {scale:5s}  "
                f"SNV R2: MTL-MS={r2_ms_s:.4f} MTL-SS={r2_ss_s:.4f} dR2={d_s:+.4f}   |   "
                f"INDEL R2: MTL-MS={r2_ms_i:.4f} MTL-SS={r2_ss_i:.4f} dR2={d_i:+.4f}"
            )

    # ---- write fold-level results
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "cv_folds.tsv", sep="\t", index=False)

    # ---- aggregate stats (per task AND per scale)
    summary = {
        "ctype": args.ctype,
        "folds": args.folds,
        "repeats": args.repeats,
        "N": len(folds),
        "loss": args.loss,
        "early_stop_scale": args.target_scale,
        "val_frac": args.val_frac,
        "zscore_in_fold": bool(args.zscore_in_fold),
        "feature_clip": args.feature_clip,
        "missing_targets": args.missing_targets,
        "results": {t: {s: {} for s in SCALES} for t in TASKS},
    }

    for task in TASKS:
        for scale in SCALES:
            d = deltas[task][scale]
            mean_d, (lo_d, hi_d) = mean_ci_t(d, alpha=0.05)
            p_w = p_wilcoxon_signed(d)
            p_perm = p_permutation_signflip(d, B=args.perm_B, seed=args.seed)
            effect = cohen_d_paired(d)
            win_rate = float(np.mean(np.array(d) > 0.0)) if len(d) else float("nan")

            r2_ms_all = df[(df.task == task) & (df.model == "MTL-MS") & (df.scale == scale)]["r2"].tolist()
            r2_ss_all = df[(df.task == task) & (df.model == "MTL-SS") & (df.scale == scale)]["r2"].tolist()
            ms_mean, (ms_lo, ms_hi) = mean_ci_t(r2_ms_all)
            ss_mean, (ss_lo, ss_hi) = mean_ci_t(r2_ss_all)

            summary["results"][task][scale] = {
                "r2": {
                    "MTL-MS": {"mean": ms_mean, "ci95": [ms_lo, ms_hi]},
                    "MTL-SS": {"mean": ss_mean, "ci95": [ss_lo, ss_hi]},
                },
                "delta_r2": {
                    "mean": mean_d,
                    "ci95": [lo_d, hi_d],
                    "p_wilcoxon": p_w,
                    "p_permutation_signflip": p_perm,
                    "effect_size_cohen_d": effect,
                    "win_rate": win_rate,
                },
            }

    with open(outdir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n[done] wrote:")
    print("  -", outdir / "cv_folds.tsv")
    print("  -", outdir / "summary.json")


if __name__ == "__main__":
    main()
