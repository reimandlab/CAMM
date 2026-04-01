#!/usr/bin/env python3
"""run_model_hier_single_task_cv.py

Hierarchical *multi-scale only* model (single task) with repeated K-fold CV.

This is the same coarse-to-fine multi-scale architecture as
`run_model_hier_multi.py`, but **without** multi-task sharing between SNV and
INDEL.

You typically run it twice:
  * --task snv   (predict SNV counts at 1mb/100kb/10kb)
  * --task indel (predict INDEL counts at 1mb/100kb/10kb)

CV settings (defaults):
  * 5-fold
  * 10 repeats

Outputs:
  <outdir>/repeatXX/foldYY/  (fold artifacts)
  <outdir>/cv_results.tsv
  <outdir>/cv_summary.json

Notes
-----
* This is window-level CV (random split over 10kb windows).
* Targets can be kept as NaN (partial supervision) or filled as 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


SCALES = ["1mb", "100kb", "10kb"]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- I/O ----------------


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
        raise KeyError(
            f"Cancer type '{ctype}' not in mutation table. "
            f"Available (sample): {df_mut.columns.tolist()[:12]}"
        )
    return df_mut[["chr", "start", c]].rename(columns={c: "y"})


def _zscore_inplace(A: np.ndarray) -> None:
    m = np.nanmean(A, axis=0, keepdims=True)
    s = np.nanstd(A, axis=0, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    A -= m
    A /= s


def _sanitize_inplace(
    A: np.ndarray, fill: float = 0.0, clip_abs: float | None = None
) -> None:
    np.nan_to_num(A, copy=False, nan=fill, posinf=fill, neginf=fill)
    if clip_abs is not None:
        np.clip(A, -clip_abs, clip_abs, out=A)


def _feat_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("chr", "start")]


def prepare_arrays_single_task(
    *,
    ca_1mb: Path,
    ca_100kb: Path,
    ca_10kb: Path,
    mut_1mb: Path,
    mut_100kb: Path,
    mut_10kb: Path,
    ctype: str,
    zscore: bool = False,
    feature_clip: float | None = None,
    missing_targets: str = "zero",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    List[str],
    List[str],
]:
    """Prepare aligned CA features at 3 scales + a SINGLE task target at 3 scales.

    Uses the same approach as the multi-task script: align everything to 10kb CA.
    """

    ca1 = _canon_cols(_read_ca(ca_1mb))
    ca2 = _canon_cols(_read_ca(ca_100kb))
    ca3 = _canon_cols(_read_ca(ca_10kb))

    m1 = _canon_cols(_read_mut(mut_1mb))
    m2 = _canon_cols(_read_mut(mut_100kb))
    m3 = _canon_cols(_read_mut(mut_10kb))

    for df, name in [
        (ca1, "CA_1MB"),
        (ca2, "CA_100KB"),
        (ca3, "CA_10KB"),
        (m1, "MUT_1MB"),
        (m2, "MUT_100KB"),
        (m3, "MUT_10KB"),
    ]:
        if not {"chr", "start"}.issubset(df.columns):
            raise KeyError(f"{name} must contain 'chr' and 'start'")

    base = ca3[["chr", "start"]].copy()
    base["start"] = (
        pd.to_numeric(base["start"], errors="coerce").fillna(0).astype("int64")
    )
    base["start_100kb"] = ((base["start"] - 1) // 100_000) * 100_000 + 1
    base["start_1mb"] = ((base["start"] - 1) // 1_000_000) * 1_000_000 + 1

    X1_cols = _feat_cols(ca1)
    base = base.merge(
        ca1.rename(columns={"start": "start_1mb"}),
        on=["chr", "start_1mb"],
        how="left",
    )
    X2_cols = _feat_cols(ca2)
    base = base.merge(
        ca2.rename(columns={"start": "start_100kb"}),
        on=["chr", "start_100kb"],
        how="left",
    )
    X3_cols = _feat_cols(ca3)
    base = base.merge(ca3, on=["chr", "start"], how="left")

    y1 = _select_ct(m1, ctype).rename(columns={"y": "y_1mb"})
    y2 = _select_ct(m2, ctype).rename(columns={"y": "y_100kb"})
    y3 = _select_ct(m3, ctype).rename(columns={"y": "y_10kb"})

    base = base.merge(
        y1.rename(columns={"start": "start_1mb"}),
        on=["chr", "start_1mb"],
        how="left",
    )
    base = base.merge(
        y2.rename(columns={"start": "start_100kb"}),
        on=["chr", "start_100kb"],
        how="left",
    )
    base = base.merge(y3, on=["chr", "start"], how="left")

    if missing_targets == "zero":
        for col in ["y_1mb", "y_100kb", "y_10kb"]:
            if base[col].isna().any():
                base[col] = base[col].fillna(0.0)

    base = base.reset_index(drop=True)

    X1 = base[X1_cols].to_numpy(np.float32, copy=False)
    X2 = base[X2_cols].to_numpy(np.float32, copy=False)
    X3 = base[X3_cols].to_numpy(np.float32, copy=False)

    def clip_targets(a: np.ndarray) -> np.ndarray:
        arr = a.astype(np.float32, copy=False)
        mask = np.isfinite(arr) & (arr < 0)
        arr[mask] = 0.0
        return arr

    Y1 = clip_targets(base["y_1mb"].to_numpy(np.float32, copy=False))
    Y2 = clip_targets(base["y_100kb"].to_numpy(np.float32, copy=False))
    Y3 = clip_targets(base["y_10kb"].to_numpy(np.float32, copy=False))

    for X in (X1, X2, X3):
        _sanitize_inplace(X, fill=0.0)
    if zscore:
        _zscore_inplace(X1)
        _zscore_inplace(X2)
        _zscore_inplace(X3)
    if feature_clip is not None:
        _sanitize_inplace(X1, clip_abs=feature_clip)
        _sanitize_inplace(X2, clip_abs=feature_clip)
        _sanitize_inplace(X3, clip_abs=feature_clip)

    return X1, X2, X3, Y1, Y2, Y3, X1_cols, X2_cols, X3_cols


class SingleTaskDataset(Dataset):
    def __init__(self, X1, X2, X3, Y1, Y2, Y3):
        n = len(Y3)
        for arr in [X1, X2, X3, Y1, Y2, Y3]:
            assert len(arr) == n
        self.x1 = torch.from_numpy(X1).float()
        self.x2 = torch.from_numpy(X2).float()
        self.x3 = torch.from_numpy(X3).float()
        self.y1 = torch.from_numpy(Y1).float()
        self.y2 = torch.from_numpy(Y2).float()
        self.y3 = torch.from_numpy(Y3).float()

    def __len__(self):
        return len(self.y3)

    def __getitem__(self, i: int):
        return self.x1[i], self.x2[i], self.x3[i], self.y1[i], self.y2[i], self.y3[i]


# ------------- Model -------------


class GatingLayer(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.lin = nn.Linear(input_size, input_size)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        g = torch.sigmoid(self.lin(x))
        return x * g, g


class Branch(nn.Module):
    def __init__(self, input_size: int, out_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.gate = GatingLayer(input_size)
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        gx, g = self.gate(x)
        e = self.mlp(gx)
        return e, g


class HierSingleTask(nn.Module):
    """Hierarchical multi-scale model for a single task."""

    def __init__(
        self,
        f1: int,
        f2: int,
        f3: int,
        branch_dim: int = 256,
        hidden: int = 256,
        dropout: float = 0.3,
        loss_type: str = "poisson",
        mu_caps: Dict[str, float] | None = None,
        fix_theta: float | None = None,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.mu_caps = mu_caps or {s: 1e6 for s in SCALES}
        self.fix_theta = fix_theta

        self.b1 = Branch(f1, out_dim=branch_dim, dropout=dropout)
        self.b2 = Branch(f2, out_dim=branch_dim, dropout=dropout)
        self.b3 = Branch(f3, out_dim=branch_dim, dropout=dropout)

        self.shared1 = nn.Sequential(
            nn.Linear(branch_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
        )
        self.shared2 = nn.Sequential(
            nn.Linear(branch_dim + hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
        )
        self.shared3 = nn.Sequential(
            nn.Linear(branch_dim + hidden + hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
        )

        def head():
            return nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())

        self.mu_1mb = head()
        self.mu_100kb = head()
        self.mu_10kb = head()

        if fix_theta is None:
            self.theta_1mb = nn.Parameter(torch.tensor([10.0]))
            self.theta_100kb = nn.Parameter(torch.tensor([10.0]))
            self.theta_10kb = nn.Parameter(torch.tensor([10.0]))
        else:
            self.register_buffer("theta_fixed", torch.tensor([fix_theta], dtype=torch.float32))

        self.last_gate_means = (None, None, None)

    def forward(self, x1, x2, x3):
        e1, g1 = self.b1(x1)
        h1 = self.shared1(e1)
        e2, g2 = self.b2(x2)
        h2 = self.shared2(torch.cat([e2, h1], dim=1))
        e3, g3 = self.b3(x3)
        h3 = self.shared3(torch.cat([e3, h1, h2], dim=1))

        self.last_gate_means = (
            g1.mean(0).detach(),
            g2.mean(0).detach(),
            g3.mean(0).detach(),
        )

        def clamp(mu, scale):
            return torch.clamp(mu.squeeze(-1), 1e-6, self.mu_caps[scale])

        mu = {
            "1mb": clamp(self.mu_1mb(h1), "1mb"),
            "100kb": clamp(self.mu_100kb(h2), "100kb"),
            "10kb": clamp(self.mu_10kb(h3), "10kb"),
        }

        if self.fix_theta is None:
            th = {
                "1mb": F.softplus(self.theta_1mb),
                "100kb": F.softplus(self.theta_100kb),
                "10kb": F.softplus(self.theta_10kb),
            }
        else:
            th = {s: self.theta_fixed for s in SCALES}
        return mu, th


# ------------- Loss/metrics -------------


def poisson_nll(mu, y):
    return nn.PoissonNLLLoss(log_input=False, full=True, reduction="mean")(mu, y)


def nb_nll_stable(y, mu, theta, eps: float = 1e-8):
    y = torch.clamp(y, 0, 1e12)
    theta = torch.clamp(theta, 1e-6, 1e12)
    log_theta = torch.log(theta + eps)
    log_mu = torch.log(mu + eps)
    log_theta_mu = log_theta + torch.log1p(mu / (theta + eps))
    ll = (
        torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1.0)
        + theta * (log_theta - log_theta_mu)
        + y * (log_mu - log_theta_mu)
    )
    return -ll.mean()


def _loss_one(loss_type: str, mu, y, theta):
    if loss_type == "mse":
        return F.mse_loss(mu, y)
    if loss_type == "poisson":
        return poisson_nll(mu, y)
    return nb_nll_stable(y, mu, theta)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    P = {s: [] for s in SCALES}
    Y = {s: [] for s in SCALES}
    TH = {s: [] for s in SCALES}

    for batch in loader:
        x1, x2, x3, y1, y2, y3 = batch
        x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)
        mu, th = model(x1, x2, x3)
        y = {"1mb": y1, "100kb": y2, "10kb": y3}
        for s in SCALES:
            P[s].append(torch.nan_to_num(mu[s], nan=0.0, posinf=1e12, neginf=0.0).cpu())
            Y[s].append(y[s])
            TH[s].append(float(th[s].item()))

    def stats(p_list: List[torch.Tensor], y_list: List[torch.Tensor]):
        p = torch.cat(p_list).numpy().astype(np.float64)
        y = torch.cat(y_list).numpy().astype(np.float64)
        mask = np.isfinite(y)
        if not np.any(mask):
            return (np.nan, np.nan, np.nan, float(np.mean(p)), float(np.std(p)))
        p = p[mask]
        y = y[mask]
        return (
            mean_absolute_error(y, p),
            mean_squared_error(y, p),
            r2_score(y, p),
            float(np.mean(p)),
            float(np.std(p)),
        )

    metrics = {s: {} for s in SCALES}
    for s in SCALES:
        mae, mse, r2, pm, ps = stats(P[s], Y[s])
        metrics[s] = {"mae": mae, "mse": mse, "r2": r2, "pm": pm, "ps": ps}

    theta_avg = {s: float(np.mean(TH[s])) if TH[s] else np.nan for s in SCALES}
    return metrics, theta_avg


def inv_softplus_stable(y: float, eps: float = 1e-8) -> float:
    y = max(y, eps)
    if y < 20.0:
        return float(np.log(np.expm1(y)))
    return float(y + np.log1p(-np.exp(-y)))


def init_mu_biases(model: nn.Module, y_means: Dict[str, float], mu_caps: Dict[str, float]):
    with torch.no_grad():
        head_map = {
            "1mb": model.mu_1mb[0],
            "100kb": model.mu_100kb[0],
            "10kb": model.mu_10kb[0],
        }
        for s in SCALES:
            cap = mu_caps[s]
            mean_y = y_means[s]
            target = 1.0 if not np.isfinite(mean_y) else float(min(mean_y, 0.8 * cap))
            pre = inv_softplus_stable(target)
            head_map[s].bias.data.fill_(pre)


def train_one_fold(
    *,
    X1: np.ndarray,
    X2: np.ndarray,
    X3: np.ndarray,
    Y1: np.ndarray,
    Y2: np.ndarray,
    Y3: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    outdir: Path,
    run_seed: int,
) -> Dict:
    outdir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(run_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")

    # Optional per-fold low-variance feature dropping
    X1_use, X2_use, X3_use = X1, X2, X3
    if args.drop_zero_var_thresh is not None:
        s1 = X1[train_idx].std(axis=0)
        s2 = X2[train_idx].std(axis=0)
        s3 = X3[train_idx].std(axis=0)
        keep1 = s1 >= args.drop_zero_var_thresh
        keep2 = s2 >= args.drop_zero_var_thresh
        keep3 = s3 >= args.drop_zero_var_thresh
        X1_use = X1[:, keep1]
        X2_use = X2[:, keep2]
        X3_use = X3[:, keep3]
        np.save(outdir / "keep_mask_1mb.npy", keep1)
        np.save(outdir / "keep_mask_100kb.npy", keep2)
        np.save(outdir / "keep_mask_10kb.npy", keep3)

    ds = SingleTaskDataset(X1_use, X2_use, X3_use, Y1, Y2, Y3)
    train_ds = Subset(ds, train_idx.tolist())
    val_ds = Subset(ds, val_idx.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_tr = {"1mb": Y1[train_idx], "100kb": Y2[train_idx], "10kb": Y3[train_idx]}
    mu_caps = {s: float(max(10.0, np.nanpercentile(y_tr[s], 99.9) * 2.0)) for s in SCALES}
    y_means = {s: float(np.nanmean(y_tr[s])) for s in SCALES}

    model = HierSingleTask(
        X1_use.shape[1],
        X2_use.shape[1],
        X3_use.shape[1],
        branch_dim=256,
        hidden=256,
        dropout=args.dropout,
        loss_type=args.loss,
        mu_caps=mu_caps,
        fix_theta=args.fix_theta,
    ).to(device)
    init_mu_biases(model, y_means, mu_caps)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    w_scales = {"1mb": args.w1mb, "100kb": args.w100kb, "10kb": args.w10kb}

    def primary_score(metrics: Dict) -> float:
        if args.primary_scale == "all_mean":
            return float(np.nanmean([metrics[s]["r2"] for s in SCALES]))
        return float(metrics[args.primary_scale]["r2"])

    best = -1e30
    best_info = None
    no_imp = 0
    log_rows: List[List[float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        nobs = 0
        for b in train_loader:
            x1, x2, x3, y1b, y2b, y3b = b
            x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)
            y1b, y2b, y3b = y1b.to(device), y2b.to(device), y3b.to(device)

            mu, th = model(x1, x2, x3)
            y = {"1mb": y1b, "100kb": y2b, "10kb": y3b}

            loss_scales = {}
            for s in SCALES:
                m = torch.isfinite(y[s])
                if m.any():
                    loss_scales[s] = _loss_one(args.loss, mu[s][m], y[s][m], th[s])
                else:
                    loss_scales[s] = torch.tensor(0.0, device=device)

            total = sum(w_scales[s] * loss_scales[s] for s in SCALES)

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            tot += total.item() * len(x1)
            nobs += len(x1)

        tr_loss = tot / max(nobs, 1)

        metrics, theta_avg = evaluate(model, val_loader, device)
        score = primary_score(metrics)

        flat = []
        for name in ["mae", "mse", "r2", "pm", "ps"]:
            for s in SCALES:
                flat.append(metrics[s][name])
        th_flat = [theta_avg[s] for s in SCALES]
        log_rows.append([epoch, tr_loss, score] + flat + th_flat)

        if score > best:
            best = float(score)
            no_imp = 0
            torch.save(model.state_dict(), outdir / "best_model.pt")
            best_info = {
                "epoch": epoch,
                "train_loss": float(tr_loss),
                "primary_scale": args.primary_scale,
                "primary_score": float(score),
                "metrics": metrics,
                "theta_avg": theta_avg,
                "mu_caps": mu_caps,
                "y_means": y_means,
                "run_seed": int(run_seed),
            }
        else:
            no_imp += 1
            if no_imp >= args.patience:
                break

    headers = (
        ["epoch", "trainLoss", "primaryScore"]
        + [f"val{name.upper()}_{s}" for name in ["mae", "mse", "r2", "pm", "ps"] for s in SCALES]
        + [f"thetaAvg_{s}" for s in SCALES]
    )
    with open(outdir / "train_log.tsv", "w", newline="") as fh:
        csv.writer(fh, delimiter="\t").writerows([headers, *log_rows])

    if best_info is None:
        best_info = {
            "epoch": None,
            "train_loss": None,
            "primary_scale": args.primary_scale,
            "primary_score": None,
            "metrics": None,
            "theta_avg": None,
            "mu_caps": mu_caps,
            "y_means": y_means,
            "run_seed": int(run_seed),
        }
    with open(outdir / "best_epoch.json", "w") as fh:
        json.dump(best_info, fh, indent=2)

    if hasattr(model, "last_gate_means") and model.last_gate_means[0] is not None:
        g1, g2, g3 = model.last_gate_means
        np.savetxt(outdir / "gates_mean_1MB.tsv", g1.cpu().numpy(), delimiter="\t")
        np.savetxt(outdir / "gates_mean_100KB.tsv", g2.cpu().numpy(), delimiter="\t")
        np.savetxt(outdir / "gates_mean_10KB.tsv", g3.cpu().numpy(), delimiter="\t")

    del model
    del opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_info


def flatten_best_metrics(best_info: Dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    metrics = best_info.get("metrics")
    if metrics is None:
        return out
    for s in SCALES:
        for k, v in metrics[s].items():
            out[f"{s}_{k}"] = float(v) if v is not None else np.nan
    return out


def summarize_cv(rows: List[Dict]) -> Dict:
    if not rows:
        return {"n": 0, "mean": {}, "std": {}}
    keys = set()
    for r in rows:
        keys.update(k for k, v in r.items() if isinstance(v, (int, float)) and k not in {"repeat", "fold", "seed"})
    mean = {}
    std = {}
    for k in sorted(keys):
        vals = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        mean[k] = float(np.nanmean(vals))
        std[k] = float(np.nanstd(vals))
    return {"n": int(len(rows)), "mean": mean, "std": std}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["snv", "indel"], required=True)

    # CA
    ap.add_argument("--ca_1mb", required=True)
    ap.add_argument("--ca_100kb", required=True)
    ap.add_argument("--ca_10kb", required=True)

    # Mutation CSVs for the selected task
    ap.add_argument("--mut_1mb", required=True)
    ap.add_argument("--mut_100kb", required=True)
    ap.add_argument("--mut_10kb", required=True)

    ap.add_argument("--ctype", required=True)
    ap.add_argument("--loss", choices=["mse", "poisson", "nb"], default="poisson")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--feature_clip", type=float, default=None)
    ap.add_argument("--drop_zero_var_thresh", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--fix_theta", type=float, default=None)

    ap.add_argument(
        "--missing_targets",
        choices=["nan", "zero"],
        default="zero",
        help=(
            "How to treat missing windows after merging mutation targets. "
            "'nan' keeps partial supervision; 'zero' fills missing with 0."
        ),
    )

    ap.add_argument(
        "--primary_scale",
        choices=["10kb", "100kb", "1mb", "all_mean"],
        default="10kb",
    )

    ap.add_argument("--w1mb", type=float, default=1.0)
    ap.add_argument("--w100kb", type=float, default=1.0)
    ap.add_argument("--w10kb", type=float, default=1.0)

    # CV controls
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--repeat_id", type=int, default=None)
    ap.add_argument("--fold_id", type=int, default=None)
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    set_all_seeds(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "args.json", "w") as fh:
        json.dump(vars(args), fh, indent=2)

    print(
        f"========== CV HierSingleTask | task={args.task} ctype={args.ctype} | "
        f"folds={args.folds} repeats={args.repeats} =========="
    )

    X1, X2, X3, Y1, Y2, Y3, C1, C2, C3 = prepare_arrays_single_task(
        ca_1mb=Path(args.ca_1mb),
        ca_100kb=Path(args.ca_100kb),
        ca_10kb=Path(args.ca_10kb),
        mut_1mb=Path(args.mut_1mb),
        mut_100kb=Path(args.mut_100kb),
        mut_10kb=Path(args.mut_10kb),
        ctype=args.ctype,
        zscore=args.zscore,
        feature_clip=args.feature_clip,
        missing_targets=args.missing_targets,
    )
    n = len(Y3)
    print(f"Prepared arrays: X1MB={X1.shape} X100KB={X2.shape} X10KB={X3.shape}  N={n}")

    rows: List[Dict] = []
    indices = np.arange(n)
    for rep in range(args.repeats):
        if args.repeat_id is not None and rep != args.repeat_id:
            continue
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed + rep)
        for fold, (tr, va) in enumerate(kf.split(indices)):
            if args.fold_id is not None and fold != args.fold_id:
                continue

            fold_out = outdir / f"repeat{rep:02d}" / f"fold{fold:02d}"
            best_path = fold_out / "best_epoch.json"
            if args.resume and best_path.exists():
                with open(best_path) as fh:
                    best_info = json.load(fh)
            else:
                run_seed = int(args.seed + rep * 1000 + fold)
                best_info = train_one_fold(
                    X1=X1,
                    X2=X2,
                    X3=X3,
                    Y1=Y1,
                    Y2=Y2,
                    Y3=Y3,
                    train_idx=tr,
                    val_idx=va,
                    args=args,
                    outdir=fold_out,
                    run_seed=run_seed,
                )

            row = {
                "repeat": int(rep),
                "fold": int(fold),
                "seed": int(best_info.get("run_seed", args.seed + rep * 1000 + fold)),
                "best_epoch": best_info.get("epoch"),
                "primary_score": best_info.get("primary_score"),
            }
            row.update(flatten_best_metrics(best_info))
            rows.append(row)

            print(
                f"[rep {rep:02d} fold {fold:02d}] best_epoch={row['best_epoch']} "
                f"primary_score={row['primary_score']}"
            )

    if rows:
        keys = ["repeat", "fold", "seed", "best_epoch", "primary_score"]
        metric_keys = sorted({k for r in rows for k in r.keys() if k not in keys})
        keys = keys + metric_keys
        with open(outdir / "cv_results.tsv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    summary = summarize_cv(rows)
    summary["ctype"] = args.ctype
    summary["task"] = args.task
    summary["folds"] = args.folds
    summary["repeats"] = args.repeats
    with open(outdir / "cv_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\nSaved CV outputs in:", str(outdir))


if __name__ == "__main__":
    main()
