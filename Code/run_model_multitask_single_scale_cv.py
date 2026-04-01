#!/usr/bin/env python3
"""run_model_multitask_single_scale_cv.py

Multi-task (SNV+INDEL) model **without** multi-scale hierarchy.

In this design we train **one model per scale** (1mb, 100kb, or 10kb):
  * shared feature tower for the chosen scale
  * task-specific heads for SNV and INDEL

This script runs repeated K-fold cross-validation (defaults: 5 folds x 10 repeats)
for a **single scale** specified by --scale.

Typical usage (3 separate runs):
  python run_model_multitask_single_scale_cv.py --scale 1mb   ...
  python run_model_multitask_single_scale_cv.py --scale 100kb ...
  python run_model_multitask_single_scale_cv.py --scale 10kb  ...

Outputs:
  <outdir>/repeatXX/foldYY/
  <outdir>/cv_results.tsv
  <outdir>/cv_summary.json
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


TASKS = ["snv", "indel"]
SCALE_ALIASES = {
    "1mb": "1mb",
    "100kb": "100kb",
    "10kb": "10kb",
}


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


def prepare_single_scale_arrays(
    *,
    ca_path: Path,
    snv_path: Path,
    indel_path: Path,
    ctype: str,
    zscore: bool = False,
    feature_clip: float | None = None,
    missing_targets: str = "zero",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Prepare X, y_snv, y_indel on the CA grid for a single scale."""

    ca = _canon_cols(_read_ca(ca_path))
    snv = _canon_cols(_read_mut(snv_path))
    indel = _canon_cols(_read_mut(indel_path))

    for df, name in [(ca, "CA"), (snv, "SNV"), (indel, "INDEL")]:
        if not {"chr", "start"}.issubset(df.columns):
            raise KeyError(f"{name} must contain 'chr' and 'start'")

    base = ca[["chr", "start"]].copy()
    base["start"] = (
        pd.to_numeric(base["start"], errors="coerce").fillna(0).astype("int64")
    )

    X_cols = _feat_cols(ca)
    base = base.merge(ca, on=["chr", "start"], how="left")

    y_snv = _select_ct(snv, ctype).rename(columns={"y": "y_snv"})
    y_ind = _select_ct(indel, ctype).rename(columns={"y": "y_indel"})
    base = base.merge(y_snv, on=["chr", "start"], how="left")
    base = base.merge(y_ind, on=["chr", "start"], how="left")

    if missing_targets == "zero":
        for col in ["y_snv", "y_indel"]:
            if base[col].isna().any():
                base[col] = base[col].fillna(0.0)

    X = base[X_cols].to_numpy(np.float32, copy=False)
    Ys = base["y_snv"].to_numpy(np.float32, copy=False)
    Yi = base["y_indel"].to_numpy(np.float32, copy=False)

    # clip negatives (keep NaNs)
    for Y in (Ys, Yi):
        mask = np.isfinite(Y) & (Y < 0)
        Y[mask] = 0.0

    _sanitize_inplace(X, fill=0.0)
    if zscore:
        _zscore_inplace(X)
    if feature_clip is not None:
        _sanitize_inplace(X, clip_abs=feature_clip)

    return X, Ys, Yi, X_cols


class MultiTaskScaleDataset(Dataset):
    def __init__(self, X: np.ndarray, Ys: np.ndarray, Yi: np.ndarray):
        assert len(X) == len(Ys) == len(Yi)
        self.x = torch.from_numpy(X).float()
        self.ys = torch.from_numpy(Ys).float()
        self.yi = torch.from_numpy(Yi).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.ys[i], self.yi[i]


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


class SingleScaleTower(nn.Module):
    def __init__(self, input_size: int, hidden: int = 256, dropout: float = 0.3):
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
            nn.Linear(256, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        gx, g = self.gate(x)
        h = self.mlp(gx)
        return h, g


class MultiTaskSingleScale(nn.Module):
    """Shared single-scale tower; task-specific heads for SNV and INDEL."""

    def __init__(
        self,
        f: int,
        hidden: int = 256,
        dropout: float = 0.3,
        loss_type: str = "poisson",
        mu_caps: Dict[str, float] | None = None,
        fix_theta: float | None = None,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.mu_caps = mu_caps or {"snv": 1e6, "indel": 1e6}
        self.fix_theta = fix_theta

        self.tower = SingleScaleTower(f, hidden=hidden, dropout=dropout)

        def head():
            return nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())

        self.mu_snv = head()
        self.mu_indel = head()

        if fix_theta is None:
            self.theta_snv = nn.Parameter(torch.tensor([10.0]))
            self.theta_indel = nn.Parameter(torch.tensor([10.0]))
        else:
            self.register_buffer("theta_fixed", torch.tensor([fix_theta], dtype=torch.float32))

        self.logsigma_snv = nn.Parameter(torch.tensor(0.0))
        self.logsigma_indel = nn.Parameter(torch.tensor(0.0))

        self.last_gate_mean = None

    def forward(self, x):
        h, g = self.tower(x)
        self.last_gate_mean = g.mean(0).detach()

        def clamp(mu, task):
            return torch.clamp(mu.squeeze(-1), 1e-6, self.mu_caps[task])

        mu = {
            "snv": clamp(self.mu_snv(h), "snv"),
            "indel": clamp(self.mu_indel(h), "indel"),
        }

        if self.fix_theta is None:
            th = {
                "snv": F.softplus(self.theta_snv),
                "indel": F.softplus(self.theta_indel),
            }
        else:
            th = {"snv": self.theta_fixed, "indel": self.theta_fixed}
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
    P = {t: [] for t in TASKS}
    Y = {t: [] for t in TASKS}
    TH = {t: [] for t in TASKS}

    for x, ys, yi in loader:
        x = x.to(device)
        mu, th = model(x)
        P["snv"].append(torch.nan_to_num(mu["snv"], nan=0.0, posinf=1e12, neginf=0.0).cpu())
        P["indel"].append(torch.nan_to_num(mu["indel"], nan=0.0, posinf=1e12, neginf=0.0).cpu())
        Y["snv"].append(ys)
        Y["indel"].append(yi)
        TH["snv"].append(float(th["snv"].item()))
        TH["indel"].append(float(th["indel"].item()))

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

    metrics = {}
    for t in TASKS:
        mae, mse, r2, pm, ps = stats(P[t], Y[t])
        metrics[t] = {"mae": mae, "mse": mse, "r2": r2, "pm": pm, "ps": ps}

    theta_avg = {t: float(np.mean(TH[t])) if TH[t] else np.nan for t in TASKS}
    return metrics, theta_avg


def inv_softplus_stable(y: float, eps: float = 1e-8) -> float:
    y = max(y, eps)
    if y < 20.0:
        return float(np.log(np.expm1(y)))
    return float(y + np.log1p(-np.exp(-y)))


def init_mu_biases(model: MultiTaskSingleScale, y_means: Dict[str, float], mu_caps: Dict[str, float]):
    with torch.no_grad():
        head_map = {"snv": model.mu_snv[0], "indel": model.mu_indel[0]}
        for task in TASKS:
            cap = mu_caps[task]
            mean_y = y_means[task]
            target = 1.0 if not np.isfinite(mean_y) else float(min(mean_y, 0.8 * cap))
            head_map[task].bias.data.fill_(inv_softplus_stable(target))


# --------------------------
# Training per fold
# --------------------------


def train_one_fold(
    *,
    X: np.ndarray,
    Ys: np.ndarray,
    Yi: np.ndarray,
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

    X_use = X
    keep = None
    if args.drop_zero_var_thresh is not None:
        s = X[train_idx].std(axis=0)
        keep = s >= args.drop_zero_var_thresh
        X_use = X[:, keep]
        np.save(outdir / "keep_mask.npy", keep)

    ds = MultiTaskScaleDataset(X_use, Ys, Yi)
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

    # caps + bias init from train only
    y_tr = {"snv": Ys[train_idx], "indel": Yi[train_idx]}
    mu_caps = {
        t: float(max(10.0, np.nanpercentile(y_tr[t], 99.9) * 2.0))
        for t in TASKS
    }
    y_means = {t: float(np.nanmean(y_tr[t])) for t in TASKS}

    model = MultiTaskSingleScale(
        f=X_use.shape[1],
        hidden=256,
        dropout=args.dropout,
        loss_type=args.loss,
        mu_caps=mu_caps,
        fix_theta=args.fix_theta,
    ).to(device)
    init_mu_biases(model, y_means, mu_caps)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def primary_score(metrics: Dict) -> float:
        if args.primary_task == "both_mean":
            return float(np.nanmean([metrics[t]["r2"] for t in TASKS]))
        return float(metrics[args.primary_task]["r2"])

    best = -1e30
    best_info = None
    no_imp = 0
    log_rows: List[List[float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        nobs = 0
        for x, ys_b, yi_b in train_loader:
            x = x.to(device)
            ys_b = ys_b.to(device)
            yi_b = yi_b.to(device)

            mu, th = model(x)

            # SNV
            mask_s = torch.isfinite(ys_b)
            if mask_s.any():
                loss_s = _loss_one(args.loss, mu["snv"][mask_s], ys_b[mask_s], th["snv"])
            else:
                loss_s = torch.tensor(0.0, device=device)

            # INDEL
            mask_i = torch.isfinite(yi_b)
            if mask_i.any():
                loss_i = _loss_one(args.loss, mu["indel"][mask_i], yi_b[mask_i], th["indel"])
            else:
                loss_i = torch.tensor(0.0, device=device)

            def UW(loss, logsigma):
                return torch.exp(-2 * logsigma) * loss + 2 * logsigma

            total = UW(loss_s, model.logsigma_snv) + UW(loss_i, model.logsigma_indel)

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            tot += total.item() * len(x)
            nobs += len(x)

        tr_loss = tot / max(nobs, 1)
        metrics, theta_avg = evaluate(model, val_loader, device)
        score = primary_score(metrics)

        # log row
        flat = []
        for t in TASKS:
            for name in ["mae", "mse", "r2", "pm", "ps"]:
                flat.append(metrics[t][name])
        log_rows.append(
            [
                epoch,
                tr_loss,
                score,
                float(model.logsigma_snv.item()),
                float(model.logsigma_indel.item()),
            ]
            + flat
            + [theta_avg[t] for t in TASKS]
        )

        if score > best:
            best = float(score)
            no_imp = 0
            torch.save(model.state_dict(), outdir / "best_model.pt")
            best_info = {
                "epoch": epoch,
                "train_loss": float(tr_loss),
                "primary_task": args.primary_task,
                "primary_score": float(score),
                "metrics": metrics,
                "theta_avg": theta_avg,
                "logsigma_snv": float(model.logsigma_snv.item()),
                "logsigma_indel": float(model.logsigma_indel.item()),
                "mu_caps": mu_caps,
                "y_means": y_means,
                "run_seed": int(run_seed),
            }
        else:
            no_imp += 1
            if no_imp >= args.patience:
                break

    headers = (
        ["epoch", "trainLoss", "primaryScore", "logsigma_snv", "logsigma_indel"]
        + [f"{t}_{n}" for t in TASKS for n in ["valMAE", "valMSE", "valR2", "predMean", "predStd"]]
        + [f"thetaAvg_{t}" for t in TASKS]
    )
    with open(outdir / "train_log.tsv", "w", newline="") as fh:
        csv.writer(fh, delimiter="\t").writerows([headers, *log_rows])

    if best_info is None:
        best_info = {
            "epoch": None,
            "train_loss": None,
            "primary_task": args.primary_task,
            "primary_score": None,
            "metrics": None,
            "theta_avg": None,
            "logsigma_snv": None,
            "logsigma_indel": None,
            "mu_caps": mu_caps,
            "y_means": y_means,
            "run_seed": int(run_seed),
        }

    with open(outdir / "best_epoch.json", "w") as fh:
        json.dump(best_info, fh, indent=2)

    if getattr(model, "last_gate_mean", None) is not None:
        np.savetxt(outdir / "gates_mean.tsv", model.last_gate_mean.cpu().numpy(), delimiter="\t")

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
    for t in TASKS:
        for k, v in metrics[t].items():
            out[f"{t}_{k}"] = float(v) if v is not None else np.nan
    return out


def summarize_cv(rows: List[Dict]) -> Dict:
    if not rows:
        return {"n": 0, "mean": {}, "std": {}}
    keys = set()
    for r in rows:
        keys.update(
            k
            for k, v in r.items()
            if isinstance(v, (int, float)) and k not in {"repeat", "fold", "seed"}
        )
    mean = {}
    std = {}
    for k in sorted(keys):
        vals = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        mean[k] = float(np.nanmean(vals))
        std[k] = float(np.nanstd(vals))
    return {"n": int(len(rows)), "mean": mean, "std": std}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["1mb", "100kb", "10kb"], required=True)

    # CA paths for each scale
    ap.add_argument("--ca_1mb", required=True)
    ap.add_argument("--ca_100kb", required=True)
    ap.add_argument("--ca_10kb", required=True)
    # SNV paths for each scale
    ap.add_argument("--snv_1mb", required=True)
    ap.add_argument("--snv_100kb", required=True)
    ap.add_argument("--snv_10kb", required=True)
    # INDEL paths for each scale
    ap.add_argument("--indel_1mb", required=True)
    ap.add_argument("--indel_100kb", required=True)
    ap.add_argument("--indel_10kb", required=True)

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
    )

    # early stopping target
    ap.add_argument("--primary_task", choices=["snv", "indel", "both_mean"], default="snv")

    # CV controls
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--repeat_id", type=int, default=None)
    ap.add_argument("--fold_id", type=int, default=None)
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    scale = SCALE_ALIASES[args.scale]
    if scale == "1mb":
        ca_path = Path(args.ca_1mb)
        snv_path = Path(args.snv_1mb)
        indel_path = Path(args.indel_1mb)
    elif scale == "100kb":
        ca_path = Path(args.ca_100kb)
        snv_path = Path(args.snv_100kb)
        indel_path = Path(args.indel_100kb)
    else:
        ca_path = Path(args.ca_10kb)
        snv_path = Path(args.snv_10kb)
        indel_path = Path(args.indel_10kb)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "args.json", "w") as fh:
        json.dump(vars(args), fh, indent=2)

    print(
        f"========== CV MultiTaskSingleScale | scale={scale} | ctype={args.ctype} | "
        f"folds={args.folds} repeats={args.repeats} =========="
    )

    X, Ys, Yi, X_cols = prepare_single_scale_arrays(
        ca_path=ca_path,
        snv_path=snv_path,
        indel_path=indel_path,
        ctype=args.ctype,
        zscore=args.zscore,
        feature_clip=args.feature_clip,
        missing_targets=args.missing_targets,
    )
    n = len(Ys)
    print(f"Prepared arrays: X={X.shape}  N={n}")

    rows: List[Dict] = []
    idx = np.arange(n)
    for rep in range(args.repeats):
        if args.repeat_id is not None and rep != args.repeat_id:
            continue
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed + rep)
        for fold, (tr, va) in enumerate(kf.split(idx)):
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
                    X=X,
                    Ys=Ys,
                    Yi=Yi,
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
    summary["scale"] = scale
    summary["folds"] = args.folds
    summary["repeats"] = args.repeats
    with open(outdir / "cv_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\nSaved CV outputs in:", str(outdir))


if __name__ == "__main__":
    main()
