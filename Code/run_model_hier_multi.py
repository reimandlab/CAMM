#!/usr/bin/env python3
"""run_model_hier_multi.py

Hierarchical multi-task model (SNV + INDEL) with coarse-to-fine context:

  1) 1MB branch -> shared_1 -> h1 -> heads for 1MB (SNV & INDEL)
  2) 100KB branch + h1 -> shared_2 -> h2 -> heads for 100KB (SNV & INDEL)
  3) 10KB branch + h1 + h2 -> shared_3 -> h3 -> heads for 10KB (SNV / INDEL)

Key behaviors:
  • Multi-scale sharing: 1MB representation conditions 100KB; (1MB,100KB) condition 10KB.
  • Multi-task sharing: SNV and INDEL share all feature towers; only heads are task-specific.
  • Targets can be treated as:
      - missing (NaN): partial supervision
      - zero: absent rows in mutation tables are interpreted as 0 mutations
        (common when mutation tables are stored sparsely: only non-zero bins are present).
"""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCALES = ["1mb","100kb","10kb"]
TASKS  = ["snv","indel"]

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
    return df_mut[["chr","start", c]].rename(columns={c: "y"})

def _zscore_inplace(A: np.ndarray) -> None:
    m = np.nanmean(A, axis=0, keepdims=True)
    s = np.nanstd(A, axis=0, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    A -= m
    A /= s

def _sanitize_inplace(A: np.ndarray, fill: float = 0.0, clip_abs: float|None = None) -> None:
    np.nan_to_num(A, copy=False, nan=fill, posinf=fill, neginf=fill)
    if clip_abs is not None:
        np.clip(A, -clip_abs, clip_abs, out=A)

def _feat_cols(df): 
    return [c for c in df.columns if c not in ('chr','start')]

def prepare_arrays(
    ca_1mb: Path, ca_100kb: Path, ca_10kb: Path,
    snv_1mb: Path, snv_100kb: Path, snv_10kb: Path,
    indel_1mb: Path, indel_100kb: Path, indel_10kb: Path,
    ctype: str,
    zscore: bool=False,
    feature_clip: float|None=None,
    missing_targets: str = "zero",
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

    # Quick sanity: how many windows are in each file?
    def _nuniq(df: pd.DataFrame) -> int:
        return int(df[['chr','start']].drop_duplicates().shape[0])
    print(
        "CA windows (unique chr+start): "
        f"1MB={_nuniq(ca1)}  100KB={_nuniq(ca2)}  10KB={_nuniq(ca3)}"
    )

    for df,name in [
        (ca1,'CA_1MB'),(ca2,'CA_100KB'),(ca3,'CA_10KB'),
        (s1,'SNV_1MB'),(s2,'SNV_100KB'),(s3,'SNV_10KB'),
        (i1,'INDEL_1MB'),(i2,'INDEL_100KB'),(i3,'INDEL_10KB')
    ]:
        if not {'chr','start'}.issubset(df.columns):
            raise KeyError(f"{name} must contain 'chr' and 'start'")

    # Base grid from 10kb CA
    base = ca3[['chr','start']].copy()
    base['start'] = pd.to_numeric(base['start'], errors='coerce').fillna(0).astype('int64')
    base['start_100kb'] = ((base['start'] - 1) // 100_000) * 100_000 + 1
    base['start_1mb']   = ((base['start'] - 1) // 1_000_000) * 1_000_000 + 1

    # Merge CA features
    X1_cols = _feat_cols(ca1)
    base = base.merge(
        ca1.rename(columns={'start':'start_1mb'}),
        on=['chr','start_1mb'], how='left'
    )
    X2_cols = _feat_cols(ca2)
    base = base.merge(
        ca2.rename(columns={'start':'start_100kb'}),
        on=['chr','start_100kb'], how='left'
    )
    X3_cols = _feat_cols(ca3)
    base = base.merge(ca3, on=['chr','start'], how='left')

    # Merge targets (keep NaNs for missing windows)
    y_snv_1 = _select_ct(s1, ctype).rename(columns={'y':'y_snv_1mb'})
    y_snv_2 = _select_ct(s2, ctype).rename(columns={'y':'y_snv_100kb'})
    y_snv_3 = _select_ct(s3, ctype).rename(columns={'y':'y_snv_10kb'})
    y_ind_1 = _select_ct(i1, ctype).rename(columns={'y':'y_ind_1mb'})
    y_ind_2 = _select_ct(i2, ctype).rename(columns={'y':'y_ind_100kb'})
    y_ind_3 = _select_ct(i3, ctype).rename(columns={'y':'y_ind_10kb'})

    print(
        f"Mutation windows for '{ctype}' (unique chr+start): "
        f"SNV 1MB={_nuniq(y_snv_1)} 100KB={_nuniq(y_snv_2)} 10KB={_nuniq(y_snv_3)} | "
        f"INDEL 1MB={_nuniq(y_ind_1)} 100KB={_nuniq(y_ind_2)} 10KB={_nuniq(y_ind_3)}"
    )

    base = base.merge(
        y_snv_1.rename(columns={'start':'start_1mb'}),
        on=['chr','start_1mb'], how='left'
    )
    base = base.merge(
        y_snv_2.rename(columns={'start':'start_100kb'}),
        on=['chr','start_100kb'], how='left'
    )
    base = base.merge(y_snv_3, on=['chr','start'], how='left')
    base = base.merge(
        y_ind_1.rename(columns={'start':'start_1mb'}),
        on=['chr','start_1mb'], how='left'
    )
    base = base.merge(
        y_ind_2.rename(columns={'start':'start_100kb'}),
        on=['chr','start_100kb'], how='left'
    )
    base = base.merge(y_ind_3, on=['chr','start'], how='left')

    # ---- Missingness report (before optional filling) ----
    before = len(base)
    need = [
        'y_snv_1mb','y_snv_100kb','y_snv_10kb',
        'y_ind_1mb','y_ind_100kb','y_ind_10kb'
    ]
    print(f"Total CA 10kb windows in base grid: {before}")
    print("\n--- Missingness summary per target ---")
    for col in need:
        miss = base[col].isna().sum()
        present = before - miss
        frac = miss / before if before > 0 else 0.0
        print(f"{col:11s}: missing={miss:7d}  present={present:7d}  (missing frac={frac:.4f})")

    # Many mutation tables are stored sparsely (only non-zero windows are present).
    # In that common case, NaN after merge means "0 mutations", not "unknown".
    if missing_targets == "zero":
        n_filled = 0
        for col in need:
            m = int(base[col].isna().sum())
            if m:
                base[col] = base[col].fillna(0.0)
                n_filled += m
        if n_filled:
            print(f"Filled {n_filled} missing target values with 0.0 (missing_targets=zero)")
        # Re-report to make it explicit that training will see all windows
        print("\n--- Missingness after filling (targets) ---")
        for col in need:
            miss = base[col].isna().sum()
            present = before - miss
            frac = miss / before if before > 0 else 0.0
            print(f"{col:11s}: missing={miss:7d}  present={present:7d}  (missing frac={frac:.4f})")

    base = base.reset_index(drop=True)

    # Features
    X1 = base[X1_cols].to_numpy(np.float32, copy=False)
    X2 = base[X2_cols].to_numpy(np.float32, copy=False)
    X3 = base[X3_cols].to_numpy(np.float32, copy=False)

    # Targets: keep NaNs to indicate missing, clip negatives to 0
    def clip_targets(a: np.ndarray) -> np.ndarray:
        arr = a.astype(np.float32, copy=False)
        # keep NaNs, only enforce non-negativity where finite
        mask = np.isfinite(arr) & (arr < 0)
        arr[mask] = 0.0
        return arr

    Y_snv_1 = clip_targets(base['y_snv_1mb'].to_numpy(np.float32, copy=False))
    Y_snv_2 = clip_targets(base['y_snv_100kb'].to_numpy(np.float32, copy=False))
    Y_snv_3 = clip_targets(base['y_snv_10kb'].to_numpy(np.float32, copy=False))
    Y_ind_1 = clip_targets(base['y_ind_1mb'].to_numpy(np.float32, copy=False))
    Y_ind_2 = clip_targets(base['y_ind_100kb'].to_numpy(np.float32, copy=False))
    Y_ind_3 = clip_targets(base['y_ind_10kb'].to_numpy(np.float32, copy=False))

    # Sanitize features (not targets)
    for X in (X1, X2, X3):
        _sanitize_inplace(X, fill=0.0)
    if zscore:
        _zscore_inplace(X1); _zscore_inplace(X2); _zscore_inplace(X3)
    if feature_clip is not None:
        _sanitize_inplace(X1, clip_abs=feature_clip)
        _sanitize_inplace(X2, clip_abs=feature_clip)
        _sanitize_inplace(X3, clip_abs=feature_clip)

    return (
        X1,X2,X3,
        Y_snv_1,Y_snv_2,Y_snv_3,
        Y_ind_1,Y_ind_2,Y_ind_3,
        X1_cols,X2_cols,X3_cols
    )

class SNVIndelDataset(Dataset):
    def __init__(self, X1,X2,X3, Y_s1,Y_s2,Y_s3, Y_i1,Y_i2,Y_i3):
        n = len(Y_s3)
        for arr in [X1,X2,X3,Y_s1,Y_s2,Y_s3,Y_i1,Y_i2,Y_i3]:
            assert len(arr) == n
        self.x1 = torch.from_numpy(X1).float()
        self.x2 = torch.from_numpy(X2).float()
        self.x3 = torch.from_numpy(X3).float()
        self.ys1 = torch.from_numpy(Y_s1).float()
        self.ys2 = torch.from_numpy(Y_s2).float()
        self.ys3 = torch.from_numpy(Y_s3).float()
        self.yi1 = torch.from_numpy(Y_i1).float()
        self.yi2 = torch.from_numpy(Y_i2).float()
        self.yi3 = torch.from_numpy(Y_i3).float()

    def __len__(self):
        return len(self.ys3)

    def __getitem__(self, i):
        return (
            self.x1[i], self.x2[i], self.x3[i],
            self.ys1[i], self.ys2[i], self.ys3[i],
            self.yi1[i], self.yi2[i], self.yi3[i]
        )

# ------------- Model -------------

class GatingLayer(nn.Module):
    def __init__(self, input_size:int):
        super().__init__()
        self.lin = nn.Linear(input_size, input_size)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
    def forward(self, x):
        g = torch.sigmoid(self.lin(x))
        return x * g, g

class Branch(nn.Module):
    def __init__(self, input_size:int, out_dim:int=256, dropout:float=0.3):
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
    """
    Shared hierarchy for 1MB/100KB/10KB; task-specific heads:
      mu{scale}_{task}  with task in {snv, indel}
    """
    def __init__(self, f1:int,f2:int,f3:int, branch_dim:int=256, hidden:int=256,
                 dropout:float=0.3, loss_type:str='poisson',
                 mu_caps:Dict[str,Dict[str,float]]|None=None, fix_theta:float|None=None):
        super().__init__()
        self.loss_type = loss_type
        self.mu_caps = mu_caps or {t:{s:1e6 for s in SCALES} for t in TASKS}
        self.fix_theta = fix_theta

        self.b1 = Branch(f1, out_dim=branch_dim, dropout=dropout)
        self.b2 = Branch(f2, out_dim=branch_dim, dropout=dropout)
        self.b3 = Branch(f3, out_dim=branch_dim, dropout=dropout)

        # Shared at 1MB and 100KB
        self.shared1 = nn.Sequential(
            nn.Linear(branch_dim, hidden), nn.ReLU(),
            nn.BatchNorm1d(hidden), nn.Dropout(dropout)
        )
        self.shared2 = nn.Sequential(
            nn.Linear(branch_dim + hidden, hidden), nn.ReLU(),
            nn.BatchNorm1d(hidden), nn.Dropout(dropout)
        )

        # 10KB tower is SHARED across SNV & INDEL; only heads are task-specific.
        self.shared3 = nn.Sequential(
            nn.Linear(branch_dim + hidden + hidden, hidden), nn.ReLU(),
            nn.BatchNorm1d(hidden), nn.Dropout(dropout)
        )

        def head():
            return nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())

        # Per-scale, per-task heads
        self.mu_1mb_snv   = head(); self.mu_1mb_indel   = head()
        self.mu_100kb_snv = head(); self.mu_100kb_indel = head()
        self.mu_10kb_snv  = head(); self.mu_10kb_indel  = head()

        # theta (for NB)
        if fix_theta is None:
            self.theta_1mb_snv     = nn.Parameter(torch.tensor([10.0]))
            self.theta_1mb_indel   = nn.Parameter(torch.tensor([10.0]))
            self.theta_100kb_snv   = nn.Parameter(torch.tensor([10.0]))
            self.theta_100kb_indel = nn.Parameter(torch.tensor([10.0]))
            self.theta_10kb_snv    = nn.Parameter(torch.tensor([10.0]))
            self.theta_10kb_indel  = nn.Parameter(torch.tensor([10.0]))
        else:
            self.register_buffer('theta_fixed', torch.tensor([fix_theta], dtype=torch.float32))

        # Uncertainty weighting parameters (learned task weights)
        self.logsigma_snv   = nn.Parameter(torch.tensor(0.0))
        self.logsigma_indel = nn.Parameter(torch.tensor(0.0))

        self.last_gate_means = (None,None,None)

    def forward(self, x1,x2,x3):
        e1,g1 = self.b1(x1)
        h1    = self.shared1(e1)
        e2,g2 = self.b2(x2)
        h2    = self.shared2(torch.cat([e2,h1], dim=1))
        e3,g3 = self.b3(x3)

        # Shared 10kb tower
        h3 = self.shared3(torch.cat([e3,h1,h2], dim=1))

        self.last_gate_means = (
            g1.mean(0).detach(),
            g2.mean(0).detach(),
            g3.mean(0).detach()
        )

        def clamp(mu, task, scale):
            return torch.clamp(
                mu.squeeze(-1),
                1e-6,
                self.mu_caps[task][scale]
            )

        mu = {
            "snv": {
                "1mb":   clamp(self.mu_1mb_snv(h1),       "snv","1mb"),
                "100kb": clamp(self.mu_100kb_snv(h2),     "snv","100kb"),
                "10kb":  clamp(self.mu_10kb_snv(h3),      "snv","10kb"),
            },
            "indel": {
                "1mb":   clamp(self.mu_1mb_indel(h1),       "indel","1mb"),
                "100kb": clamp(self.mu_100kb_indel(h2),     "indel","100kb"),
                "10kb":  clamp(self.mu_10kb_indel(h3),      "indel","10kb"),
            },
        }

        if self.fix_theta is None:
            def sp(x): return F.softplus(x)
            th = {
                "snv": {
                    "1mb":   sp(self.theta_1mb_snv),
                    "100kb": sp(self.theta_100kb_snv),
                    "10kb":  sp(self.theta_10kb_snv),
                },
                "indel": {
                    "1mb":   sp(self.theta_1mb_indel),
                    "100kb": sp(self.theta_100kb_indel),
                    "10kb":  sp(self.theta_10kb_indel),
                },
            }
        else:
            th = {
                "snv":   {s: self.theta_fixed for s in SCALES},
                "indel": {s: self.theta_fixed for s in SCALES},
            }
        return mu, th

# ------------- Loss/metrics -------------

def poisson_nll(mu, y):
    return nn.PoissonNLLLoss(
        log_input=False, full=True, reduction='mean'
    )(mu, y)

def nb_nll_stable(y, mu, theta, eps: float = 1e-8):
    y     = torch.clamp(y,     0,      1e12)
    theta = torch.clamp(theta, 1e-6,   1e12)
    log_theta   = torch.log(theta + eps)
    log_mu      = torch.log(mu + eps)
    log_theta_mu = log_theta + torch.log1p(mu/(theta+eps))
    ll = (
        torch.lgamma(y+theta) - torch.lgamma(theta) - torch.lgamma(y+1.0)
        + theta*(log_theta-log_theta_mu)
        + y*(log_mu-log_theta_mu)
    )
    return -ll.mean()

def _loss_one(loss_type, mu, y, theta):
    if loss_type == 'mse':
        return F.mse_loss(mu, y)
    if loss_type == 'poisson':
        return poisson_nll(mu, y)
    return nb_nll_stable(y, mu, theta)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    P  = {t:{s:[] for s in SCALES} for t in TASKS}
    Y  = {t:{s:[] for s in SCALES} for t in TASKS}
    TH = {t:{s:[] for s in SCALES} for t in TASKS}

    for batch in loader:
        (x1,x2,x3, ys1,ys2,ys3, yi1,yi2,yi3) = batch
        x1,x2,x3 = x1.to(device), x2.to(device), x3.to(device)
        mu,th   = model(x1,x2,x3)
        ysnv = {"1mb":ys1,"100kb":ys2,"10kb":ys3}
        yind = {"1mb":yi1,"100kb":yi2,"10kb":yi3}
        for s in SCALES:
            # predictions are always finite; keep tensors as-is
            P["snv"][s].append(
                torch.nan_to_num(
                    mu["snv"][s],
                    nan=0.0, posinf=1e12, neginf=0.0
                ).cpu()
            )
            Y["snv"][s].append(ysnv[s])
            P["indel"][s].append(
                torch.nan_to_num(
                    mu["indel"][s],
                    nan=0.0, posinf=1e12, neginf=0.0
                ).cpu()
            )
            Y["indel"][s].append(yind[s])
            TH["snv"][s].append(th["snv"][s].item())
            TH["indel"][s].append(th["indel"][s].item())

    def stats(PY):
        p = torch.cat(PY[0]).numpy().astype(np.float64)
        y = torch.cat(PY[1]).numpy().astype(np.float64)
        mask = np.isfinite(y)
        if not np.any(mask):
            # no valid targets for this task/scale in val set
            return (np.nan, np.nan, np.nan, float(np.mean(p)), float(np.std(p)))
        p = p[mask]
        y = y[mask]
        return (
            mean_absolute_error(y, p),
            mean_squared_error(y, p),
            r2_score(y, p),
            float(np.mean(p)),
            float(np.std(p))
        )

    metrics = {t:{s:{} for s in SCALES} for t in TASKS}
    for t in TASKS:
        for s in SCALES:
            mae,mse,r2,pm,ps = stats((P[t][s], Y[t][s]))
            metrics[t][s] = {"mae":mae,"mse":mse,"r2":r2,"pm":pm,"ps":ps}

    theta_avg = {
        t: {
            s: float(np.mean(TH[t][s])) if TH[t][s] else np.nan
            for s in SCALES
        }
        for t in TASKS
    }
    return metrics, theta_avg

def inv_softplus_stable(y: float, eps: float = 1e-8) -> float:
    y = max(y, eps)
    if y < 20.0:
        return float(np.log(np.expm1(y)))
    return float(y + np.log1p(-np.exp(-y)))

def init_mu_biases(model: nn.Module, y_means: Dict[str,Dict[str,float]],
                   mu_caps: Dict[str,Dict[str,float]]):
    with torch.no_grad():
        head_map = {
            ("snv","1mb"):   model.mu_1mb_snv[0],
            ("snv","100kb"):model.mu_100kb_snv[0],
            ("snv","10kb"): model.mu_10kb_snv[0],
            ("indel","1mb"):   model.mu_1mb_indel[0],
            ("indel","100kb"): model.mu_100kb_indel[0],
            ("indel","10kb"):  model.mu_10kb_indel[0],
        }
        for task in TASKS:
            for s in SCALES:
                cap    = mu_caps[task][s]
                mean_y = y_means[task][s]
                # If mean_y is NaN (no data at this scale), fall back to small value
                if not np.isfinite(mean_y):
                    target = 1.0
                else:
                    target = float(min(mean_y, 0.8*cap))
                pre = inv_softplus_stable(target)
                head_map[(task,s)].bias.data.fill_(pre)

# ------------- Main -------------

def main():
    ap = argparse.ArgumentParser()
    # CA
    ap.add_argument('--ca_1mb', required=True)
    ap.add_argument('--ca_100kb', required=True)
    ap.add_argument('--ca_10kb', required=True)
    # SNV
    ap.add_argument('--snv_1mb', required=True)
    ap.add_argument('--snv_100kb', required=True)
    ap.add_argument('--snv_10kb', required=True)
    # INDEL
    ap.add_argument('--indel_1mb', required=True)
    ap.add_argument('--indel_100kb', required=True)
    ap.add_argument('--indel_10kb', required=True)

    ap.add_argument('--ctype', required=True)
    ap.add_argument('--loss', choices=['mse','poisson','nb'], default='poisson')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--zscore', action='store_true')
    ap.add_argument('--feature_clip', type=float, default=None)
    ap.add_argument('--drop_zero_var_thresh', type=float, default=None)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--theta_l2', type=float, default=0.0)
    ap.add_argument('--fix_theta', type=float, default=None)

    ap.add_argument(
        '--missing_targets',
        choices=['nan','zero'],
        default='zero',
        help=(
            "How to treat missing windows after merging mutation targets. "
            "'nan' keeps partial supervision; 'zero' fills missing with 0. "
            "Use 'zero' when your mutation CSVs are sparse (only non-zero bins are present)."
        )
    )

    # early stopping target
    ap.add_argument('--primary_task', choices=['snv','indel','both_mean'], default='snv')
    ap.add_argument('--primary_scale', choices=['10kb','100kb','1mb','all_mean'], default='10kb')

    # scale weights (shared by both tasks)
    ap.add_argument('--w1mb', type=float, default=1.0)
    ap.add_argument('--w100kb', type=float, default=1.0)
    ap.add_argument('--w10kb', type=float, default=1.0)

    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"========== CANCER TYPE: {args.ctype} | Hierarchical Multi-task (SNV+INDEL) ==========")
    print("Fine-scale sharing: SHARED 10kb tower (task-specific heads only)")
    print("Task weighting: Uncertainty weighting (learned logsigma per task)")
    print(f"Targets: missing_targets={args.missing_targets} (nan=partial, zero=fill as 0)")

    X1,X2,X3, Ys1,Ys2,Ys3, Yi1,Yi2,Yi3, C1,C2,C3 = prepare_arrays(
        Path(args.ca_1mb), Path(args.ca_100kb), Path(args.ca_10kb),
        Path(args.snv_1mb), Path(args.snv_100kb), Path(args.snv_10kb),
        Path(args.indel_1mb), Path(args.indel_100kb), Path(args.indel_10kb),
        ctype=args.ctype,
        zscore=args.zscore,
        feature_clip=args.feature_clip,
        missing_targets=args.missing_targets,
    )
    print(f"Prepared arrays: X1MB={X1.shape} X100KB={X2.shape} X10KB={X3.shape}  N={len(X3)}")

    ds = SNVIndelDataset(X1,X2,X3, Ys1,Ys2,Ys3, Yi1,Yi2,Yi3)
    n_train = int(0.8*len(ds))
    n_val   = len(ds) - n_train
    train_ds, val_ds = random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # variance on train split (for optional dropping)
    if hasattr(train_ds, "indices"):
        tr_idx = np.array(train_ds.indices)
    else:
        tr_idx = np.random.RandomState(args.seed).choice(
            len(ds), size=n_train, replace=False
        )
    X1_tr, X2_tr, X3_tr = X1[tr_idx], X2[tr_idx], X3[tr_idx]

    def var_report(X, name, cols):
        s = X.std(axis=0)
        print(
            f"{name} features: {X.shape[1]}  "
            f"zero-std frac (train): {float(np.mean(s<1e-8)):.4f}"
        )
        return s

    s1 = var_report(X1_tr,"1MB",C1)
    s2 = var_report(X2_tr,"100KB",C2)
    s3 = var_report(X3_tr,"10KB",C3)

    if args.drop_zero_var_thresh is not None:
        def drop_low_var(X_all, s, th, nm):
            keep = s >= th
            print(f"Dropping {np.sum(~keep)} {nm} features with std<{th}")
            return X_all[:, keep]
        X1 = drop_low_var(X1, s1, args.drop_zero_var_thresh, "1MB")
        X2 = drop_low_var(X2, s2, args.drop_zero_var_thresh, "100KB")
        X3 = drop_low_var(X3, s3, args.drop_zero_var_thresh, "10KB")
        ds = SNVIndelDataset(X1,X2,X3, Ys1,Ys2,Ys3, Yi1,Yi2,Yi3)
        n_train = int(0.8*len(ds))
        n_val   = len(ds) - n_train
        train_ds, val_ds = random_split(
            ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed)
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, drop_last=True, pin_memory=True
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=args.batch_size,
        shuffle=False, pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # baselines (val) – NaN-aware
    with torch.no_grad():
        yv_snv = {s:[] for s in SCALES}
        yv_ind = {s:[] for s in SCALES}
        for b in val_loader:
            _,_,_, s1b,s2b,s3b, i1b,i2b,i3b = b
            yv_snv["1mb"].append(s1b)
            yv_snv["100kb"].append(s2b)
            yv_snv["10kb"].append(s3b)
            yv_ind["1mb"].append(i1b)
            yv_ind["100kb"].append(i2b)
            yv_ind["10kb"].append(i3b)
        for s in SCALES:
            for t, yv in [("SNV", yv_snv), ("INDEL", yv_ind)]:
                y = torch.cat(yv[s]).numpy().astype(np.float64)
                mask = np.isfinite(y)
                if not np.any(mask):
                    print(f"Val baseline {t} {s.upper()}: no finite targets in val split")
                    continue
                y = y[mask]
                mse_zero = mean_squared_error(y, np.zeros_like(y))
                mse_mean = mean_squared_error(y, np.full_like(y, y.mean()))
                print(
                    f"Val baseline {t} {s.upper()}: "
                    f"MSE_zero={mse_zero:.4f}  MSE_mean={mse_mean:.4f}"
                )

    # caps + init (separate per task/scale), NaN-aware
    if hasattr(train_ds,"indices"):
        tr_idx = np.array(train_ds.indices)
    else:
        tr_idx = np.random.RandomState(args.seed).choice(
            len(ds), size=int(0.8*len(ds)), replace=False
        )

    y_tr = {
        "snv":   {"1mb":Ys1[tr_idx], "100kb":Ys2[tr_idx], "10kb":Ys3[tr_idx]},
        "indel": {"1mb":Yi1[tr_idx], "100kb":Yi2[tr_idx], "10kb":Yi3[tr_idx]},
    }

    mu_caps = {
        t: {
            s: float(
                max(
                    10.0,
                    np.nanpercentile(y_tr[t][s], 99.9) * 2.0
                )
            )
            for s in SCALES
        }
        for t in TASKS
    }
    y_means = {
        t: {
            s: float(np.nanmean(y_tr[t][s]))
            for s in SCALES
        }
        for t in TASKS
    }
    print("Caps p99.9*2 (SNV):", {s: round(mu_caps["snv"][s],3) for s in SCALES})
    print("Caps p99.9*2 (INDEL):", {s: round(mu_caps["indel"][s],3) for s in SCALES})

    model = HierMulti(
        X1.shape[1], X2.shape[1], X3.shape[1],
        branch_dim=256, hidden=256, dropout=args.dropout,
        loss_type=args.loss, mu_caps=mu_caps, fix_theta=args.fix_theta
    ).to(device)
    init_mu_biases(model, y_means, mu_caps)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    w_scales = {"1mb":args.w1mb, "100kb":args.w100kb, "10kb":args.w10kb}

    def primary_score(metrics):
        # metrics[task][scale]["r2"]
        def mean_task(task):
            return float(np.nanmean([metrics[task][s]["r2"] for s in SCALES]))
        def mean_scale(scale):
            return float(np.nanmean([metrics[t][scale]["r2"] for t in TASKS]))
        if args.primary_task == "both_mean" and args.primary_scale == "all_mean":
            return float(np.nanmean(
                [metrics[t][s]["r2"] for t in TASKS for s in SCALES]
            ))
        if args.primary_task == "both_mean":
            return mean_scale(args.primary_scale)
        if args.primary_scale == "all_mean":
            return mean_task(args.primary_task)
        return float(metrics[args.primary_task][args.primary_scale]["r2"])

    best = -1e30
    best_info = None
    no_imp = 0
    log = []

    for epoch in range(1, args.epochs+1):
        # ---------------- train ----------------
        model.train()
        tot = 0.0
        nobs = 0
        for b in train_loader:
            (x1,x2,x3, ys1,ys2,ys3, yi1,yi2,yi3) = b
            x1,x2,x3 = x1.to(device),x2.to(device),x3.to(device)
            ys1,ys2,ys3 = ys1.to(device),ys2.to(device),ys3.to(device)
            yi1,yi2,yi3 = yi1.to(device),yi2.to(device),yi3.to(device)

            mu,th = model(x1,x2,x3)

            y_snv = {"1mb":ys1,"100kb":ys2,"10kb":ys3}
            y_ind = {"1mb":yi1,"100kb":yi2,"10kb":yi3}

            loss_snv_scales  = {}
            loss_ind_scales  = {}

            for s in SCALES:
                # SNV
                mask_snv = torch.isfinite(y_snv[s])
                if mask_snv.any():
                    loss_snv_scales[s] = _loss_one(
                        args.loss,
                        mu["snv"][s][mask_snv],
                        y_snv[s][mask_snv],
                        th["snv"][s]
                    )
                else:
                    loss_snv_scales[s] = torch.tensor(0.0, device=device)

                # INDEL
                mask_ind = torch.isfinite(y_ind[s])
                if mask_ind.any():
                    loss_ind_scales[s] = _loss_one(
                        args.loss,
                        mu["indel"][s][mask_ind],
                        y_ind[s][mask_ind],
                        th["indel"][s]
                    )
                else:
                    loss_ind_scales[s] = torch.tensor(0.0, device=device)

            # sum over scales with user scale weights
            loss_snv   = sum(w_scales[s]*loss_snv_scales[s]   for s in SCALES)
            loss_indel = sum(w_scales[s]*loss_ind_scales[s]   for s in SCALES)

            # Uncertainty weighting
            def UW(loss, logsigma):
                return torch.exp(-2*logsigma) * loss + 2*logsigma
            total = UW(loss_snv, model.logsigma_snv) + UW(loss_indel, model.logsigma_indel)

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            tot   += total.item() * len(x1)
            nobs  += len(x1)

        tr_loss = tot / max(nobs, 1)

        # ---------------- val ----------------
        metrics, theta_avg = evaluate(model, val_loader, device)
        score = primary_score(metrics)

        r2_line = " | ".join([
            f"{t.upper()} R2 [1MB|100KB|10KB]= "
            f"{metrics[t]['1mb']['r2']:.4f}|"
            f"{metrics[t]['100kb']['r2']:.4f}|"
            f"{metrics[t]['10kb']['r2']:.4f}"
            for t in TASKS
        ])
        print(
            f"Epoch {epoch}/{args.epochs}  trainLoss={tr_loss:.4f}  {r2_line}  "
            f"primary=({args.primary_task},{args.primary_scale}):{score:.4f}  "
            f"logsigma_snv={model.logsigma_snv.item():.3f} "
            f"logsigma_indel={model.logsigma_indel.item():.3f}"
        )

        # flat log row
        flat = []
        for t in TASKS:
            for name in ["mae","mse","r2","pm","ps"]:
                for s in SCALES:
                    flat.append(metrics[t][s][name])
        th_flat = [theta_avg[t][s] for t in TASKS for s in SCALES]
        log.append([
            epoch, tr_loss, score,
            model.logsigma_snv.item(),
            model.logsigma_indel.item()
        ] + flat + th_flat)

        if score > best:
            best = float(score)
            no_imp = 0
            torch.save(model.state_dict(), outdir/'best_model.pt')
            best_info = {
                "epoch": epoch,
                "train_loss": float(tr_loss),
                "primary_task": args.primary_task,
                "primary_scale": args.primary_scale,
                "primary_score": float(score),
                "metrics": metrics,
                "theta_avg": theta_avg,
                "logsigma_snv": float(model.logsigma_snv.item()),
                "logsigma_indel": float(model.logsigma_indel.item()),
            }
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print("Early stopping after", epoch, "epochs")
                break

    with open(outdir/'metrics.txt','w') as fh:
        fh.write(
            f"ctype\t{args.ctype}\n"
            f"loss\t{args.loss}\n"
            f"primary_task\t{args.primary_task}\n"
            f"primary_scale\t{args.primary_scale}\n"
            f"best_primary\t{best}\n"
        )

    if hasattr(model, "last_gate_means") and model.last_gate_means[0] is not None:
        g1,g2,g3 = model.last_gate_means
        np.savetxt(outdir/'gates_mean_1MB.tsv',   g1.cpu().numpy(), delimiter='\t')
        np.savetxt(outdir/'gates_mean_100KB.tsv', g2.cpu().numpy(), delimiter='\t')
        np.savetxt(outdir/'gates_mean_10KB.tsv',  g3.cpu().numpy(), delimiter='\t')

    headers = (
        ['epoch','trainLoss','primaryScore','logsigma_snv','logsigma_indel'] +
        [f'{t}_{n}_{s}' for t in TASKS
                        for n in ['valMAE','valMSE','valR2','predMean','predStd']
                        for s in SCALES] +
        [f'thetaAvg_{t}_{s}' for t in TASKS for s in SCALES]
    )
    with open(outdir/'train_log.tsv','w', newline='') as fh:
        csv.writer(fh, delimiter='\t').writerows([headers, *log])

    if best_info is not None:
        with open(outdir/'best_epoch.json','w') as fh:
            json.dump(best_info, fh, indent=2)
        print("\nBest epoch =", round(best_info["primary_score"],4))
        print("Saved: best_model.pt, metrics.txt, train_log.tsv, best_epoch.json in:", outdir)
    else:
        print("Finished. Saved current model outputs in:", outdir)

if __name__ == '__main__':
    main()
