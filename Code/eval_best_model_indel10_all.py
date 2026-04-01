#!/usr/bin/env python3
"""
eval_best_model_indel10_all.py

Evaluate the trained HierMulti best_model.pt on **all** CA windows,
but only for INDEL at 10kb.

- Uses CA+RT features at 1MB / 100KB / 10KB for all windows
- Uses HMF_indel_10KB.csv for observed INDEL counts
    * missing windows are treated as 0 mutations (fillna(0))
- Ignores SNV and INDEL at 1MB / 100KB for evaluation
- Loads best_model.pt
- Outputs:
    * <ctype>_indel_10kb_pred_vs_obs_all.tsv  (all windows)
    * <ctype>_indel_10kb_underest_z>3.0_all.tsv
    * <ctype>_indel_10kb_underest_z>4.0_all.tsv
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SCALES = ["1mb", "100kb", "10kb"]
TASKS = ["snv", "indel"]


# ---------------- I/O helpers ----------------

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


def _sanitize_inplace(A: np.ndarray, fill: float = 0.0, clip_abs: float | None = None) -> None:
    np.nan_to_num(A, copy=False, nan=fill, posinf=fill, neginf=fill)
    if clip_abs is not None:
        np.clip(A, -clip_abs, clip_abs, out=A)


def _zscore_inplace(A: np.ndarray) -> None:
    m = np.nanmean(A, axis=0, keepdims=True)
    s = np.nanstd(A, axis=0, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    A -= m
    A /= s


def _feat_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in ("chr", "start")]


def prepare_features_and_indel10_all(
    ca_1mb: Path,
    ca_100kb: Path,
    ca_10kb: Path,
    indel_10kb: Path,
    ctype: str,
    zscore: bool = True,
    feature_clip: float | None = 8.0,
):
    """
    Build CA+RT feature arrays and INDEL 10kb targets for ALL windows.

    - Base grid: CA 10kb windows (N ~ 264,105)
    - Features: CA at 1MB/100KB/10KB
    - Target: y_indel_10kb, fill missing with 0.0
    """
    # CA
    ca1 = _canon_cols(_read_ca(ca_1mb))
    ca2 = _canon_cols(_read_ca(ca_100kb))
    ca3 = _canon_cols(_read_ca(ca_10kb))

    if not {"chr", "start"}.issubset(ca3.columns):
        raise KeyError("CA 10kb table must contain 'chr' and 'start'")

    # Base grid: all 10kb windows
    base = ca3[["chr", "start"]].copy()
    base["start"] = (
        pd.to_numeric(base["start"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    base["start_100kb"] = ((base["start"] - 1) // 100_000) * 100_000 + 1
    base["start_1mb"] = ((base["start"] - 1) // 1_000_000) * 1_000_000 + 1

    before = len(base)
    print(f"Total CA 10kb windows in base grid: {before}")

    # Feature columns before merging
    X1_cols = _feat_cols(ca1)
    X2_cols = _feat_cols(ca2)
    X3_cols = _feat_cols(ca3)

    # Merge 1MB and 100KB features via start_1mb/start_100kb
    base = base.merge(
        ca1.rename(columns={"start": "start_1mb"}),
        on=["chr", "start_1mb"],
        how="left",
    )
    base = base.merge(
        ca2.rename(columns={"start": "start_100kb"}),
        on=["chr", "start_100kb"],
        how="left",
    )
    # Merge 10kb features directly
    base = base.merge(ca3, on=["chr", "start"], how="left")

    # INDEL 10kb target
    i3 = _canon_cols(_read_mut(indel_10kb))
    y_indel_3 = _select_ct(i3, ctype).rename(columns={"y": "y_indel_10kb"})
    base = base.merge(y_indel_3, on=["chr", "start"], how="left")
    base["y_indel_10kb"] = base["y_indel_10kb"].fillna(0.0)

    # Build feature matrices
    X1 = base[X1_cols].to_numpy(np.float32, copy=False)
    X2 = base[X2_cols].to_numpy(np.float32, copy=False)
    X3 = base[X3_cols].to_numpy(np.float32, copy=False)

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

    y_true = base["y_indel_10kb"].to_numpy(np.float64, copy=False)

    coords = base[["chr", "start", "start_100kb", "start_1mb"]].copy()

    print(f"Prepared arrays: X1MB={X1.shape} X100KB={X2.shape} X10KB={X3.shape}  N={len(base)}")
    return X1, X2, X3, y_true, coords


# ---------------- Model (same architecture) ----------------

class GatingLayer(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.lin = nn.Linear(input_size, input_size)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x: torch.Tensor):
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

    def forward(self, x: torch.Tensor):
        gx, g = self.gate(x)
        e = self.mlp(gx)
        return e, g


class HierMulti(nn.Module):
    """
    Same architecture as in run_model_hier_multi.py.
    For evaluation we only use INDEL_10kb outputs, but we must define all heads
    to load best_model.pt.
    """

    def __init__(
        self,
        f1: int,
        f2: int,
        f3: int,
        branch_dim: int = 256,
        hidden: int = 256,
        dropout: float = 0.3,
        loss_type: str = "poisson",
        mu_caps: Dict[str, Dict[str, float]] | None = None,
        fix_theta: float | None = None,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.mu_caps = mu_caps or {t: {s: 1e6 for s in SCALES} for t in TASKS}
        self.fix_theta = fix_theta

        self.b1 = Branch(f1, out_dim=branch_dim, dropout=dropout)
        self.b2 = Branch(f2, out_dim=branch_dim, dropout=dropout)
        self.b3 = Branch(f3, out_dim=branch_dim, dropout=dropout)

        # Shared at 1MB and 100KB
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

        # 10KB tower is SHARED across SNV & INDEL; only heads are task-specific.
        self.shared3 = nn.Sequential(
            nn.Linear(branch_dim + hidden + hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
        )

        def head():
            return nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())

        # per scale, per task heads
        self.mu_1mb_snv = head()
        self.mu_1mb_indel = head()
        self.mu_100kb_snv = head()
        self.mu_100kb_indel = head()
        self.mu_10kb_snv = head()
        self.mu_10kb_indel = head()

        # theta parameters (unused for evaluation, but needed for state_dict)
        if fix_theta is None:
            self.theta_1mb_snv = nn.Parameter(torch.tensor([10.0]))
            self.theta_1mb_indel = nn.Parameter(torch.tensor([10.0]))
            self.theta_100kb_snv = nn.Parameter(torch.tensor([10.0]))
            self.theta_100kb_indel = nn.Parameter(torch.tensor([10.0]))
            self.theta_10kb_snv = nn.Parameter(torch.tensor([10.0]))
            self.theta_10kb_indel = nn.Parameter(torch.tensor([10.0]))
        else:
            self.register_buffer(
                "theta_fixed", torch.tensor([fix_theta], dtype=torch.float32)
            )

        # uncertainty weights (not used in evaluation but must exist)
        self.logsigma_snv = nn.Parameter(torch.tensor(0.0))
        self.logsigma_indel = nn.Parameter(torch.tensor(0.0))

        self.last_gate_means = (None, None, None)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        e1, g1 = self.b1(x1)
        h1 = self.shared1(e1)
        e2, g2 = self.b2(x2)
        h2 = self.shared2(torch.cat([e2, h1], dim=1))
        e3, g3 = self.b3(x3)

        # Shared 10kb tower
        h3 = self.shared3(torch.cat([e3, h1, h2], dim=1))

        self.last_gate_means = (
            g1.mean(0).detach(),
            g2.mean(0).detach(),
            g3.mean(0).detach(),
        )

        def clamp(mu, task, scale):
            return torch.clamp(
                mu.squeeze(-1), 1e-6, self.mu_caps[task][scale]
            )

        mu = {
            "snv": {
                "1mb": clamp(self.mu_1mb_snv(h1), "snv", "1mb"),
                "100kb": clamp(self.mu_100kb_snv(h2), "snv", "100kb"),
                "10kb": clamp(self.mu_10kb_snv(h3), "snv", "10kb"),
            },
            "indel": {
                "1mb": clamp(self.mu_1mb_indel(h1), "indel", "1mb"),
                "100kb": clamp(
                    self.mu_100kb_indel(h2), "indel", "100kb"
                ),
                "10kb": clamp(
                    self.mu_10kb_indel(h3), "indel", "10kb"
                ),
            },
        }

        if self.fix_theta is None:
            def sp(x):
                return F.softplus(x)

            th = {
                "snv": {
                    "1mb": sp(self.theta_1mb_snv),
                    "100kb": sp(self.theta_100kb_snv),
                    "10kb": sp(self.theta_10kb_snv),
                },
                "indel": {
                    "1mb": sp(self.theta_1mb_indel),
                    "100kb": sp(self.theta_100kb_indel),
                    "10kb": sp(self.theta_10kb_indel),
                },
            }
        else:
            th = {
                "snv": {s: self.theta_fixed for s in SCALES},
                "indel": {s: self.theta_fixed for s in SCALES},
            }

        return mu, th


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True, help="Project root ($PROJ).")
    ap.add_argument("--ctype", required=True, help="Cancer type (e.g. breast).")

    ap.add_argument(
        "--ca_dir",
        default=None,
        help="Dir with tcga_atac_with_repliseq.{1mb,100kb,10kb}.tsv.gz "
             "(default: <proj>/data/tcga_ca_with_rt)",
    )
    ap.add_argument(
        "--data_dir",
        default=None,
        help="Mutation CSV dir (default: <proj>/data/new_ca_rt_mutation)",
    )
    ap.add_argument(
        "--best_dir",
        default=None,
        help="Directory with best_model.pt "
             "(default: <proj>/final_version/best_models/<ctype>_best)",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Where to write TSV outputs (default: same as --best_dir).",
    )
    ap.add_argument(
        "--z3",
        type=float,
        default=3.0,
        help="z-score threshold for underestimation (default 3.0).",
    )
    ap.add_argument(
        "--z4",
        type=float,
        default=4.0,
        help="z-score threshold for strong underestimation (default 4.0).",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Batch size for inference (default 4096).",
    )
    args = ap.parse_args()

    PROJ = Path(args.proj).resolve()
    ca_dir = Path(args.ca_dir) if args.ca_dir else PROJ / "data" / "tcga_ca_with_rt"
    data_dir = Path(args.data_dir) if args.data_dir else PROJ / "data" / "new_ca_rt_mutation"

    best_dir = (
        Path(args.best_dir)
        if args.best_dir
        else PROJ / "final_version" / "best_models" / f"{args.ctype}_best"
    )
    outdir = Path(args.outdir) if args.outdir else best_dir
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== EVAL BEST MODEL: INDEL 10kb ON ALL WINDOWS ===")
    print(f"Project : {PROJ}")
    print(f"Cancer  : {args.ctype}")
    print(f"CA dir  : {ca_dir}")
    print(f"Data dir: {data_dir}")
    print(f"Best dir: {best_dir}")
    print(f"Outdir  : {outdir}")

    # Paths
    ca_1mb = ca_dir / "tcga_atac_with_repliseq.1mb.tsv.gz"
    ca_100kb = ca_dir / "tcga_atac_with_repliseq.100kb.tsv.gz"
    ca_10kb = ca_dir / "tcga_atac_with_repliseq.10kb.tsv.gz"
    indel_10kb = data_dir / "HMF_indel_10KB.csv"

    for p in [ca_1mb, ca_100kb, ca_10kb, indel_10kb]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # Build features and INDEL 10kb targets for ALL windows
    X1, X2, X3, y_true, coords = prepare_features_and_indel10_all(
        ca_1mb,
        ca_100kb,
        ca_10kb,
        indel_10kb,
        ctype=args.ctype,
        zscore=True,
        feature_clip=8.0,
    )

    n = len(y_true)
    f1, f2, f3 = X1.shape[1], X2.shape[1], X3.shape[1]

    print(f"N windows: {n}")
    print(f"Feature dims: 1MB={f1}, 100KB={f2}, 10KB={f3}")

    # Use large caps to avoid unnecessary clipping
    mu_caps = {
        "snv": {s: 1e6 for s in SCALES},
        "indel": {s: 1e6 for s in SCALES},
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    best_model_path = best_dir / "best_model.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(f"best_model.pt not found at: {best_model_path}")

    # Load checkpoint (CPU) so we can infer architectural params reliably.
    state = torch.load(best_model_path, map_location="cpu")

    # ---- Infer checkpoint config (so eval stays compatible if you change defaults) ----
    # Feature dims expected by the checkpoint (these *must* match your eval features).
    exp_f1 = int(state["b1.gate.lin.weight"].shape[0])
    exp_f2 = int(state["b2.gate.lin.weight"].shape[0])
    exp_f3 = int(state["b3.gate.lin.weight"].shape[0])

    if (f1, f2, f3) != (exp_f1, exp_f2, exp_f3):
        raise ValueError(
            "Feature dimension mismatch between eval features and checkpoint.\n"
            f"  Eval features : f1={f1}, f2={f2}, f3={f3}\n"
            f"  Checkpoint expects: f1={exp_f1}, f2={exp_f2}, f3={exp_f3}\n\n"
            "This usually happens if the training run used feature dropping (e.g. --drop_zero_var_thresh) "
            "or a different CA/RT feature set/ordering than the eval script.\n"
            "To fix: ensure eval uses the *exact* same feature columns (and order) as training."
        )

    # Hidden dims are inferred from the shared tower weights.
    hidden_ckpt = int(state["shared1.0.weight"].shape[0])
    branch_dim_ckpt = int(state["shared1.0.weight"].shape[1])

    # Whether theta was fixed during training.
    fix_theta_ckpt = float(state["theta_fixed"].view(-1)[0].item()) if "theta_fixed" in state else None

    model = HierMulti(
        f1,
        f2,
        f3,
        branch_dim=branch_dim_ckpt,
        hidden=hidden_ckpt,
        dropout=0.3,
        loss_type="poisson",
        mu_caps=mu_caps,
        fix_theta=fix_theta_ckpt,
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    print(
        "Loaded best_model.pt "
        f"(branch_dim={branch_dim_ckpt}, hidden={hidden_ckpt}, fix_theta={fix_theta_ckpt})"
    )

    # Inference for INDEL 10kb over all windows
    batch_size = args.batch_size
    preds = np.zeros_like(y_true, dtype=np.float64)

    with torch.no_grad():
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            xb1 = torch.from_numpy(X1[start_idx:end_idx]).to(device)
            xb2 = torch.from_numpy(X2[start_idx:end_idx]).to(device)
            xb3 = torch.from_numpy(X3[start_idx:end_idx]).to(device)

            mu, _ = model(xb1, xb2, xb3)
            batch_pred = (
                mu["indel"]["10kb"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            preds[start_idx:end_idx] = batch_pred

    # Residuals & z-scores (obs - pred)
    residuals = y_true - preds
    resid_mean = float(residuals.mean())
    resid_std = float(residuals.std(ddof=0))

    print(f"Residual mean (obs - pred): {resid_mean:.4f}")
    print(f"Residual std: {resid_std:.4f}")

    if resid_std < 1e-8:
        z_scores = np.zeros_like(residuals)
        print("WARNING: residual std ~ 0, z-scores set to 0.")
    else:
        z_scores = (residuals - resid_mean) / resid_std

    # Build full dataframe
    df = coords.copy()
    df["task"] = "indel"
    df["scale"] = "10kb"
    df["obs"] = y_true
    df["pred"] = preds
    df["residual"] = residuals
    df["z_resid"] = z_scores

    # Write full table
    full_path = outdir / f"{args.ctype}_indel_10kb_pred_vs_obs_all.tsv"
    df.to_csv(full_path, sep="\t", index=False)
    print(f"Wrote full obs/pred table to: {full_path}")

    # Underestimation: obs >> pred => positive residual, high z
    df_z3 = df[df["z_resid"] > args.z3].copy()
    df_z4 = df[df["z_resid"] > args.z4].copy()

    z3_path = outdir / f"{args.ctype}_indel_10kb_underest_z>{args.z3}_all.tsv"
    z4_path = outdir / f"{args.ctype}_indel_10kb_underest_z>{args.z4}_all.tsv"

    df_z3.to_csv(z3_path, sep="\t", index=False)
    df_z4.to_csv(z4_path, sep="\t", index=False)

    print(f"Windows with underestimation z > {args.z3}: {len(df_z3)}  -> {z3_path}")
    print(f"Windows with underestimation z > {args.z4}: {len(df_z4)}  -> {z4_path}")


if __name__ == "__main__":
    main()
