#!/usr/bin/env python3
import argparse, json, os, shutil, subprocess, sys, uuid, atexit, tempfile
from pathlib import Path
import optuna
from datetime import datetime

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True, help="Project root (same $PROJ you use).")
    ap.add_argument("--ctype", required=True, help="Cancer type column (e.g., breast).")

    # NEW: where tcga_atac_with_repliseq.*.tsv.gz live
    ap.add_argument("--ca_dir", default=None,
                    help="Dir with tcga_atac_with_repliseq.{1mb,100kb,10kb}.tsv.gz "
                         "(default: <proj>/data/tcga_ca_with_rt)")

    # unchanged
    ap.add_argument("--data_dir", default=None,
                    help="Mutation CSV dir (default: <proj>/data/new_ca_rt_mutation)")
    ap.add_argument("--script_path", default=None,
                    help="Training script (default: <proj>/scripts/run_model_hier_multi.py)")

    ap.add_argument("--study", default="hier_multi_optuna", help="Optuna study name.")
    ap.add_argument("--storage", default=None, help="Optuna storage URL, e.g., sqlite:////path/db.sqlite3")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=None, help="Stop optimization after N seconds.")
    ap.add_argument("--gpu_queue_note", default="", help="Optional note to include in outdir naming.")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--base_out", default=None, help="Base directory for trials (default: <proj>/optuna_trials)")
    ap.add_argument("--loss", default="poisson", choices=["poisson","mse","nb"])
    ap.add_argument("--primary_task", default="snv", choices=["snv","indel","both_mean"])
    ap.add_argument("--primary_scale", default="all_mean", choices=["10kb","100kb","1mb","all_mean"])
    return ap.parse_args()

# ---------- GZip helpers (pigz preferred) ----------
def _which(cmd):
    return shutil.which(cmd)

def _pick_gzip_tools():
    if _which("pigz"):
        return ("pigz", ["pigz","-dc"], ["pigz","-t"])
    if _which("gzip"):
        return ("gzip", ["gzip","-dc"], ["gzip","-t"])
    raise RuntimeError("Neither pigz nor gzip found in PATH.")

def _gz_test(test_cmd, path):
    try:
        subprocess.check_call(test_cmd + [str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gzip test failed for: {path}") from e

def _gz_cat(cat_cmd, gz_path, out_path):
    with open(out_path, "wb") as fh:
        subprocess.check_call(cat_cmd + [str(gz_path)], stdout=fh)

def prepare_ca_inputs(ca_dir: Path, tmp_root: Path):
    """
    Decompress:
      tcga_atac_with_repliseq.1mb.tsv.gz
      tcga_atac_with_repliseq.100kb.tsv.gz
      tcga_atac_with_repliseq.10kb.tsv.gz
    into a temporary directory and return TSV paths.
    """
    tool, cat_cmd, test_cmd = _pick_gzip_tools()
    print(f"[prep] Using {tool} for decompression")

    gz_1mb   = ca_dir / "tcga_atac_with_repliseq.1mb.tsv.gz"
    gz_100kb = ca_dir / "tcga_atac_with_repliseq.100kb.tsv.gz"
    gz_10kb  = ca_dir / "tcga_atac_with_repliseq.10kb.tsv.gz"
    for f in [gz_1mb, gz_100kb, gz_10kb]:
        if not f.exists() or f.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty: {f}")
        _gz_test(test_cmd, f)

    tmp_dir = Path(tempfile.mkdtemp(prefix="tcga_rt_", dir=str(tmp_root)))
    print(f"[prep] Decompressing CA+RT TSVs into: {tmp_dir}")

    tsv_1mb   = tmp_dir / "tcga_atac_with_repliseq.1mb.tsv"
    tsv_100kb = tmp_dir / "tcga_atac_with_repliseq.100kb.tsv"
    tsv_10kb  = tmp_dir / "tcga_atac_with_repliseq.10kb.tsv"

    _gz_cat(cat_cmd, gz_1mb,   tsv_1mb)
    _gz_cat(cat_cmd, gz_100kb, tsv_100kb)
    _gz_cat(cat_cmd, gz_10kb,  tsv_10kb)

    # Clean up on exit
    def _cleanup():
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
    atexit.register(_cleanup)

    return {"1mb": str(tsv_1mb), "100kb": str(tsv_100kb), "10kb": str(tsv_10kb)}

# ---------- Command builder ----------
def build_cmd(args, trial_outdir: Path, hp: dict, ca_tsv: dict):
    PROJ = Path(args.proj)
    DATA = Path(args.data_dir) if args.data_dir else PROJ / "data" / "new_ca_rt_mutation"
    SCRIPT = Path(args.script_path) if args.script_path else PROJ / "scripts" / "run_model_hier_multi.py"

    cmd = [
        sys.executable, str(SCRIPT),

        # NEW: pass decompressed TSVs
        "--ca_1mb",   ca_tsv["1mb"],
        "--ca_100kb", ca_tsv["100kb"],
        "--ca_10kb",  ca_tsv["10kb"],

        # Mutations (unchanged)
        "--snv_1mb",    str(DATA/"HMF_snv_1MB.csv"),
        "--snv_100kb",  str(DATA/"HMF_snv_100KB.csv"),
        "--snv_10kb",   str(DATA/"HMF_snv_10KB.csv"),
        "--indel_1mb",   str(DATA/"HMF_indel_1MB.csv"),
        "--indel_100kb", str(DATA/"HMF_indel_100KB.csv"),
        "--indel_10kb",  str(DATA/"HMF_indel_10KB.csv"),

        "--ctype", args.ctype,
        "--loss", args.loss,
        "--primary_task", args.primary_task,
        "--primary_scale", args.primary_scale,
        "--epochs", str(args.epochs),
        "--batch_size", str(hp["batch_size"]),
        "--lr", str(hp["lr"]),
        "--dropout", str(hp["dropout"]),
        "--w1mb", str(hp["w1mb"]),
        "--w100kb", str(hp["w100kb"]),
        "--w10kb", str(hp["w10kb"]),
        "--patience", str(hp["patience"]),
        "--outdir", str(trial_outdir),
        "--seed", str(hp["seed"]),
    ]

    if hp["zscore"]:
        cmd.append("--zscore")
    if hp["feature_clip"] is not None:
        cmd += ["--feature_clip", str(hp["feature_clip"])]
    if hp["drop_zero_var_thresh"] is not None:
        cmd += ["--drop_zero_var_thresh", str(hp["drop_zero_var_thresh"])]

    return cmd

def objective_builder(args, base_out: Path, ca_tsv: dict):
    logs_root = base_out / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial):
        # ---- Search space ----
        hp = {
            "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.6),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 384, 512]),
            "w1mb": trial.suggest_float("w1mb", 0.5, 2.0),
            "w100kb": trial.suggest_float("w100kb", 0.5, 2.0),
            "w10kb": trial.suggest_float("w10kb", 0.5, 2.5),
            "patience": trial.suggest_int("patience", 15, 45, step=5),
            "zscore": trial.suggest_categorical("zscore", [True, False]),
            "feature_clip": trial.suggest_categorical("feature_clip", [None, 6.0, 8.0, 10.0]),
            "drop_zero_var_thresh": trial.suggest_categorical("drop_zero_var_thresh", [None, 1e-5, 3e-5, 1e-4]),
            "seed": trial.suggest_int("seed", 1, 9999),
        }

        # ---- Trial outdir ----
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_id = f"{stamp}_{uuid.uuid4().hex[:8]}_t{trial.number}"
        trial_out = base_out / f"{args.ctype}_{trial_id}"
        trial_out.mkdir(parents=True, exist_ok=True)

        # ---- Run training script ----
        cmd = build_cmd(args, trial_out, hp, ca_tsv)
        log_path = trial_out / "stdout.txt"
        with open(log_path, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            return -1e9

        # ---- Read best score ----
        best_json = trial_out / "best_epoch.json"
        if not best_json.exists():
            return -1e9
        with open(best_json, "r") as fh:
            best = json.load(fh)
        score = float(best.get("primary_score", -1e9))

        # ---- annotate trial
        trial.set_user_attr("outdir", str(trial_out))
        trial.set_user_attr("logs", str(log_path))
        trial.set_user_attr("best_epoch", best.get("epoch"))
        trial.set_user_attr("logsigma_snv", best.get("logsigma_snv"))
        trial.set_user_attr("logsigma_indel", best.get("logsigma_indel"))
        return score

    return objective

def main():
    args = parse_args()
    PROJ = Path(args.proj)

    base_out = Path(args.base_out) if args.base_out else PROJ / "optuna_trials"
    base_out = base_out / f"{args.ctype}_{args.gpu_queue_note}".strip("_")
    base_out.mkdir(parents=True, exist_ok=True)

    # NEW: where to find gz files
    ca_dir = Path(args.ca_dir) if args.ca_dir else PROJ / "data" / "tcga_ca_with_rt"
    if not ca_dir.exists():
        raise FileNotFoundError(f"CA dir not found: {ca_dir}")

    # Decompress once per run into a temp folder under base_out
    ca_tsv = prepare_ca_inputs(ca_dir, tmp_root=base_out)

    storage = args.storage or f"sqlite:///{(base_out/'optuna.sqlite3').resolve()}"
    study = optuna.create_study(study_name=args.study, storage=storage,
                                direction="maximize", load_if_exists=True)
    objective = objective_builder(args, base_out, ca_tsv)
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    print("\n=== OPTUNA BEST TRIAL ===")
    bt = study.best_trial
    print("Value (primary R2):", bt.value)
    print("Params:", bt.params)
    print("Attrs:", bt.user_attrs)

if __name__ == "__main__":
    main()
