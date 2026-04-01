#!/usr/bin/env python3
"""
Plot observed vs predicted 10kb windows for selected genes.

Uses underest_v2 outlier outputs:
- combined_10kb_underest_z4_genes_max_z.tsv (gene list + max cancer/variant), or
- a direct joined table such as combined_log2resid_z4_step4_with_cancer_flags.tsv
- hg19_genes_gff.bed (gene coordinates)
- <cancer>/<cancer>_<variant>_10kb_pred_vs_obs_all.tsv (window counts)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())


def find_pred_obs_file(input_dir: Path, cancer: str, variant: str) -> Path:
    candidates = [
        input_dir / f"{cancer}_{variant}_10kb_pred_vs_obs_all.tsv",
        input_dir / cancer / f"{cancer}_{variant}_10kb_pred_vs_obs_all.tsv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    pattern = f"**/{cancer}_{variant}_10kb_pred_vs_obs_all.tsv"
    matches = sorted(input_dir.glob(pattern))
    if matches:
        return matches[0]

    tested = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        f"Missing pred_vs_obs file for {cancer}/{variant}.\n"
        f"Tried:\n{tested}\n"
        f"And recursive pattern: {pattern}"
    )


def pick_gene_row(gene_df: pd.DataFrame, gene: str) -> pd.Series:
    rows = gene_df[gene_df["gene"] == gene].copy()
    if rows.empty:
        raise KeyError(f"Gene '{gene}' not found in hg19_genes_gff.bed")
    rows["len"] = rows["end"] - rows["start"]
    rows = rows.sort_values(["len"], ascending=[False])
    return rows.iloc[0]


def build_selection_from_table(input_table: Path, z_col: str, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(input_table, sep="\t")
    required = {"gene", "cancer_type", "variant", z_col}
    missing = required.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"{input_table} missing required columns: {missing_cols}")

    base = df[["gene", "cancer_type", "variant", z_col]].dropna(subset=["gene"]).copy()
    base = base.rename(columns={z_col: "z_resid"})

    gene_ct_max = base.groupby(["gene", "cancer_type"], as_index=False)["z_resid"].max()
    gene_totals = (
        gene_ct_max.groupby("gene", as_index=False)["z_resid"]
        .sum()
        .sort_values("z_resid", ascending=False)
        .head(top_n)
        .rename(columns={"z_resid": "stacked_z_resid"})
    )

    best_rows = (
        base.sort_values(["gene", "z_resid"], ascending=[True, False])
        .drop_duplicates(subset=["gene"], keep="first")
        .rename(columns={"cancer_type": "max_cancer_type"})
    )

    selected = gene_totals.merge(best_rows, on="gene", how="left")
    selected = selected.sort_values("stacked_z_resid", ascending=False)
    return selected


def plot_one_gene(
    gene: str,
    cancer: str,
    variant: str,
    z_value: float,
    gene_df: pd.DataFrame,
    input_dir: Path,
    output_dir: Path,
    flank: int,
) -> Path:
    gene_row = pick_gene_row(gene_df, gene)
    gene_chr = str(gene_row["chr"])
    gene_start = int(gene_row["start"])
    gene_end = int(gene_row["end"])

    pred_obs_path = find_pred_obs_file(input_dir, cancer, variant)
    df = pd.read_csv(pred_obs_path, sep="\t")
    required = {"chr", "start", "obs", "pred"}
    missing = required.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"{pred_obs_path} missing required columns: {missing_cols}")

    mask = (df["chr"] == gene_chr) & (df["start"].between(gene_start - flank, gene_end + flank))
    sub = df.loc[mask, ["chr", "start", "obs", "pred"]].copy()
    if sub.empty:
        raise ValueError(
            f"No windows found around {gene} ({gene_chr}:{gene_start}-{gene_end}) in {pred_obs_path}"
        )
    sub["center"] = sub["start"] + 5000

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axvspan(gene_start, gene_end, color="#fde0dd", alpha=0.6, label=f"{gene} locus")

    bar_width = 9000
    ax.bar(sub["center"], sub["obs"], width=bar_width, color="#4C72B0", alpha=0.7, label="Observed", align="center")
    ax.bar(sub["center"], sub["pred"], width=bar_width, color="#DD8452", alpha=0.7, label="Predicted", align="center")

    ax.set_xlabel(f"Genomic position ({gene_chr})")
    ax.set_ylabel(f"{variant.upper()} counts (10kb windows)")
    ax.legend()
    ax.set_xticks([])

    out_name = f"{safe_name(gene)}_{safe_name(cancer)}_{safe_name(variant)}_obs_pred_zgt10.png"
    out_path = output_dir / out_name
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "outlier_regions"
    default_output = default_input / "plots" / "specific_gene_windows_zgt10"

    parser = argparse.ArgumentParser(
        description="Plot gene windows for genes with z_resid > threshold from underest_v2 outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help="Directory containing outlier_regions files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory for generated plots.",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=10.0,
        help="Only genes with z_resid strictly greater than this threshold are plotted.",
    )
    parser.add_argument(
        "--flank",
        type=int,
        default=250_000,
        help="Flanking bases on each side of gene locus for window plotting.",
    )
    parser.add_argument(
        "--input-table",
        type=Path,
        default=None,
        help=(
            "Optional direct input table with gene/cancer_type/variant/z columns, "
            "for example combined_log2resid_z4_step4_with_cancer_flags.tsv."
        ),
    )
    parser.add_argument(
        "--z-col",
        type=str,
        default="z_log_resid",
        help="Z-score column to use with --input-table.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top genes to plot when using --input-table.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default=None,
        help="Optional manifest filename. Defaults depend on input mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 25,
            "axes.titlesize": 25,
            "axes.labelsize": 25,
            "xtick.labelsize": 25,
            "ytick.labelsize": 25,
            "legend.fontsize": 25,
        }
    )

    if args.input_table is not None:
        input_table = args.input_table.resolve()
        selected = build_selection_from_table(input_table, z_col=args.z_col, top_n=args.top_n)
        if selected.empty:
            raise SystemExit(f"No genes found in {input_table}.")
        manifest_name = args.manifest_name or f"{input_table.stem}_gene_window_plots_manifest.tsv"
    else:
        max_z_path = input_dir / "combined_10kb_underest_z4_genes_max_z.tsv"
        if not max_z_path.exists():
            raise FileNotFoundError(f"Missing genes max-z file: {max_z_path}")
        genes_max = pd.read_csv(max_z_path, sep="\t")
        required = {"gene", "max_cancer_type", "variant", "z_resid"}
        missing = required.difference(genes_max.columns)
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"{max_z_path} missing required columns: {missing_cols}")

        selected = genes_max[genes_max["z_resid"] > args.z_threshold].sort_values(
            "z_resid", ascending=False
        )
        if selected.empty:
            raise SystemExit(f"No genes found with z_resid > {args.z_threshold}.")
        manifest_name = args.manifest_name or "zgt10_gene_window_plots_manifest.tsv"

    bed_path = input_dir / "hg19_genes_gff.bed"
    if not bed_path.exists():
        raise FileNotFoundError(f"Missing gene bed file: {bed_path}")
    gene_df = pd.read_csv(bed_path, sep="\t")
    bed_required = {"chr", "start", "end", "gene"}
    bed_missing = bed_required.difference(gene_df.columns)
    if bed_missing:
        missing_cols = ", ".join(sorted(bed_missing))
        raise ValueError(f"{bed_path} missing required columns: {missing_cols}")

    manifest_rows: list[dict[str, str]] = []
    failures: list[tuple[str, str]] = []
    for _, row in selected.iterrows():
        gene = str(row["gene"])
        cancer = str(row["max_cancer_type"])
        variant = str(row["variant"])
        z_val = float(row["z_resid"])
        try:
            out_path = plot_one_gene(
                gene=gene,
                cancer=cancer,
                variant=variant,
                z_value=z_val,
                gene_df=gene_df,
                input_dir=input_dir,
                output_dir=output_dir,
                flank=args.flank,
            )
            manifest_rows.append(
                {
                    "gene": gene,
                    "max_cancer_type": cancer,
                    "variant": variant,
                    "z_resid": f"{z_val:.6f}",
                    "plot_file": out_path.name,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append((gene, str(exc)))

    manifest = pd.DataFrame(manifest_rows)
    if "stacked_z_resid" in selected.columns and not manifest.empty:
        manifest = manifest.merge(
            selected[["gene", "stacked_z_resid"]].drop_duplicates(),
            on="gene",
            how="left",
        )

    manifest_path = output_dir / manifest_name
    manifest.to_csv(manifest_path, sep="\t", index=False)

    print(f"Input dir: {input_dir}")
    if args.input_table is None:
        print(f"Threshold: z_resid > {args.z_threshold}")
    else:
        print(f"Input table: {input_table}")
        print(f"Top genes requested: {args.top_n}")
    print(f"Requested genes: {len(selected)}")
    print(f"Generated plots: {len(manifest_rows)}")
    print(f"Manifest: {manifest_path}")
    if failures:
        print(f"Failed genes: {len(failures)}")
        for gene, msg in failures:
            print(f"- {gene}: {msg}")
    else:
        print("Failed genes: 0")


if __name__ == "__main__":
    main()
