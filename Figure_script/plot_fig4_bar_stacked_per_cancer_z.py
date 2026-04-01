#!/usr/bin/env python3
"""
Generate two stacked bar plots from underest_v2 outlier regions:
1) Top N cancer genes by stacked per-cancer max z-score
2) Top N genes (all) by stacked per-cancer max z-score, with cancer genes in red
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CANCER_ORDER = ["breast", "colorectal", "skin", "lung", "prostate", "esophagus"]
VARIANT_ORDER = ["snv", "indel"]
PALETTE = {
    "breast": "#E75480",
    "colorectal": "#1F78B4",
    "skin": "#F2CC1F",
    "lung": "#33A02C",
    "prostate": "#6A3D9A",
    "esophagus": "#FF8C00",
}


def format_window_label(chrom: str, start: int, end: int) -> str:
    chrom_str = str(chrom)
    if chrom_str.startswith("chr"):
        chrom_str = chrom_str[3:]
    return f"{chrom_str}:{start / 1e6:.2f}-{end / 1e6:.2f}M"


def find_underest_file(input_dir: Path, cancer: str, variant: str) -> Path:
    candidates = [
        input_dir / f"{cancer}_{variant}_10kb_underest_z_4.0_all.tsv",
        input_dir / f"{cancer}_{variant}_10kb_underest_z>4.0_all.tsv",
        input_dir / cancer / f"{cancer}_{variant}_10kb_underest_z_4.0_all.tsv",
        input_dir / cancer / f"{cancer}_{variant}_10kb_underest_z>4.0_all.tsv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    pattern = f"**/{cancer}_{variant}_10kb_underest_z*4.0_all.tsv"
    matches = sorted(input_dir.glob(pattern))
    if matches:
        return matches[0]

    tested = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        f"Missing per-cancer underestimated file for {cancer}/{variant}.\n"
        f"Tried canonical paths:\n{tested}\n"
        f"And recursive pattern: {pattern}"
    )


def load_per_cancer_z(input_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for cancer in CANCER_ORDER:
        for variant in VARIANT_ORDER:
            path = find_underest_file(input_dir, cancer, variant)
            df = pd.read_csv(path, sep="\t")
            required = {"chr", "start", "z_resid"}
            missing = required.difference(df.columns)
            if missing:
                missing_cols = ", ".join(sorted(missing))
                raise ValueError(f"{path} missing required columns: {missing_cols}")
            part = df[["chr", "start", "z_resid"]].copy()
            part["cancer_type"] = cancer
            frames.append(part)
    return pd.concat(frames, ignore_index=True)


def load_mapping(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "combined_10kb_underest_z4_step3_with_genes.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing step3 mapping file: {path}")
    mapping = pd.read_csv(path, sep="\t")
    required = {"chr", "start", "gene"}
    missing = required.difference(mapping.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"{path} missing required columns: {missing_cols}")
    return mapping[["chr", "start", "gene"]].drop_duplicates()


def load_cancer_gene_set(input_dir: Path) -> set[str]:
    candidates = [
        input_dir / "combined_10kb_underest_z4_step6_unique_cancer_genes.tsv",
        input_dir / "combined_10kb_underest_z4_cancer_types_by_gene.tsv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, sep="\t")
            if "gene" not in df.columns:
                raise ValueError(f"{path} must contain 'gene' column.")
            return set(df["gene"].dropna().astype(str))

    step5 = input_dir / "combined_10kb_underest_z4_step5_cancer_genes_only.tsv"
    if step5.exists():
        df = pd.read_csv(step5, sep="\t")
        if "gene" not in df.columns:
            raise ValueError(f"{step5} must contain 'gene' column.")
        return set(df["gene"].dropna().astype(str))

    tried = "\n".join(f"- {p}" for p in [*candidates, step5])
    raise FileNotFoundError(f"Could not find cancer gene list. Tried:\n{tried}")


def infer_companion_window_table(input_table: Path) -> Path | None:
    name = input_table.name
    if "_step4_with_cancer_flags.tsv" not in name:
        return None
    candidate = input_table.with_name(name.replace("_step4_with_cancer_flags.tsv", "_step1.tsv"))
    if candidate.exists():
        return candidate
    return None


def load_direct_table(
    input_table: Path,
    z_col: str,
    include_no_gene_windows: bool = False,
    window_table: Path | None = None,
) -> tuple[pd.DataFrame, set[str], int]:
    df = pd.read_csv(input_table, sep="\t")
    required = {"gene", "cancer_type", z_col}
    missing = required.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"{input_table} missing required columns: {missing_cols}")

    merged_all = df[["gene", "cancer_type", z_col]].dropna(subset=["gene"]).copy()
    merged_all = merged_all.rename(columns={z_col: "z_resid"})

    cancer_gene_set: set[str] = set()
    flag_cols = [col for col in ["in_oncokb", "in_cgc"] if col in df.columns]
    if flag_cols:
        cancer_mask = pd.Series(False, index=df.index)
        for col in flag_cols:
            cancer_mask |= df[col].astype(str).str.strip().str.lower().eq("yes")
        cancer_gene_set = set(df.loc[cancer_mask, "gene"].dropna().astype(str))

    no_gene_count = 0
    if include_no_gene_windows:
        keys = ["chr", "start", "cancer_type", "variant"]
        key_required = set(keys)
        key_missing = key_required.difference(df.columns)
        if key_missing:
            missing_cols = ", ".join(sorted(key_missing))
            raise ValueError(
                f"{input_table} missing required columns for no-gene window inclusion: {missing_cols}"
            )

        resolved_window_table = window_table or infer_companion_window_table(input_table)
        if resolved_window_table is None or not resolved_window_table.exists():
            raise FileNotFoundError(
                "Could not determine companion window table for no-gene windows. "
                "Pass --window-table explicitly."
            )

        window_df = pd.read_csv(resolved_window_table, sep="\t")
        window_required = set(keys) | {z_col}
        window_missing = window_required.difference(window_df.columns)
        if window_missing:
            missing_cols = ", ".join(sorted(window_missing))
            raise ValueError(f"{resolved_window_table} missing required columns: {missing_cols}")

        window_keys = df[keys].drop_duplicates().assign(has_gene=True)
        no_gene_df = window_df.merge(window_keys, on=keys, how="left")
        no_gene_df = no_gene_df[no_gene_df["has_gene"].isna()].copy()
        if "end" not in no_gene_df.columns:
            no_gene_df["end"] = no_gene_df["start"] + 9999
        no_gene_df["gene"] = no_gene_df.apply(
            lambda row: format_window_label(
                chrom=row["chr"],
                start=int(row["start"]),
                end=int(row["end"]),
            ),
            axis=1,
        )
        no_gene_df = no_gene_df[["gene", "cancer_type", z_col]].rename(columns={z_col: "z_resid"})
        no_gene_count = len(no_gene_df)
        merged_all = pd.concat([merged_all, no_gene_df], ignore_index=True)

    return merged_all, cancer_gene_set, no_gene_count


def build_pivot(gene_ct_max: pd.DataFrame, top_n: int) -> pd.DataFrame:
    gene_totals = gene_ct_max.groupby("gene")["z_resid"].sum().sort_values(ascending=False)
    top_genes = gene_totals.index if top_n <= 0 else gene_totals.head(top_n).index
    pivot = gene_ct_max[gene_ct_max["gene"].isin(top_genes)].pivot_table(
        index="gene", columns="cancer_type", values="z_resid", fill_value=0
    )
    gene_order = gene_totals.loc[top_genes].index
    cancer_order = pivot.sum(axis=0).sort_values(ascending=False).index
    return pivot.reindex(index=gene_order, columns=cancer_order, fill_value=0)


def draw_stacked_barh(
    pivot: pd.DataFrame,
    title: str,
    output_png: Path,
    output_pdf: Path,
    cancer_gene_set: set[str] | None = None,
    y_label: str = "Gene",
) -> None:
    fig_height = max(10, len(pivot.index) * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    bottom = None
    for cancer_type in pivot.columns:
        values = pivot[cancer_type]
        color = PALETTE.get(cancer_type, "#B0B0B0")
        if bottom is None:
            ax.barh(pivot.index, values, color=color, label=cancer_type)
            bottom = values.copy()
        else:
            ax.barh(pivot.index, values, left=bottom, color=color, label=cancer_type)
            bottom = bottom + values

    ax.invert_yaxis()
    ax.set_xlabel("z-score (stacked)")
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    if cancer_gene_set is not None:
        for label in ax.get_yticklabels():
            if label.get_text() in cancer_gene_set:
                label.set_color("red")
    fig.tight_layout()
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "outlier_regions"
    default_output = default_input / "plots"

    parser = argparse.ArgumentParser(
        description="Plot top genes by stacked per-cancer z-score for underest_v2."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help="Directory containing outlier_regions step files and per-cancer inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory to write output plots.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of genes/windows to plot in each figure. Use 0 or a negative value to plot all.",
    )
    parser.add_argument(
        "--input-table",
        type=Path,
        default=None,
        help=(
            "Optional pre-joined table with gene, cancer_type, and z-score column "
            "(for example combined_log2resid_z4_step4_with_cancer_flags.tsv)."
        ),
    )
    parser.add_argument(
        "--z-col",
        type=str,
        default="z_log_resid",
        help="Z-score column to use with --input-table.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional filename prefix for outputs when using custom input.",
    )
    parser.add_argument(
        "--include-no-gene-windows",
        action="store_true",
        help="When using --input-table, include outlier windows from a companion step1 table that lack gene overlaps.",
    )
    parser.add_argument(
        "--window-table",
        type=Path,
        default=None,
        help="Optional explicit step1 window table to use with --include-no-gene-windows.",
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
            "font.size": 21,
            "axes.titlesize": 21,
            "axes.labelsize": 21,
            "xtick.labelsize": 21,
            "ytick.labelsize": 21,
            "legend.fontsize": 21,
        }
    )

    if args.input_table is not None:
        input_table = args.input_table.resolve()
        resolved_window_table = args.window_table.resolve() if args.window_table else None
        merged_all, cancer_gene_set, no_gene_count = load_direct_table(
            input_table,
            z_col=args.z_col,
            include_no_gene_windows=args.include_no_gene_windows,
            window_table=resolved_window_table,
        )
        output_prefix = args.output_prefix or input_table.stem
    else:
        z_df = load_per_cancer_z(input_dir)
        mapping = load_mapping(input_dir)
        cancer_gene_set = load_cancer_gene_set(input_dir)
        merged_all = pd.merge(z_df, mapping, on=["chr", "start"], how="left").dropna(subset=["gene"])
        no_gene_count = 0
        output_prefix = args.output_prefix

    gene_ct_max_all = merged_all.groupby(["gene", "cancer_type"], as_index=False)["z_resid"].max()
    pivot_all = build_pivot(gene_ct_max_all, top_n=args.top_n)
    rank_label = f"top{args.top_n}" if args.top_n > 0 else "all"
    all_base = (
        f"{output_prefix}_stacked_{rank_label}_per_cancer_z_all_genes"
        if output_prefix
        else f"stacked_{rank_label}_per_cancer_z_all_genes"
    )
    all_png = output_dir / f"{all_base}.png"
    all_pdf = output_dir / f"{all_base}.pdf"
    draw_stacked_barh(
        pivot=pivot_all,
        title="",
        output_png=all_png,
        output_pdf=all_pdf,
        cancer_gene_set=cancer_gene_set,
        y_label="Gene/Window",
    )

    merged_cancer = merged_all[merged_all["gene"].isin(cancer_gene_set)]
    gene_ct_max_cancer = merged_cancer.groupby(["gene", "cancer_type"], as_index=False)["z_resid"].max()
    pivot_cancer = build_pivot(gene_ct_max_cancer, top_n=args.top_n)
    cancer_base = (
        f"{output_prefix}_stacked_{rank_label}_per_cancer_z"
        if output_prefix
        else f"stacked_{rank_label}_per_cancer_z"
    )
    cancer_png = output_dir / f"{cancer_base}.png"
    cancer_pdf = output_dir / f"{cancer_base}.pdf"
    draw_stacked_barh(
        pivot=pivot_cancer,
        title="",
        output_png=cancer_png,
        output_pdf=cancer_pdf,
        cancer_gene_set=None,
        y_label="Gene",
    )

    if args.input_table is not None:
        print(f"Input table: {input_table}")
        if args.include_no_gene_windows:
            print(f"Included no-gene windows: {no_gene_count}")
    else:
        print(f"Input dir: {input_dir}")
    print(f"Wrote {all_png} and {all_pdf}")
    print(f"Wrote {cancer_png} and {cancer_pdf}")


if __name__ == "__main__":
    main()
