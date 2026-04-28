#!/usr/bin/env python3
"""Plot selected Repli-seq profile vs mutation Spearman heatmaps.

The script generates two figure files:
1. breast: MCF-7 G1/G2/S1/S2/S3/S4 vs breast SNV and Indel counts
2. lung: NHEK G1/G2/S1/S2/S3/S4 vs lung SNV and Indel counts

Each figure contains one heatmap per genomic scale and reports rho plus significance.
Outputs are written under `data/` by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_CELL_SIZE = 0.40
TEXT_COLOR = "#000000"
COLORBAR_LABEL = "Spearman rho"
LEGEND_WIDTH = 0.50
LEGEND_HEIGHT = 3.20

SCALE_ORDER = ["1mb", "100kb", "10kb"]
SCALE_CONFIG = {
    "10kb": {
        "label": "10 kb",
        "snv": DATA_DIR / "HMF_snv_10KB.csv",
        "indel": DATA_DIR / "HMF_indel_10KB.csv",
        "features": DATA_DIR / "tcga_atac_with_repliseq.10kb.tsv.gz",
    },
    "100kb": {
        "label": "100 kb",
        "snv": DATA_DIR / "HMF_snv_100KB.csv",
        "indel": DATA_DIR / "HMF_indel_100KB.csv",
        "features": DATA_DIR / "tcga_atac_with_repliseq.100kb.tsv.gz",
    },
    "1mb": {
        "label": "1 Mb",
        "snv": DATA_DIR / "HMF_snv_1MB.csv",
        "indel": DATA_DIR / "HMF_indel_1MB.csv",
        "features": DATA_DIR / "tcga_atac_with_repliseq.1mb.tsv.gz",
    },
}

MUTATION_ORDER = ["SNV", "Indel"]
PLOT_CONFIG = {
    "breast": {
        "title": "Breast",
        "mutation_column": "breast",
        "profiles": [
            ("MCF7_G1", "MCF-7 G1"),
            ("MCF7_S1", "MCF-7 S1"),
            ("MCF7_S2", "MCF-7 S2"),
            ("MCF7_S3", "MCF-7 S3"),
            ("MCF7_S4", "MCF-7 S4"),
            ("MCF7_G2", "MCF-7 G2"),
        ],
    },
    "lung": {
        "title": "Lung",
        "mutation_column": "lung",
        "profiles": [
            ("NHEK_G1", "NHEK G1"),
            ("NHEK_S1", "NHEK S1"),
            ("NHEK_S2", "NHEK S2"),
            ("NHEK_S3", "NHEK S3"),
            ("NHEK_S4", "NHEK S4"),
            ("NHEK_G2", "NHEK G2"),
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot selected Repli-seq profile vs mutation Spearman correlations."
    )
    parser.add_argument(
        "--out-dir",
        default=str(DATA_DIR),
        help="Directory for figure files.",
    )
    parser.add_argument(
        "--tsv-prefix",
        default=str(DATA_DIR / "selected_repliseq_profile_mutation_spearman"),
        help="Output prefix for TSV tables.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=10,
        help="Base font size for the heatmap figures.",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=DEFAULT_CELL_SIZE,
        help="Approximate heatmap cell size in inches; lower values make boxes smaller.",
    )
    return parser.parse_args()


def load_feature_table(path: Path, profile_columns: list[str]) -> pd.DataFrame:
    usecols = ["chr", "start", "end", *profile_columns]
    return pd.read_csv(path, sep="\t", usecols=usecols)


def load_mutation_table(path: Path, cancer: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["chr", "start", "end", cancer])
    return df.rename(columns={cancer: f"{prefix}_{cancer}"})


def spearman_stats(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3:
        return float("nan"), float("nan"), int(len(pair))
    if pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return float("nan"), float("nan"), int(len(pair))
    result = spearmanr(pair["x"], pair["y"])
    return float(result.correlation), float(result.pvalue), int(len(pair))


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    out = pd.Series(float("nan"), index=pvalues.index, dtype=float)
    valid = pvalues.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return out

    prev = 1.0
    adjusted: list[tuple[int, float]] = []
    for rank in range(m, 0, -1):
        idx = valid.index[rank - 1]
        pvalue = float(valid.iloc[rank - 1])
        qvalue = min(prev, pvalue * m / rank)
        prev = qvalue
        adjusted.append((idx, qvalue))

    for idx, qvalue in adjusted:
        out.loc[idx] = qvalue
    return out


def significance_label(qvalue: float) -> str:
    if pd.isna(qvalue):
        return "NA"
    if qvalue <= 0.001:
        return "***"
    if qvalue <= 0.01:
        return "**"
    if qvalue <= 0.05:
        return "*"
    return "ns"


def build_annotation_label(rho: float, qvalue: float) -> str:
    if pd.isna(rho):
        return "NA"
    return f"{rho:.2f}\n{significance_label(qvalue)}"


def build_scale_table(cancer_key: str, scale_key: str) -> pd.DataFrame:
    scale_cfg = SCALE_CONFIG[scale_key]
    cancer_cfg = PLOT_CONFIG[cancer_key]
    cancer_col = cancer_cfg["mutation_column"]
    profiles = cancer_cfg["profiles"]
    profile_ids = [profile_id for profile_id, _ in profiles]

    feature_df = load_feature_table(scale_cfg["features"], profile_ids)
    snv = load_mutation_table(scale_cfg["snv"], cancer_col, prefix="snv")
    indel = load_mutation_table(scale_cfg["indel"], cancer_col, prefix="indel")
    merged = snv.merge(indel, on=["chr", "start", "end"], how="inner")
    merged = merged.merge(feature_df, on=["chr", "start", "end"], how="inner")

    rows = []
    for profile_id, profile_label in profiles:
        for mutation_label, mutation_col in [
            ("SNV", f"snv_{cancer_col}"),
            ("Indel", f"indel_{cancer_col}"),
        ]:
            rho, pvalue, n_windows = spearman_stats(merged[profile_id], merged[mutation_col])
            rows.append(
                {
                    "cancer": cancer_key,
                    "cancer_label": cancer_cfg["title"],
                    "scale": scale_key,
                    "scale_label": scale_cfg["label"],
                    "profile_id": profile_id,
                    "profile_label": profile_label,
                    "mutation_type": mutation_label,
                    "spearman_rho": rho,
                    "spearman_pvalue": pvalue,
                    "n_windows": n_windows,
                }
            )

    return pd.DataFrame(rows)


def pivot_metric(long_df: pd.DataFrame, cancer_key: str, value_col: str) -> pd.DataFrame:
    return (
        long_df.pivot(index="profile_label", columns="mutation_type", values=value_col)
        .reindex(index=[label for _, label in PLOT_CONFIG[cancer_key]["profiles"]], columns=MUTATION_ORDER)
    )


def setup_style(font_size: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "axes.edgecolor": TEXT_COLOR,
            "xtick.labelsize": font_size - 1,
            "xtick.color": TEXT_COLOR,
            "ytick.labelsize": font_size - 1,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_theme(style="white")


def compute_figure_size(cell_size: float) -> tuple[float, float]:
    panel_rows = len(next(iter(PLOT_CONFIG.values()))["profiles"])
    panel_cols = len(MUTATION_ORDER)
    fig_width = max(4.8, len(SCALE_ORDER) * panel_cols * cell_size + 2.4)
    fig_height = max(4.4, panel_rows * cell_size + 1.8)
    return fig_width, fig_height


def save_color_legend(out_path: Path, cmap, font_size: float) -> None:
    fig, ax = plt.subplots(figsize=(LEGEND_WIDTH, LEGEND_HEIGHT))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    mappable = ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=cmap)
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, cax=ax, orientation="vertical")
    colorbar.set_label(COLORBAR_LABEL, color=TEXT_COLOR)
    colorbar.ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR, length=3)
    colorbar.outline.set_edgecolor(TEXT_COLOR)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_cancer_heatmaps(
    cancer_key: str,
    scale_tables: dict[str, pd.DataFrame],
    annotation_tables: dict[str, pd.DataFrame],
    out_dir: Path,
    font_size: float,
    cell_size: float,
) -> None:
    setup_style(font_size)
    out_prefix = out_dir / f"{cancer_key}_repliseq_profile_mutation_spearman_heatmap"
    legend_path = out_prefix.with_name(f"{out_prefix.stem}_legend.pdf")

    fig_width, fig_height = compute_figure_size(cell_size)
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.10, wspace=0.22)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    for idx, scale_key in enumerate(SCALE_ORDER):
        table = scale_tables[scale_key]
        ax = axes[idx]
        sns.heatmap(
            table,
            ax=ax,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            center=0,
            linewidths=0.8,
            linecolor="#FFFFFF",
            annot=annotation_tables[scale_key],
            fmt="",
            cbar=False,
            annot_kws={"fontsize": font_size - 2, "color": TEXT_COLOR},
        )
        ax.set_title(SCALE_CONFIG[scale_key]["label"], pad=4)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0, pad=2, length=0)
        ax.tick_params(axis="y", rotation=0, pad=1, length=0)
        for text in ax.texts:
            text.set_color(TEXT_COLOR)
        if idx > 0:
            ax.tick_params(axis="y", labelleft=False)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    save_color_legend(legend_path, cmap=cmap, font_size=font_size)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    tsv_prefix = Path(args.tsv_prefix)

    long_tables = []
    for cancer_key in ["breast", "lung"]:
        cancer_long = pd.concat(
            [build_scale_table(cancer_key, scale_key) for scale_key in SCALE_ORDER],
            ignore_index=True,
        )
        cancer_long["spearman_qvalue"] = benjamini_hochberg(cancer_long["spearman_pvalue"])
        cancer_long["significant_fdr_0_05"] = cancer_long["spearman_qvalue"] <= 0.05
        cancer_long["annotation_label"] = cancer_long.apply(
            lambda row: build_annotation_label(row["spearman_rho"], row["spearman_qvalue"]),
            axis=1,
        )
        long_tables.append(cancer_long)

        scale_tables: dict[str, pd.DataFrame] = {}
        annotation_tables: dict[str, pd.DataFrame] = {}
        for scale_key in SCALE_ORDER:
            scale_long = cancer_long.loc[cancer_long["scale"] == scale_key].copy()
            rho_table = pivot_metric(scale_long, cancer_key, "spearman_rho")
            pvalue_table = pivot_metric(scale_long, cancer_key, "spearman_pvalue")
            qvalue_table = pivot_metric(scale_long, cancer_key, "spearman_qvalue")
            sig_table = pivot_metric(scale_long, cancer_key, "significant_fdr_0_05")
            annotation_table = pivot_metric(scale_long, cancer_key, "annotation_label")

            scale_tables[scale_key] = rho_table
            annotation_tables[scale_key] = annotation_table
            tsv_prefix.parent.mkdir(parents=True, exist_ok=True)
            rho_table.to_csv(
                tsv_prefix.with_name(f"{tsv_prefix.name}.{cancer_key}.{scale_key}.tsv"),
                sep="\t",
                na_rep="NA",
            )
            pvalue_table.to_csv(
                tsv_prefix.with_name(f"{tsv_prefix.name}.{cancer_key}.{scale_key}.pvalue.tsv"),
                sep="\t",
                na_rep="NA",
            )
            qvalue_table.to_csv(
                tsv_prefix.with_name(f"{tsv_prefix.name}.{cancer_key}.{scale_key}.qvalue.tsv"),
                sep="\t",
                na_rep="NA",
            )
            sig_table.to_csv(
                tsv_prefix.with_name(f"{tsv_prefix.name}.{cancer_key}.{scale_key}.significant.tsv"),
                sep="\t",
                na_rep="NA",
            )
        plot_cancer_heatmaps(
            cancer_key,
            scale_tables,
            annotation_tables,
            out_dir=out_dir,
            font_size=args.font_size,
            cell_size=args.cell_size,
        )

    all_long = pd.concat(long_tables, ignore_index=True)
    tsv_prefix.parent.mkdir(parents=True, exist_ok=True)
    all_long.to_csv(tsv_prefix.with_name(f"{tsv_prefix.name}.long.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
