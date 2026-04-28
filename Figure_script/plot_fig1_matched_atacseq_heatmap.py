#!/usr/bin/env python3
"""Plot matched ATAC vs mutation Spearman correlations as heatmaps.

For each of the six cancer types, this script:
1. averages matched ATAC-seq columns per genomic window,
2. computes Spearman rho plus p/q values against SNV and INDEL mutation counts,
3. draws one vertically stacked heatmap panel per genomic scale.

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

CANCER_ORDER = ["breast", "colorectal", "esophagus", "lung", "prostate", "skin"]
CANCER_TITLES = {
    "breast": "Breast",
    "colorectal": "Colorectal",
    "esophagus": "Esophagus",
    "lung": "Lung",
    "prostate": "Prostate",
    "skin": "Skin",
}

CANCER_META = {
    "breast": {
        "atac_prefixes": ["BRCA_"],
    },
    "colorectal": {
        "atac_prefixes": ["COAD_", "READ_"],
    },
    "esophagus": {
        "atac_prefixes": ["ESCA_"],
    },
    "lung": {
        "atac_prefixes": ["LUAD_", "LUSC_"],
    },
    "prostate": {
        "atac_prefixes": ["PRAD_"],
    },
    "skin": {
        "atac_prefixes": ["SKCM_"],
    },
}

SCALE_CONFIG = {
    "10kb": {
        "label": "10 kb",
        "snv": DATA_DIR / "HMF_snv_10KB.csv",
        "indel": DATA_DIR / "HMF_indel_10KB.csv",
        "atac": DATA_DIR / "tcga_atac_with_repliseq.10kb.tsv.gz",
    },
    "100kb": {
        "label": "100 kb",
        "snv": DATA_DIR / "HMF_snv_100KB.csv",
        "indel": DATA_DIR / "HMF_indel_100KB.csv",
        "atac": DATA_DIR / "tcga_atac_with_repliseq.100kb.tsv.gz",
    },
    "1mb": {
        "label": "1 Mb",
        "snv": DATA_DIR / "HMF_snv_1MB.csv",
        "indel": DATA_DIR / "HMF_indel_1MB.csv",
        "atac": DATA_DIR / "tcga_atac_with_repliseq.1mb.tsv.gz",
    },
}
SCALE_ORDER = ["1mb", "100kb", "10kb"]

ROW_ORDER = ["SNV", "Indel"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot matched ATAC vs mutation Spearman correlations."
    )
    parser.add_argument(
        "--out-prefix",
        default=str(DATA_DIR / "matched_epigenome_mutation_spearman_heatmap"),
        help="Output prefix for the figure files.",
    )
    parser.add_argument(
        "--tsv-prefix",
        default=str(DATA_DIR / "matched_epigenome_mutation_spearman"),
        help="Output prefix for TSV tables.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=10,
        help="Base font size for the heatmap figure.",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=DEFAULT_CELL_SIZE,
        help="Approximate heatmap cell size in inches; lower values make boxes smaller.",
    )
    return parser.parse_args()


def read_header(path: Path, sep: str) -> list[str]:
    return pd.read_csv(path, sep=sep, nrows=0).columns.tolist()


def select_atac_columns(columns: list[str]) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for cancer in CANCER_ORDER:
        prefixes = tuple(CANCER_META[cancer]["atac_prefixes"])
        matched = sorted(col for col in columns if col.startswith(prefixes))
        if not matched:
            raise ValueError(f"No matched ATAC columns found for {cancer}")
        selected[cancer] = matched
    return selected


def load_window_means(
    path: Path,
    sep: str,
    column_groups: dict[str, list[str]],
    value_prefix: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    usecols = ["chr", "start", "end"]
    for columns in column_groups.values():
        usecols.extend(columns)
    usecols = list(dict.fromkeys(usecols))

    df = pd.read_csv(path, sep=sep, usecols=usecols)
    out = df[["chr", "start", "end"]].copy()
    counts = {}
    for cancer, columns in column_groups.items():
        out[f"{value_prefix}_{cancer}"] = df[columns].mean(axis=1)
        counts[cancer] = len(columns)
    return out, counts


def load_mutation_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = ["chr", "start", "end", *CANCER_ORDER]
    missing = [col for col in keep if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[keep].copy()


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


def build_scale_table(scale_key: str) -> pd.DataFrame:
    cfg = SCALE_CONFIG[scale_key]

    atac_columns = select_atac_columns(read_header(cfg["atac"], sep="\t"))

    atac_means, atac_counts = load_window_means(
        cfg["atac"], sep="\t", column_groups=atac_columns, value_prefix="atac"
    )
    snv = load_mutation_table(cfg["snv"]).rename(
        columns={cancer: f"snv_{cancer}" for cancer in CANCER_ORDER}
    )
    indel = load_mutation_table(cfg["indel"]).rename(
        columns={cancer: f"indel_{cancer}" for cancer in CANCER_ORDER}
    )

    merged = snv.merge(indel, on=["chr", "start", "end"], how="inner")
    merged = merged.merge(atac_means, on=["chr", "start", "end"], how="inner")

    rows = []
    for cancer in CANCER_ORDER:
        metric_pairs = [
            ("SNV", f"snv_{cancer}", "SNV"),
            ("Indel", f"indel_{cancer}", "Indel"),
        ]
        for row_label, mutation_col, mutation_type in metric_pairs:
            feature_col = f"atac_{cancer}"
            rho, pvalue, n_windows = spearman_stats(merged[feature_col], merged[mutation_col])
            rows.append(
                {
                    "scale": scale_key,
                    "scale_label": cfg["label"],
                    "cancer": cancer,
                    "cancer_label": CANCER_TITLES[cancer],
                    "row_label": row_label,
                    "epigenome": "ATAC-seq",
                    "mutation_type": mutation_type,
                    "spearman_rho": rho,
                    "spearman_pvalue": pvalue,
                    "n_windows": n_windows,
                    "n_feature_columns": atac_counts[cancer],
                }
            )

    return pd.DataFrame(rows)


def pivot_metric(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        long_df.pivot(index="row_label", columns="cancer_label", values=value_col)
        .reindex(index=ROW_ORDER, columns=[CANCER_TITLES[c] for c in CANCER_ORDER])
    )


def setup_style(font_size: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": font_size,
            "axes.titlesize": font_size + 2,
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
    panel_rows = len(ROW_ORDER)
    panel_cols = len(CANCER_ORDER)
    fig_width = max(4.6, panel_cols * cell_size + 1.8)
    fig_height = max(4.4, len(SCALE_CONFIG) * panel_rows * cell_size + 2.0)
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


def plot_heatmaps(
    scale_tables: dict[str, pd.DataFrame],
    annotation_tables: dict[str, pd.DataFrame],
    out_prefix: Path,
    font_size: float,
    cell_size: float,
) -> None:
    setup_style(font_size)
    legend_path = out_prefix.with_name(f"{out_prefix.stem}_legend.pdf")
    fig_width, fig_height = compute_figure_size(cell_size)
    fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.22, right=0.98, top=0.96, bottom=0.08, hspace=0.22)
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
        ax.set_title(SCALE_CONFIG[scale_key]["label"])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=35, pad=2, length=0)
        ax.tick_params(axis="y", rotation=0, pad=1, length=0)
        for text in ax.texts:
            text.set_color(TEXT_COLOR)
        if idx != 2:
            ax.tick_params(axis="x", labelbottom=False)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    save_color_legend(legend_path, cmap=cmap, font_size=font_size)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    tsv_prefix = Path(args.tsv_prefix)

    long_tables = [build_scale_table(scale_key) for scale_key in SCALE_ORDER]

    all_long = pd.concat(long_tables, ignore_index=True)
    all_long["spearman_qvalue"] = benjamini_hochberg(all_long["spearman_pvalue"])
    all_long["significant_fdr_0_05"] = all_long["spearman_qvalue"] <= 0.05
    all_long["annotation_label"] = all_long.apply(
        lambda row: build_annotation_label(row["spearman_rho"], row["spearman_qvalue"]),
        axis=1,
    )
    tsv_prefix.parent.mkdir(parents=True, exist_ok=True)
    all_long.to_csv(tsv_prefix.with_name(f"{tsv_prefix.name}.long.tsv"), sep="\t", index=False)

    wide_tables = {}
    annotation_tables = {}
    for scale_key in SCALE_ORDER:
        scale_long = all_long.loc[all_long["scale"] == scale_key].copy()
        rho_table = pivot_metric(scale_long, "spearman_rho")
        pvalue_table = pivot_metric(scale_long, "spearman_pvalue")
        qvalue_table = pivot_metric(scale_long, "spearman_qvalue")
        sig_table = pivot_metric(scale_long, "significant_fdr_0_05")
        annotation_table = pivot_metric(scale_long, "annotation_label")

        wide_tables[scale_key] = rho_table
        annotation_tables[scale_key] = annotation_table

        rho_table.to_csv(
            tsv_prefix.with_name(f"{tsv_prefix.name}.{scale_key}.tsv"),
            sep="\t",
            na_rep="NA",
        )
        pvalue_table.to_csv(
            tsv_prefix.with_name(f"{tsv_prefix.name}.{scale_key}.pvalue.tsv"),
            sep="\t",
            na_rep="NA",
        )
        qvalue_table.to_csv(
            tsv_prefix.with_name(f"{tsv_prefix.name}.{scale_key}.qvalue.tsv"),
            sep="\t",
            na_rep="NA",
        )
        sig_table.to_csv(
            tsv_prefix.with_name(f"{tsv_prefix.name}.{scale_key}.significant.tsv"),
            sep="\t",
            na_rep="NA",
        )

    plot_heatmaps(
        wide_tables,
        annotation_tables,
        out_prefix=out_prefix,
        font_size=args.font_size,
        cell_size=args.cell_size,
    )


if __name__ == "__main__":
    main()
