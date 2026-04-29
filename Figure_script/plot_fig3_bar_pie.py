#!/usr/bin/env python3
"""Plot signed feature-output direction for top SHAP features.

The direction metric uses the row-level correlation between feature value and
SHAP attribution from `shap_beeswarm_10kb.tsv.gz`:

- positive correlation: higher feature values tend to increase model output
- negative correlation: higher feature values tend to decrease model output

This script writes:
- one summary TSV per cancer type
- a combined summary TSV
- one high-resolution panel per cancer type
- one combined multi-panel figure
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


CANCER_META = {
    "breast": {"aliases": {"brca"}, "color": "#C65A7A", "title": "Breast"},
    "colorectal": {"aliases": {"coad", "read"}, "color": "#2F7E9E", "title": "Colorectal"},
    "esophagus": {"aliases": {"esca"}, "color": "#B86A2E", "title": "Esophagus"},
    "lung": {"aliases": {"luad", "lusc"}, "color": "#4A8C56", "title": "Lung"},
    "prostate": {"aliases": {"prad"}, "color": "#7B5EA7", "title": "Prostate"},
    "skin": {"aliases": {"skcm"}, "color": "#C7931A", "title": "Skin"},
}

UNMATCHED_COLOR = "#9EA4AD"
UNMATCHED_FILL = "#D8DCE2"
TEXT_COLOR = "#222222"
GRID_COLOR = "#D8DDE3"
NEG_BG = "#F2F6FA"
POS_BG = "#FBF3F0"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot directional feature-output correlation for top SHAP features."
    )
    parser.add_argument(
        "--tag",
        default="top30",
        help="Feature set tag, typically top30 or top5pct.",
    )
    parser.add_argument(
        "--method",
        choices=("spearman", "pearson"),
        default="spearman",
        help="Correlation metric to plot on the x-axis.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=30,
        help="Maximum number of features to keep per cancer type.",
    )
    parser.add_argument(
        "--matched-only",
        action="store_true",
        help="Only plot features whose prefix matches the target cancer type.",
    )
    return parser.parse_args()


def feature_code(feature: str) -> str:
    return str(feature).split("_", 1)[0].lower()


def shorten_feature_name(feature: str) -> str:
    feature = str(feature)
    parts = feature.split("_")
    head = parts[0].upper()
    if len(parts) == 1:
        return head

    tail = None
    for token in reversed(parts[1:]):
        if re.fullmatch(r"p\d+", token) or token == "pmrg" or re.fullmatch(r"s\d+", token):
            tail = token
            break
    if tail is None:
        tail = parts[-1]

    t_token = next((token for token in reversed(parts) if re.fullmatch(r"t\d+", token)), None)
    if tail.startswith("p") and t_token is not None:
        tail = f"{t_token}.{tail}"

    return f"{head} {tail}"


def resolve_top_path(cancer_dir: Path, tag: str) -> Path | None:
    candidates = [
        cancer_dir / f"{tag}_shap_features_in_permutation_significant_10kb.tsv",
        cancer_dir / f"{tag}_significant_features.tsv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_shap_path(cancer_dir: Path) -> Path | None:
    candidates = [
        cancer_dir / "shap_beeswarm_10kb.tsv.gz",
        cancer_dir / "shap_beeswarm_10kb.tsv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def numeric_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3:
        return math.nan
    if pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return math.nan

    if method == "spearman":
        x_vals = pair["x"].rank(method="average").to_numpy(dtype=float)
        y_vals = pair["y"].rank(method="average").to_numpy(dtype=float)
    else:
        x_vals = pair["x"].to_numpy(dtype=float)
        y_vals = pair["y"].to_numpy(dtype=float)

    corr = np.corrcoef(x_vals, y_vals)[0, 1]
    return float(corr)


def load_direction_table(cancer_dir: Path, tag: str, max_features: int) -> pd.DataFrame:
    top_path = resolve_top_path(cancer_dir, tag)
    shap_path = resolve_shap_path(cancer_dir)
    if top_path is None or shap_path is None:
        raise FileNotFoundError(f"Missing required files for {cancer_dir.name}")

    top_df = pd.read_csv(top_path, sep="\t")
    required_top = {"feature", "mean_abs_attr"}
    missing_top = required_top - set(top_df.columns)
    if missing_top:
        raise ValueError(f"{top_path} missing columns: {sorted(missing_top)}")

    rank_col = next((c for c in top_df.columns if c.startswith("shap_rank_within_")), None)
    if rank_col is not None:
        top_df = top_df.sort_values(rank_col, ascending=True)
    else:
        top_df = top_df.sort_values("mean_abs_attr", ascending=False)
    top_df = top_df.head(max_features).copy()

    shap_df = pd.read_csv(shap_path, sep="\t", usecols=["row", "feature", "value", "attr"])
    shap_df = shap_df[shap_df["feature"].isin(top_df["feature"])].copy()
    if shap_df.empty:
        raise ValueError(f"No overlap between {top_path.name} and {shap_path.name}")

    cancer_type = cancer_dir.name
    aliases = CANCER_META.get(cancer_type, {}).get("aliases", {cancer_type})
    meta_cols = [c for c in ["mean_abs_attr", "mean_attr", rank_col] if c]
    meta_df = top_df[["feature", *meta_cols]].copy()

    summary_rows = []
    for feature, group in shap_df.groupby("feature", sort=False):
        meta_row = meta_df.loc[meta_df["feature"] == feature].iloc[0]
        code = feature_code(feature)
        summary_rows.append(
            {
                "cancer_type": cancer_type,
                "feature": feature,
                "display_feature": shorten_feature_name(feature),
                "feature_code": code.upper(),
                "is_matched": code in aliases,
                "n_windows": int(group[["value", "attr"]].dropna().shape[0]),
                "spearman_rho": numeric_corr(group["value"], group["attr"], method="spearman"),
                "pearson_r": numeric_corr(group["value"], group["attr"], method="pearson"),
                "mean_abs_attr": float(meta_row["mean_abs_attr"]),
                "mean_attr": float(meta_row["mean_attr"]) if "mean_attr" in meta_row else math.nan,
                "rank": int(meta_row[rank_col]) if rank_col and not pd.isna(meta_row[rank_col]) else math.nan,
            }
        )

    summary = pd.DataFrame(summary_rows)
    if rank_col is not None and summary["rank"].notna().any():
        summary = summary.sort_values(["rank", "mean_abs_attr"], ascending=[True, False])
    else:
        summary = summary.sort_values("mean_abs_attr", ascending=False)
    summary["direction"] = np.where(
        summary["spearman_rho"].fillna(0) > 0,
        "positive",
        np.where(summary["spearman_rho"].fillna(0) < 0, "negative", "flat"),
    )
    return summary.reset_index(drop=True)


def size_mapper(values: pd.Series, global_min: float, global_max: float) -> np.ndarray:
    if np.isclose(global_min, global_max):
        return np.full(len(values), 90.0)
    scaled = (values.to_numpy(dtype=float) - global_min) / (global_max - global_min)
    scaled = np.clip(scaled, 0, 1)
    return 36.0 + 150.0 * scaled


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors="#333333")
    ax.set_axisbelow(True)


def plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    cancer_type: str,
    metric_col: str,
    global_min: float,
    global_max: float,
) -> None:
    meta = CANCER_META.get(
        cancer_type,
        {"color": "#4C78A8", "title": cancer_type.replace("_", " ").title()},
    )
    accent = meta["color"]

    df = df.reset_index(drop=True).copy()
    y_pos = np.arange(len(df))
    metric_vals = df[metric_col].fillna(0)
    marker_sizes = size_mapper(df["mean_abs_attr"], global_min=global_min, global_max=global_max)

    ax.axvspan(-1, 0, color=NEG_BG, zorder=0)
    ax.axvspan(0, 1, color=POS_BG, zorder=0)
    ax.axvline(0, color="#444444", linewidth=1.0, zorder=1)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.9)

    for y, (_, row) in zip(y_pos, df.iterrows()):
        line_color = accent if row["is_matched"] else UNMATCHED_COLOR
        line_width = 2.0 if row["is_matched"] else 1.3
        ax.hlines(
            y,
            xmin=0,
            xmax=row[metric_col] if pd.notna(row[metric_col]) else 0,
            color=line_color,
            linewidth=line_width,
            alpha=0.95 if row["is_matched"] else 0.8,
            zorder=2,
        )

    facecolors = [accent if matched else UNMATCHED_FILL for matched in df["is_matched"]]
    edgecolors = [accent if matched else UNMATCHED_COLOR for matched in df["is_matched"]]
    ax.scatter(
        metric_vals,
        y_pos,
        s=marker_sizes,
        c=facecolors,
        edgecolors=edgecolors,
        linewidths=0.8,
        zorder=3,
    )

    ax.set_xlim(-1.02, 1.02)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks(y_pos)
    y_labels = ax.set_yticklabels(df["display_feature"])
    for label, matched in zip(y_labels, df["is_matched"]):
        label.set_color(accent if matched else TEXT_COLOR)
        label.set_fontweight("semibold" if matched else "normal")

    ax.invert_yaxis()
    ax.set_title(meta["title"], loc="left", color=accent, fontweight="bold", pad=6)
    ax.text(
        0.01,
        1.01,
        f"matched features highlighted ({int(df['is_matched'].sum())}/{len(df)})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    style_axis(ax)


def save_single_panel(
    df: pd.DataFrame,
    cancer_type: str,
    metric_col: str,
    metric_label: str,
    global_min: float,
    global_max: float,
    out_dir: Path,
) -> None:
    fig_height = max(4.8, 0.28 * len(df) + 1.8)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    plot_panel(ax, df, cancer_type, metric_col, global_min=global_min, global_max=global_max)
    ax.set_xlabel(metric_label)
    fig.text(
        0.125,
        0.02,
        "Negative: higher feature values lower output    Positive: higher feature values raise output",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    png_path = out_dir / f"{cancer_type}_directional_{metric_col}.png"
    pdf_path = out_dir / f"{cancer_type}_directional_{metric_col}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def save_combined_figure(
    all_tables: dict[str, pd.DataFrame],
    metric_col: str,
    metric_label: str,
    global_min: float,
    global_max: float,
    out_dir: Path,
) -> None:
    ordered_types = [ct for ct in CANCER_META if ct in all_tables]
    ncols = 2
    nrows = math.ceil(len(ordered_types) / ncols)
    height_ratios = []
    for row_idx in range(nrows):
        row_types = ordered_types[row_idx * ncols : (row_idx + 1) * ncols]
        row_max = max(len(all_tables[ct]) for ct in row_types)
        height_ratios.append(max(1.0, 0.22 * row_max))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(15.5, sum(height_ratios) + 1.8),
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax in axes_arr.ravel():
        ax.set_visible(False)

    for idx, cancer_type in enumerate(ordered_types):
        ax = axes_arr[idx // ncols, idx % ncols]
        ax.set_visible(True)
        plot_panel(
            ax,
            all_tables[cancer_type],
            cancer_type,
            metric_col,
            global_min=global_min,
            global_max=global_max,
        )
        if idx // ncols == nrows - 1:
            ax.set_xlabel(metric_label)

    matched_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="#555555",
        markerfacecolor="#555555",
        markersize=6,
        linewidth=0,
        label="Matched tissue feature",
    )
    other_handle = Line2D(
        [0],
        [0],
        marker="o",
        color=UNMATCHED_COLOR,
        markerfacecolor=UNMATCHED_FILL,
        markersize=6,
        linewidth=0,
        label="Other feature",
    )

    example_sizes = np.quantile(
        pd.concat([table["mean_abs_attr"] for table in all_tables.values()]).to_numpy(dtype=float),
        [0.2, 0.5, 0.8],
    )
    size_handles = [
        plt.scatter([], [], s=size_mapper(pd.Series([size]), global_min, global_max)[0], color="#666666")
        for size in example_sizes
    ]
    size_labels = [f"mean|SHAP| {size:.2f}" for size in example_sizes]

    fig.legend(
        handles=[matched_handle, other_handle, *size_handles],
        labels=["Matched tissue feature", "Other feature", *size_labels],
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=9,
        handletextpad=0.8,
        columnspacing=1.5,
    )
    fig.text(
        0.5,
        0.985,
        "Directional effect of top SHAP features across cancer types",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    fig.text(
        0.5,
        0.968,
        "Direction = correlation between feature value and SHAP contribution across 10-kb windows",
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
    )
    fig.text(
        0.5,
        0.02,
        "Negative: higher feature values lower output    Positive: higher feature values raise output",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=[0.03, 0.05, 0.97, 0.94])

    png_path = out_dir / f"all_cancers_directional_{metric_col}.png"
    pdf_path = out_dir / f"all_cancers_directional_{metric_col}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_style()
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    out_name = f"feature_direction_{args.tag}"
    if args.matched_only:
        out_name = f"{out_name}_matched_only"
    out_dir = root / "plots" / out_name
    panel_dir = out_dir / "panels"
    table_dir = out_dir / "tables"
    panel_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    metric_col = "spearman_rho" if args.method == "spearman" else "pearson_r"
    metric_label = (
        "Feature-output direction (Spearman rho)"
        if args.method == "spearman"
        else "Feature-output direction (Pearson r)"
    )

    cancer_dirs = [root / cancer_type for cancer_type in CANCER_META if (root / cancer_type).is_dir()]
    all_tables: dict[str, pd.DataFrame] = {}

    for cancer_dir in cancer_dirs:
        table = load_direction_table(cancer_dir, tag=args.tag, max_features=args.max_features)
        if args.matched_only:
            table = table[table["is_matched"]].reset_index(drop=True)
        if table.empty:
            continue

        out_table_path = table_dir / f"{cancer_dir.name}_{args.tag}_directional_summary.tsv"
        table.to_csv(out_table_path, sep="\t", index=False)
        all_tables[cancer_dir.name] = table

    if not all_tables:
        raise SystemExit("No cancer types produced directional summaries.")

    combined = pd.concat(all_tables.values(), ignore_index=True)
    combined_path = table_dir / f"all_cancers_{args.tag}_directional_summary.tsv"
    combined.to_csv(combined_path, sep="\t", index=False)

    global_min = float(combined["mean_abs_attr"].min())
    global_max = float(combined["mean_abs_attr"].max())

    for cancer_type, table in all_tables.items():
        save_single_panel(
            table,
            cancer_type,
            metric_col=metric_col,
            metric_label=metric_label,
            global_min=global_min,
            global_max=global_max,
            out_dir=panel_dir,
        )

    save_combined_figure(
        all_tables,
        metric_col=metric_col,
        metric_label=metric_label,
        global_min=global_min,
        global_max=global_max,
        out_dir=out_dir,
    )

    print(f"Saved summaries and plots under {out_dir}")


if __name__ == "__main__":
    main()
