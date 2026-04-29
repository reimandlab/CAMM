#!/usr/bin/env python3
"""
Plot one residual violin per cancer type using all 10kb windows, with only
outlier windows overlaid as dots.

Default inputs:
- <outlier_regions>/<cancer>/<cancer>_<variant>_10kb_pred_vs_obs_all.tsv
- <outlier_regions>/combined_10kb_underest_z4_step1.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CANCER_ORDER = ["breast", "colorectal", "skin", "lung", "prostate", "esophagus"]
VARIANT_ORDER = ["snv", "indel"]
VARIANT_LABELS = {"snv": "SNV", "indel": "Indel"}
VIOLIN_PALETTE = {"snv": "#7AA6DC", "indel": "#E39D63"}
DOT_PALETTE = {"snv": "#1F4E8C", "indel": "#A34F14"}
COMBINED_AXIS_OVERRIDES = {
    "lung": {
        "ylim": (-50, 100),
        "yticks": [-50, 0, 50, 100],
    }
}


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

    tested = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        f"Missing pred_vs_obs file for {cancer}/{variant}.\n"
        f"Tried:\n{tested}\n"
        f"And recursive pattern: {pattern}"
    )


def load_outlier_table(path: Path) -> pd.DataFrame:
    required = ["cancer_type", "variant", "residual"]
    df = pd.read_csv(path, sep="\t", usecols=required)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(missing)}")
    return df.copy()


def load_all_windows(input_dir: Path, cancer: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for variant in VARIANT_ORDER:
        path = find_pred_obs_file(input_dir, cancer, variant)
        part = pd.read_csv(path, sep="\t", usecols=["residual"]).copy()
        part["variant"] = variant
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def format_cancer_label(cancer: str) -> str:
    return cancer.replace("_", " ").title()


def draw_cancer_violin_ax(
    ax: plt.Axes,
    cancer: str,
    all_windows: pd.DataFrame,
    outliers: pd.DataFrame,
    point_size: float,
    point_alpha: float,
    show_ylabel: bool,
    font_size: float,
    axis_override: dict[str, object] | None = None,
) -> None:
    sns.violinplot(
        data=all_windows,
        x="variant",
        y="residual",
        hue="variant",
        order=VARIANT_ORDER,
        hue_order=VARIANT_ORDER,
        palette=VIOLIN_PALETTE,
        inner=None,
        cut=0,
        linewidth=1.2,
        saturation=1,
        density_norm="width",
        legend=False,
        ax=ax,
    )

    rng = np.random.default_rng(0)
    for x_pos, variant in enumerate(VARIANT_ORDER):
        subset = outliers[outliers["variant"] == variant]
        if subset.empty:
            continue
        x_jitter = x_pos + rng.uniform(-0.11, 0.11, size=len(subset))
        ax.scatter(
            x_jitter,
            subset["residual"],
            s=point_size,
            c=DOT_PALETTE[variant],
            alpha=point_alpha,
            linewidths=0,
            zorder=3,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Residual" if show_ylabel else "", fontsize=font_size)
    ax.set_xticks(range(len(VARIANT_ORDER)))
    ax.set_xticklabels([VARIANT_LABELS[variant] for variant in VARIANT_ORDER], fontsize=font_size)
    ax.set_title(format_cancer_label(cancer), fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    if axis_override is not None:
        ylim = axis_override.get("ylim")
        if ylim is not None:
            ax.set_ylim(*ylim)
        yticks = axis_override.get("yticks")
        if yticks is not None:
            ax.set_yticks(yticks)
    ax.grid(axis="y", color="#D0D0D0", alpha=0.45, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_cancer_violin(
    cancer: str,
    all_windows: pd.DataFrame,
    outliers: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    point_size: float,
    point_alpha: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    draw_cancer_violin_ax(
        ax=ax,
        cancer=cancer,
        all_windows=all_windows,
        outliers=outliers,
        point_size=point_size,
        point_alpha=point_alpha,
        show_ylabel=True,
        font_size=plt.rcParams["font.size"],
        axis_override=None,
    )

    fig.tight_layout()
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_combined_violins(
    cancer_windows: dict[str, pd.DataFrame],
    cancer_outliers: dict[str, pd.DataFrame],
    output_png: Path,
    output_pdf: Path,
    point_size: float,
    point_alpha: float,
    font_size: float,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 11.5))
    axes_flat = axes.flatten()

    for idx, cancer in enumerate(CANCER_ORDER):
        ax = axes_flat[idx]
        draw_cancer_violin_ax(
            ax=ax,
            cancer=cancer,
            all_windows=cancer_windows[cancer],
            outliers=cancer_outliers[cancer],
            point_size=point_size,
            point_alpha=point_alpha,
            show_ylabel=idx % 3 == 0,
            font_size=font_size,
            axis_override=COMBINED_AXIS_OVERRIDES.get(cancer),
        )

    fig.tight_layout()
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "outlier_regions"
    default_output = default_input / "plots" / "residual_violin_z4"

    parser = argparse.ArgumentParser(
        description="Plot per-cancer residual violins with z4 outlier windows overlaid as dots."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help="Directory containing outlier_regions files and cancer subdirectories.",
    )
    parser.add_argument(
        "--outlier-table",
        type=Path,
        default=default_input / "combined_10kb_underest_z4_step1.tsv",
        help="Step1 outlier table used for the overlaid dots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory for generated violin plots.",
    )
    parser.add_argument(
        "--font-family",
        type=str,
        default="Arial",
        help="Matplotlib font family.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=25,
        help="Base font size for the plots.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=70,
        help="Scatter point size for outlier windows.",
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.55,
        help="Scatter alpha for outlier windows.",
    )
    parser.add_argument(
        "--combined-font-size",
        type=float,
        default=None,
        help="Optional font size override for the combined 2x3 figure only.",
    )
    parser.add_argument(
        "--combined-point-size",
        type=float,
        default=None,
        help="Optional point size override for the combined 2x3 figure only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    outlier_table = args.outlier_table.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": args.font_family,
            "font.size": args.font_size,
            "axes.titlesize": args.font_size,
            "axes.labelsize": args.font_size,
            "xtick.labelsize": args.font_size,
            "ytick.labelsize": args.font_size,
            "legend.fontsize": args.font_size,
        }
    )

    outliers = load_outlier_table(outlier_table)
    cancer_windows: dict[str, pd.DataFrame] = {}
    cancer_outliers: dict[str, pd.DataFrame] = {}
    for cancer in CANCER_ORDER:
        cancer_windows[cancer] = load_all_windows(input_dir, cancer)
        cancer_outliers[cancer] = outliers[outliers["cancer_type"] == cancer].copy()
        base = f"{cancer}_residual_violin_with_outliers_z4"
        output_png = output_dir / f"{base}.png"
        output_pdf = output_dir / f"{base}.pdf"
        draw_cancer_violin(
            cancer=cancer,
            all_windows=cancer_windows[cancer],
            outliers=cancer_outliers[cancer],
            output_png=output_png,
            output_pdf=output_pdf,
            point_size=args.point_size,
            point_alpha=args.point_alpha,
        )
        print(f"Wrote {output_png}")
        print(f"Wrote {output_pdf}")

    combined_png = output_dir / "combined_residual_violin_with_outliers_z4.png"
    combined_pdf = output_dir / "combined_residual_violin_with_outliers_z4.pdf"
    draw_combined_violins(
        cancer_windows=cancer_windows,
        cancer_outliers=cancer_outliers,
        output_png=combined_png,
        output_pdf=combined_pdf,
        point_size=args.combined_point_size or args.point_size,
        point_alpha=args.point_alpha,
        font_size=args.combined_font_size or args.font_size,
    )
    print(f"Wrote {combined_png}")
    print(f"Wrote {combined_pdf}")


if __name__ == "__main__":
    main()
