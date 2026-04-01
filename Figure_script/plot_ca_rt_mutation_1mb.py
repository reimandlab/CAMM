#!/usr/bin/env python3
"""
Create a 3-panel figure for new_ca_rt_mutation data:
- Top: CA_RT signal (median across cancer types)
- Middle: SNV mutations (pancancer)
- Bottom: INDEL mutations (pancancer)
All using 1MB resolution data with shared genome window x-axis.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white", context="paper", font_scale=1.2)

def load_ca_rt_data(ca_rt_path):
    """Load CA_RT data and compute median across cancer samples"""
    print(f"📥 Loading CA_RT data: {ca_rt_path}")
    df = pd.read_csv(ca_rt_path, sep='\t')

    # Get coordinate columns
    coord_cols = ['chr', 'start', 'end']

    # Get sample columns (exclude coordinates)
    sample_cols = [col for col in df.columns if col not in coord_cols]

    # Filter for cancer samples (exclude normal samples)
    cancer_cols = [col for col in sample_cols if 'Cancer' in col or 'cancer' in col or 'tumor' in col or 'Tumor' in col]

    if len(cancer_cols) == 0:
        # If no clear cancer samples, use all samples
        cancer_cols = sample_cols
        print(f"   No clear cancer samples found, using all {len(cancer_cols)} samples")
    else:
        print(f"   Found {len(cancer_cols)} cancer samples out of {len(sample_cols)} total samples")

    print(f"   Rows: {len(df):,}")

    # Compute median CA_RT across cancer samples
    ca_median = df[cancer_cols].median(axis=1, skipna=True).astype(float)

    return df[coord_cols].copy(), ca_median, cancer_cols

def load_mutation_data(mut_path, mut_type, cancer_type='pancancer'):
    """Load mutation data (SNV or INDEL)"""
    print(f"📥 Loading {mut_type} data: {mut_path}")
    df = pd.read_csv(mut_path)

    # Get coordinate columns
    coord_cols = ['chr', 'start', 'end']
    cancer_cols = [col for col in df.columns if col not in coord_cols]

    # Select mutation signal based on cancer type
    if cancer_type == 'pancancer':
        if 'pancancer' in df.columns:
            mut_signal = df['pancancer'].astype(float)
        else:
            # Fallback: sum all non-coordinate columns
            mut_signal = df[cancer_cols].sum(axis=1).astype(float)
    elif cancer_type in df.columns:
        mut_signal = df[cancer_type].astype(float)
    else:
        raise ValueError(f"Cancer type '{cancer_type}' not found in {mut_type} data. Available: {cancer_cols}")

    print(f"   Rows: {len(df):,}")
    print(f"   Cancer type: {cancer_type}")
    print(f"   Total {mut_type} mutations: {mut_signal.sum():,}")

    return df[coord_cols].copy(), mut_signal, cancer_cols

def sort_by_genome(df):
    """Sort dataframe by chromosome and start position"""
    def chr_key(series):
        out = []
        for s in series.astype(str):
            if s.startswith('chr'):
                s = s[3:]
            if s == 'X':
                out.append(23)
            elif s == 'Y':
                out.append(24)
            else:
                try:
                    out.append(int(s))
                except:
                    out.append(99)
        return np.array(out)

    order = np.lexsort((df['start'].values, chr_key(df['chr'])))
    return df.iloc[order].reset_index(drop=True)

def smooth_series(y, win=31):
    """Apply rolling median smoothing"""
    if win <= 1:
        return y
    s = pd.Series(y)
    sm = s.rolling(window=win, center=True, min_periods=max(1, win//4)).median().to_numpy()
    mask = np.isnan(sm)
    sm[mask] = s.to_numpy()[mask]
    return sm

def winsorize(y, upper_pct=99.0):
    """Cap extreme values at specified percentile"""
    if upper_pct is None:
        return y
    p = np.nanpercentile(y, upper_pct)
    y = y.copy()
    y[y > p] = p
    return y

def apply_transform(y, method):
    import numpy as np
    y = y.astype(float)
    if method == 'none':
        return y
    if method == 'log1p':
        y_clipped = np.where(np.isnan(y), np.nan, np.maximum(y, 0.0))
        return np.log1p(y_clipped)
    if method == 'log2':
        y_clipped = np.where(np.isnan(y), np.nan, np.maximum(y, 0.0))
        return np.log2(1.0 + y_clipped)
    if method == 'zscore':
        m = np.nanmean(y)
        s = np.nanstd(y)
        if not np.isfinite(s) or s == 0:
            return np.zeros_like(y)
        return (y - m) / s
    # Fallback
    return y


def transform_suffix(method):
    if method == 'none':
        return ''
    if method == 'log1p':
        return ' (log1p)'
    if method == 'log2':
        return ' (log2)'
    if method == 'zscore':
        return ' (z-score)'
    if method == 'dual-axis':
        return ' (dual y-axes)'
    return ''

def chromosome_boundaries(chrs):
    """Find chromosome boundaries for plotting vertical lines"""
    b_ix = [0]
    labels = []
    prev = chrs[0]
    for i, c in enumerate(chrs):
        if c != prev:
            b_ix.append(i)
            labels.append(prev)
            prev = c
    b_ix.append(len(chrs))
    labels.append(prev)

    # Midpoints for labeling
    mids = [(b_ix[i] + b_ix[i+1]) // 2 for i in range(len(b_ix)-1)]

    # Clean labels (strip 'chr')
    lab = [str(l)[3:] if str(l).startswith('chr') else str(l) for l in labels]
    return b_ix, mids, lab

def save_legend_figure(handles, labels, out_path, dpi=600, font_size=12.0,
                       fig_width=12.0, fig_height=1.4):
    """Save a standalone legend-only figure."""
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.legend(handles, labels, loc='center', ncol=len(labels),
               frameon=False, prop={'size': font_size})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', transparent=True)
    if out_path.lower().endswith('.png'):
        fig.savefig(out_path[:-4] + '.pdf', bbox_inches='tight', transparent=True)
    plt.close(fig)

def plot_ca_rt_mutations(ca_coords, ca_signal, snv_signal, indel_signal,
                        out_png, smooth_ca=31, smooth_mut=31, winsor_pct=99.0,
                        mask_zeros=True, add_chrom_bars=True, dpi=600, save_pdf=False):
    """Create 3-panel plot: CA_RT + SNV + INDEL"""

    # Sort all data by genomic coordinates
    combined_df = ca_coords.copy()
    combined_df['ca_signal'] = ca_signal
    combined_df['snv_signal'] = snv_signal
    combined_df['indel_signal'] = indel_signal
    combined_df = sort_by_genome(combined_df)

    # Extract sorted signals
    y_ca = combined_df['ca_signal'].to_numpy()
    y_snv = combined_df['snv_signal'].to_numpy()
    y_indel = combined_df['indel_signal'].to_numpy()
    chrs = combined_df['chr'].to_numpy()

    # Process mutation signals
    if mask_zeros:
        y_snv = y_snv.copy()
        y_indel = y_indel.copy()
        y_snv[y_snv == 0] = np.nan
        y_indel[y_indel == 0] = np.nan

    # Winsorize extreme values
    y_snv = winsorize(y_snv, upper_pct=winsor_pct)
    y_indel = winsorize(y_indel, upper_pct=winsor_pct)

    # Apply smoothing
    if smooth_ca and smooth_ca > 1:
        y_ca = smooth_series(y_ca, win=smooth_ca)
    if smooth_mut and smooth_mut > 1:
        y_snv = smooth_series(y_snv, win=smooth_mut)
        y_indel = smooth_series(y_indel, win=smooth_mut)

    # X axis
    x = np.arange(1, len(y_ca) + 1)

    # Setup figure with 3 panels
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 9),
                            gridspec_kw={'hspace': 0.2, 'height_ratios': [1, 1, 1]})

    # Top panel: CA_RT
    ax1 = axes[0]
    ax1.plot(x, y_ca, color='#2E8B57', alpha=0.95, linewidth=1.2, label='CA_RT (median)')
    ax1.set_ylabel('CA_RT Signal')
    ax1.set_title('Chromatin Accessibility (CA_RT) across 1MB genome windows')
    ax1.legend(loc='upper right', frameon=True, framealpha=0.85, facecolor='white', edgecolor='none')
    sns.despine(ax=ax1)

def plot_ca_rt_mutations(ca_coords, ca_signal, snv_signal, indel_signal,
                        out_png, cancer_type='pancancer', smooth_ca=31, smooth_mut=31, winsor_pct=99.0,
                        mask_zeros=True, add_chrom_bars=True, dpi=600, save_pdf=False):
    """Create 2-panel plot: CA + separate SNV and INDEL lines"""

    # Sort all data by genomic coordinates
    combined_df = ca_coords.copy()
    combined_df['ca_signal'] = ca_signal
    combined_df['snv_signal'] = snv_signal
    combined_df['indel_signal'] = indel_signal
    combined_df = sort_by_genome(combined_df)

    # Extract sorted signals
    y_ca = combined_df['ca_signal'].to_numpy()
    y_snv = combined_df['snv_signal'].to_numpy()
    y_indel = combined_df['indel_signal'].to_numpy()
    chrs = combined_df['chr'].to_numpy()

    # Process mutation signals
    if mask_zeros:
        y_snv = y_snv.copy()
        y_indel = y_indel.copy()
        y_snv[y_snv == 0] = np.nan
        y_indel[y_indel == 0] = np.nan

    # Winsorize extreme values
    y_snv = winsorize(y_snv, upper_pct=winsor_pct)
    y_indel = winsorize(y_indel, upper_pct=winsor_pct)

    # Apply smoothing
    if smooth_ca and smooth_ca > 1:
        y_ca = smooth_series(y_ca, win=smooth_ca)
    if smooth_mut and smooth_mut > 1:
        y_snv = smooth_series(y_snv, win=smooth_mut)
        y_indel = smooth_series(y_indel, win=smooth_mut)

    # X axis
    x = np.arange(1, len(y_ca) + 1)

    # Setup figure with 2 panels
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 8),
                            gridspec_kw={'hspace': 0.25, 'height_ratios': [1, 1]})

    # Top panel: CA
    ax1 = axes[0]
    ax1.plot(x, y_ca, color='#2E8B57', alpha=0.95, linewidth=1.2, label='CA (median)')
    ax1.set_ylabel('CA Signal')
    ax1.set_title('Chromatin Accessibility (CA) across 1MB genome windows')
    ax1.legend(loc='upper right', frameon=True, framealpha=0.85, facecolor='white', edgecolor='none')
    sns.despine(ax=ax1)

    # Bottom panel: Separate SNV and INDEL lines
    ax2 = axes[1]
    snv_label = f'SNV ({cancer_type})'
    indel_label = f'INDEL ({cancer_type})'
    ax2.plot(x, y_snv, color='#1f77b4', linewidth=1.2, alpha=0.95, label=snv_label)
    ax2.plot(x, y_indel, color='#d62728', linewidth=1.2, alpha=0.95, label=indel_label)
    ax2.set_xlabel('Genome window index (sorted by chr, start)')
    ax2.set_ylabel('Mutation Count')
    ax2.set_title(f'Mutations - {cancer_type} across 1MB genome windows')
    ax2.legend(loc='upper right', frameon=True, framealpha=0.85, facecolor='white', edgecolor='none')
    sns.despine(ax=ax2)

    # Add chromosome boundaries and labels
    if add_chrom_bars:
        chrs_str = chrs.astype(str)
        b_ix, mids, labs = chromosome_boundaries(chrs_str)

        # Add vertical lines at chromosome boundaries
        for bx in b_ix[1:-1]:
            for ax in axes:
                ax.axvline(bx, color='lightgray', linewidth=0.6, alpha=0.6)

        # Label every other chromosome to avoid clutter
        tick_ix = [mids[i] for i in range(len(mids)) if i % 2 == 0]
        tick_labs = [labs[i] for i in range(len(labs)) if i % 2 == 0]
        ax2.set_xticks(tick_ix)
        ax2.set_xticklabels(tick_labs, fontsize=12)

    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches='tight')
    if save_pdf and out_png.lower().endswith('.png'):
        plt.savefig(out_png[:-4] + '.pdf', bbox_inches='tight')
    plt.close()

    print(f"✅ Saved combined figure: {out_png}")
    if save_pdf:
        print(f"✅ Saved PDF: {out_png[:-4] + '.pdf'}")

def load_tcga_ca_and_rt_from_gz(ca_rt_path):
    """Load CA (TCGA-only) and RT (RepliSeq phases) medians from a gzipped 1MB matrix.
    - CA = median across TCGA cancer-type columns (prefixes like BRCA_, LUAD_, etc.)
    - RT = median across columns ending with one of: G1b, S1, S2, S3, S4, G2
    Returns: coords_df, ca_median, rt_median, ca_cols, rt_cols
    """
    print(f"\U0001F4E5 Loading CA/RT (gz): {ca_rt_path}")
    df = pd.read_csv(ca_rt_path, sep='\t', compression='infer')

    coord_cols = ['chr', 'start', 'end']
    sample_cols = [c for c in df.columns if c not in coord_cols]

    # Identify RT columns by suffix
    rt_suffixes = ("G1b", "S1", "S2", "S3", "S4", "G2")
    rt_cols = [c for c in sample_cols if any(c.endswith(suf) for suf in rt_suffixes)]

    # Identify TCGA CA columns by prefixes
    tcga_prefixes = (
        'ACCx_', 'BLCA_', 'BRCA_', 'CESC_', 'CHOL_', 'COAD_', 'ESCA_', 'GBMx_',
        'HNSC_', 'KIRC_', 'KIRP_', 'LGGx_', 'LIHC_', 'LUAD_', 'LUSC_', 'MESO_',
        'PCPG_', 'PRAD_', 'SKCM_', 'STAD_', 'TGCT_', 'THCA_', 'UCEC_'
    )
    ca_cols = [c for c in sample_cols if c.startswith(tcga_prefixes) and c not in rt_cols]
    if len(ca_cols) == 0:
        # Fallback: use all non-RT columns
        ca_cols = [c for c in sample_cols if c not in rt_cols]
        print(f"   ⚠️ No TCGA-prefixed columns found; using all non-RT columns ({len(ca_cols)}) as CA")
    else:
        print(f"   Found {len(ca_cols)} TCGA CA columns and {len(rt_cols)} RT columns")

    ca_median = df[ca_cols].median(axis=1, skipna=True).astype(float)
    rt_median = df[rt_cols].median(axis=1, skipna=True).astype(float) if len(rt_cols) else pd.Series(np.nan, index=df.index)

    return df[coord_cols].copy(), ca_median, rt_median, ca_cols, rt_cols


def plot_ca_rt_mutations_with_rt(ca_coords, ca_signal, rt_signal, snv_signal, indel_signal,
                                 out_png, cancer_type='pancancer', smooth_ca=31, smooth_mut=31, winsor_pct=99.0,
                                 transform_input='log1p', transform_output='log1p',
                                 ca_count=None, rt_count=None,
                                 mask_zeros=True, add_chrom_bars=True, dpi=600, save_pdf=False,
                                 fig_width=14.0, fig_height=9.0,
                                 title_fontsize=14.0, label_fontsize=13.0,
                                 tick_fontsize=12.0, legend_fontsize=12.0,
                                 legend_out=None, hide_legends=False,
                                 legend_width=None, legend_height=1.4):
    """Create 3-panel plot: CA (top), RT (middle), SNV+INDEL (bottom) with shared X.
    Style matches the existing 'breast_separate' figure for CA/mutations; RT is added as a new middle panel.
    """
    # Sort all data by genomic coordinates
    combined_df = ca_coords.copy()
    combined_df['ca_signal'] = ca_signal
    combined_df['rt_signal'] = rt_signal
    combined_df['snv_signal'] = snv_signal
    combined_df['indel_signal'] = indel_signal
    combined_df = sort_by_genome(combined_df)

    # Extract sorted signals
    y_ca = combined_df['ca_signal'].to_numpy()
    y_rt = combined_df['rt_signal'].to_numpy()
    y_snv = combined_df['snv_signal'].to_numpy()
    y_indel = combined_df['indel_signal'].to_numpy()
    chrs = combined_df['chr'].to_numpy()

    # Process mutation signals
    if mask_zeros:
        y_snv = y_snv.copy(); y_snv[y_snv == 0] = np.nan
        y_indel = y_indel.copy(); y_indel[y_indel == 0] = np.nan

    # Winsorize mutation extremes
    y_snv = winsorize(y_snv, upper_pct=winsor_pct)
    y_indel = winsorize(y_indel, upper_pct=winsor_pct)

    # Smoothing
    if smooth_ca and smooth_ca > 1:
        y_ca = smooth_series(y_ca, win=smooth_ca)
        y_rt = smooth_series(y_rt, win=smooth_ca)
    if smooth_mut and smooth_mut > 1:
        y_snv = smooth_series(y_snv, win=smooth_mut)
        y_indel = smooth_series(y_indel, win=smooth_mut)

    x = np.arange(1, len(y_ca) + 1)

    # Panel titles requested by user
    if ca_count is not None and rt_count is not None:
        top_title = f"{ca_count} Chromatin accessibility (CA) & {rt_count} Replication timing (RT)"
    else:
        top_title = "Chromatin accessibility (CA) & Replication timing (RT)"
    bottom_title = "Somatic mutations: SNVs and INDELs"

    # Figure with 2 panels (Input: CA+RT; Output: SNV+INDEL)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(fig_width, fig_height),
                            gridspec_kw={'hspace': 0.25, 'height_ratios': [1, 1]})

    # Top: Input features (CA + RT)
    ax1 = axes[0]
    ax1b = None
    in_suffix = transform_suffix(transform_input)
    if transform_input == 'dual-axis':
        # Plot on dual y-axes without transforming values
        ln_ca, = ax1.plot(x, y_ca, color='#009E73', alpha=0.95, linewidth=1.2, label='CA (median)')
        ax1.set_ylabel('CA (left axis)', fontsize=label_fontsize)
        ax1b = ax1.twinx()
        ln_rt, = ax1b.plot(x, y_rt, color='#0072B2', alpha=0.95, linewidth=1.2, label='RT (median)')
        ax1b.set_ylabel('RT (right axis)', fontsize=label_fontsize)
        ax1.set_title(top_title, fontsize=title_fontsize)
        input_handles = [ln_rt, ln_ca]
        input_labels = ['RT (median)', 'CA (median)']
        if not hide_legends:
            ax1.legend(input_handles, input_labels, loc='upper right', bbox_to_anchor=(1, 1.12),
                       frameon=True, framealpha=0.85, facecolor='white', edgecolor='none', prop={'size': legend_fontsize})
        ax1.tick_params(axis='both', labelsize=tick_fontsize)
        ax1b.tick_params(axis='both', labelsize=tick_fontsize)
    else:
        y_ca_plot = apply_transform(y_ca, transform_input)
        y_rt_plot = apply_transform(y_rt, transform_input)
        ln_ca, = ax1.plot(x, y_ca_plot, color='#009E73', alpha=0.95, linewidth=1.2, label='CA (median)')
        ln_rt, = ax1.plot(x, y_rt_plot, color='#0072B2', alpha=0.95, linewidth=1.2, label='RT (median)')
        ax1.set_ylabel('CA, RT' + in_suffix, fontsize=label_fontsize)
        ax1.set_title(top_title, fontsize=title_fontsize)
        input_handles = [ln_rt, ln_ca]
        input_labels = ['RT (median)', 'CA (median)']
        if not hide_legends:
            ax1.legend(input_handles, input_labels,
                       loc='upper right', bbox_to_anchor=(1, 1.12),
                       frameon=True, framealpha=0.85, facecolor='white', edgecolor='none', prop={'size': legend_fontsize})
        ax1.tick_params(axis='both', labelsize=tick_fontsize)
    sns.despine(ax=ax1)

    # Bottom: Output (SNV + INDEL)
    ax2 = axes[1]
    ax2b = None
    out_suffix = transform_suffix(transform_output)
    snv_label = 'SNV'
    indel_label = 'INDEL'
    if transform_output == 'dual-axis':
        ln_snv, = ax2.plot(x, y_snv, color='#D55E00', linewidth=1.2, alpha=0.95, label=snv_label)
        ax2.set_ylabel('SNV (left axis)', fontsize=label_fontsize)
        ax2b = ax2.twinx()
        ln_indel, = ax2b.plot(x, y_indel, color='#CC79A7', linewidth=1.2, alpha=0.95, label=indel_label)
        ax2b.set_ylabel('INDEL (right axis)', fontsize=label_fontsize)
        ax2.set_title(bottom_title, fontsize=title_fontsize)
        output_handles = [ln_snv, ln_indel]
        output_labels = [snv_label, indel_label]
        if not hide_legends:
            ax2.legend([ln_snv], [snv_label], loc='upper left', bbox_to_anchor=(0, 1.12), frameon=True, framealpha=0.85, facecolor='white', edgecolor='none', prop={'size': legend_fontsize})
            ax2b.legend([ln_indel], [indel_label], loc='upper right', bbox_to_anchor=(1, 1.12), frameon=True, framealpha=0.85, facecolor='white', edgecolor='none', prop={'size': legend_fontsize})
        ax2.tick_params(axis='both', labelsize=tick_fontsize)
        ax2b.tick_params(axis='both', labelsize=tick_fontsize)
    else:
        y_snv_plot = apply_transform(y_snv, transform_output)
        y_indel_plot = apply_transform(y_indel, transform_output)
        ln_snv, = ax2.plot(x, y_snv_plot, color='#D55E00', linewidth=1.2, alpha=0.95, label=snv_label)
        ln_indel, = ax2.plot(x, y_indel_plot, color='#CC79A7', linewidth=1.2, alpha=0.95, label=indel_label)
        ax2.set_ylabel('Mutation Count' + out_suffix, fontsize=label_fontsize)

        ax2.set_title(bottom_title, fontsize=title_fontsize)
        output_handles = [ln_snv, ln_indel]
        output_labels = [snv_label, indel_label]
        if not hide_legends:
            ax2.legend(output_handles, output_labels, loc='upper right', bbox_to_anchor=(1, 1.12), frameon=True, framealpha=0.85, facecolor='white', edgecolor='none', prop={'size': legend_fontsize})
        ax2.tick_params(axis='both', labelsize=tick_fontsize)
    ax2.set_xlabel('Genome window index (sorted by chr, start)', fontsize=label_fontsize)
    sns.despine(ax=ax2)

    # Chromosome boundaries and labels
    if add_chrom_bars:
        chrs_str = chrs.astype(str)
        b_ix, mids, labs = chromosome_boundaries(chrs_str)
        for bx in b_ix[1:-1]:
            for ax in axes:
                ax.axvline(bx, color='lightgray', linewidth=0.6, alpha=0.6)
            if ax1b is not None:
                ax1b.axvline(bx, color='lightgray', linewidth=0.6, alpha=0.6)
            if ax2b is not None:
                ax2b.axvline(bx, color='lightgray', linewidth=0.6, alpha=0.6)
        tick_ix = [mids[i] for i in range(len(mids)) if i % 2 == 0]
        tick_labs = [labs[i] for i in range(len(labs)) if i % 2 == 0]
        ax2.set_xticks(tick_ix)
        ax2.set_xticklabels(tick_labs, fontsize=tick_fontsize)

    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches='tight')
    if save_pdf and out_png.lower().endswith('.png'):
        plt.savefig(out_png[:-4] + '.pdf', bbox_inches='tight')
    plt.close()

    if legend_out:
        combined_handles = input_handles + output_handles
        combined_labels = input_labels + output_labels
        save_legend_figure(
            combined_handles, combined_labels, legend_out,
            dpi=dpi,
            font_size=legend_fontsize,
            fig_width=legend_width if legend_width is not None else fig_width,
            fig_height=legend_height,
        )

    print(f"\u2705 Saved 2-panel CA+RT (input) / Mutations (output) figure: {out_png}")
    if save_pdf:
        print(f"\u2705 Saved PDF: {out_png[:-4] + '.pdf'}")
    if legend_out:
        print(f"\u2705 Saved legend-only figure: {legend_out}")

def plot_single_panel_zscore_overlay(ca_coords, ca_signal, rt_signal, snv_signal, indel_signal,
                                     out_png, cancer_type='pancancer', smooth_ca=31, smooth_mut=31,
                                     winsor_pct=99.0, mask_zeros=False, add_chrom_bars=True,
                                     dpi=600, save_pdf=False,
                                     fig_width=14.0, fig_height=5.5,
                                     title_fontsize=14.0, label_fontsize=13.0,
                                     tick_fontsize=12.0, legend_fontsize=12.0,
                                     legend_out=None, hide_legends=False,
                                     legend_width=None, legend_height=1.4):
    """Single-panel overlay of CA, RT, SNV, INDEL after z-score normalization.
    Uses color scheme consistent with the 2-panel figure.
    """
    # Sort by genome
    combined_df = ca_coords.copy()
    combined_df['ca_signal'] = ca_signal
    combined_df['rt_signal'] = rt_signal
    combined_df['snv_signal'] = snv_signal
    combined_df['indel_signal'] = indel_signal
    combined_df = sort_by_genome(combined_df)

    y_ca = combined_df['ca_signal'].to_numpy()
    y_rt = combined_df['rt_signal'].to_numpy()
    y_snv = combined_df['snv_signal'].to_numpy()
    y_indel = combined_df['indel_signal'].to_numpy()
    chrs = combined_df['chr'].to_numpy()

    # Mask zeros for mutations if requested
    if mask_zeros:
        y_snv = y_snv.copy(); y_snv[y_snv == 0] = np.nan
        y_indel = y_indel.copy(); y_indel[y_indel == 0] = np.nan

    # Winsorize mutation extremes
    y_snv = winsorize(y_snv, upper_pct=winsor_pct)
    y_indel = winsorize(y_indel, upper_pct=winsor_pct)

    # Smoothing
    if smooth_ca and smooth_ca > 1:
        y_ca = smooth_series(y_ca, win=smooth_ca)
        y_rt = smooth_series(y_rt, win=smooth_ca)
    if smooth_mut and smooth_mut > 1:
        y_snv = smooth_series(y_snv, win=smooth_mut)
        y_indel = smooth_series(y_indel, win=smooth_mut)

    # Z-score transform all four series
    y_ca_z = apply_transform(y_ca, 'zscore')
    y_rt_z = apply_transform(y_rt, 'zscore')
    y_snv_z = apply_transform(y_snv, 'zscore')
    y_indel_z = apply_transform(y_indel, 'zscore')

    x = np.arange(1, len(y_ca_z) + 1)

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    ln_ca, = ax.plot(x, y_ca_z, color='#009E73', alpha=0.95, linewidth=1.2, label='CA (median)')
    ln_rt, = ax.plot(x, y_rt_z, color='#0072B2', alpha=0.95, linewidth=1.2, label='RT (median)')
    ln_snv, = ax.plot(x, y_snv_z, color='#D55E00', linewidth=1.2, alpha=0.95, label='SNV')
    ln_indel, = ax.plot(x, y_indel_z, color='#CC79A7', linewidth=1.2, alpha=0.95, label='INDEL')

    ax.set_ylabel('Z-score', fontsize=label_fontsize)
    ax.set_title('CA, RT, SNV and INDEL (z-score normalized)', fontsize=title_fontsize)

    # Order legend as RT, CA, SNV, INDEL to mirror earlier ordering for inputs
    handles = [ln_rt, ln_ca, ln_snv, ln_indel]
    labels = ['RT (median)', 'CA (median)', 'SNV', 'INDEL']
    if not hide_legends:
        ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1.12),
                  frameon=True, framealpha=0.85, facecolor='white', edgecolor='none', prop={'size': legend_fontsize})
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    sns.despine(ax=ax)

    # Chromosome boundaries and tick labels
    if add_chrom_bars:
        chrs_str = chrs.astype(str)
        b_ix, mids, labs = chromosome_boundaries(chrs_str)
        for bx in b_ix[1:-1]:
            ax.axvline(bx, color='lightgray', linewidth=0.6, alpha=0.6)
        tick_ix = [mids[i] for i in range(len(mids)) if i % 2 == 0]
        tick_labs = [labs[i] for i in range(len(labs)) if i % 2 == 0]
        ax.set_xticks(tick_ix)
        ax.set_xticklabels(tick_labs, fontsize=tick_fontsize)
    ax.set_xlabel('Genome window index (sorted by chr, start)', fontsize=label_fontsize)

    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches='tight')
    if save_pdf and out_png.lower().endswith('.png'):
        plt.savefig(out_png[:-4] + '.pdf', bbox_inches='tight')
    plt.close()

    if legend_out:
        save_legend_figure(
            handles, labels, legend_out,
            dpi=dpi,
            font_size=legend_fontsize,
            fig_width=legend_width if legend_width is not None else fig_width,
            fig_height=legend_height,
        )

    print(f"✅ Saved single-panel z-score overlay figure: {out_png}")
    if save_pdf:
        print(f"✅ Saved PDF: {out_png[:-4] + '.pdf'}")
    if legend_out:
        print(f"✅ Saved legend-only figure: {legend_out}")

def main():
    parser = argparse.ArgumentParser(description='Create 3-panel CA_RT + SNV + INDEL figure from new_ca_rt_mutation 1MB data')
    parser.add_argument('--ca-rt-1mb', default='tcga_atacseq/tcga_ca/tcga_ca_with_rt/tcga_atac_with_repliseq.1mb.tsv.gz',
                       help='CA+RT 1MB data file (gz)')
    parser.add_argument('--snv-1mb', default='new_ca_rt_mutation/HMF_snv_1MB.csv',
                       help='SNV 1MB data file')
    parser.add_argument('--indel-1mb', default='new_ca_rt_mutation/HMF_indel_1MB.csv',
                       help='INDEL 1MB data file')
    parser.add_argument('--out-png', default='figures/CA_mutations_1MB_ca_rt_combined.png',
                       help='Output PNG file (2-panel: CA+RT input; SNV+INDEL output)')
    parser.add_argument('--smooth-ca', type=int, default=31,
                       help='Smoothing window for CA_RT (default: 31)')
    parser.add_argument('--smooth-mut', type=int, default=31,
                       help='Smoothing window for mutations (default: 31)')
    parser.add_argument('--winsor-pct', type=float, default=99.0,
                       help='Winsorization percentile (default: 99.0)')
    parser.add_argument('--mask-zeros', action='store_true',
                       help='Mask zero mutation values as NaN')
    parser.add_argument('--dpi', type=int, default=600,
                       help='Figure DPI (default: 600)')
    parser.add_argument('--save-pdf', action='store_true',
                       help='Also save PDF version')
    parser.add_argument('--cancer-type', default='pancancer',
                       help='Cancer type to plot for mutations (default: pancancer)')
    parser.add_argument('--list-cancer-types', action='store_true',
                       help='List available cancer types and exit')

    parser.add_argument('--transform-input', choices=['none','log1p','log2','zscore','dual-axis'], default='log1p',
                        help='Transformation for top panel (CA+RT): none, log1p, log2, zscore, dual-axis')
    parser.add_argument('--transform-output', choices=['none','log1p','log2','zscore','dual-axis'], default='log1p',
                        help='Transformation for bottom panel (SNV+INDEL): none, log1p, log2, zscore, dual-axis')

    parser.add_argument('--single-panel', action='store_true',
                        help='Plot single overlay panel with z-score for CA, RT, SNV, INDEL')
    parser.add_argument('--fig-width', type=float, default=14.0,
                        help='Figure width in inches (default: 14.0)')
    parser.add_argument('--fig-height', type=float, default=9.0,
                        help='Figure height in inches for 2-panel mode (default: 9.0)')
    parser.add_argument('--single-panel-height', type=float, default=5.5,
                        help='Figure height in inches for single-panel mode (default: 5.5)')
    parser.add_argument('--title-fontsize', type=float, default=14.0,
                        help='Title font size (default: 14.0)')
    parser.add_argument('--label-fontsize', type=float, default=13.0,
                        help='Axis label font size (default: 13.0)')
    parser.add_argument('--tick-fontsize', type=float, default=12.0,
                        help='Tick label font size (default: 12.0)')
    parser.add_argument('--legend-fontsize', type=float, default=12.0,
                        help='Legend font size (default: 12.0)')
    parser.add_argument('--font-family', default=None,
                        help='Matplotlib font family to use for all text, e.g. Arial')
    parser.add_argument('--ca-count-override', type=int,
                        help='Override displayed CA count in title')
    parser.add_argument('--rt-count-override', type=int,
                        help='Override displayed RT count in title')
    parser.add_argument('--legend-out', default=None,
                        help='Optional path for a separate legend-only PNG/PDF')
    parser.add_argument('--hide-legends', action='store_true',
                        help='Omit legends from the main figure when writing a separate legend file')
    parser.add_argument('--legend-width', type=float, default=None,
                        help='Legend-only figure width in inches (default: match main figure width)')
    parser.add_argument('--legend-height', type=float, default=1.4,
                        help='Legend-only figure height in inches (default: 1.4)')

    args = parser.parse_args()

    if args.font_family:
        plt.rcParams['font.family'] = args.font_family

    # Handle list cancer types option
    if args.list_cancer_types:
        print("📋 Available cancer types:")
        df_snv = pd.read_csv(args.snv_1mb)
        cancer_cols = [col for col in df_snv.columns if col not in ['chr', 'start', 'end']]
        for i, ctype in enumerate(cancer_cols, 1):
            print(f"  {i:2d}. {ctype}")
        return

    print("🔄 Creating CA + combined mutations figure\n")

    # Load data
    ca_coords, ca_signal, rt_signal, ca_cols, rt_cols = load_tcga_ca_and_rt_from_gz(args.ca_rt_1mb)
    snv_coords, snv_signal, snv_cancer_types = load_mutation_data(args.snv_1mb, 'SNV', args.cancer_type)
    indel_coords, indel_signal, indel_cancer_types = load_mutation_data(args.indel_1mb, 'INDEL', args.cancer_type)
    display_ca_count = args.ca_count_override if args.ca_count_override is not None else len(ca_cols)
    display_rt_count = args.rt_count_override if args.rt_count_override is not None else len(rt_cols)

    # Verify coordinate alignment
    print(f"\n🔍 Verifying coordinate alignment:")
    print(f"   CA/RT rows: {len(ca_coords):,}")
    print(f"   SNV rows: {len(snv_coords):,}")
    print(f"   INDEL rows: {len(indel_coords):,}")

    # Check if coordinates match
    ca_key = ca_coords['chr'].astype(str) + ':' + ca_coords['start'].astype(str) + '-' + ca_coords['end'].astype(str)
    snv_key = snv_coords['chr'].astype(str) + ':' + snv_coords['start'].astype(str) + '-' + snv_coords['end'].astype(str)
    indel_key = indel_coords['chr'].astype(str) + ':' + indel_coords['start'].astype(str) + '-' + indel_coords['end'].astype(str)

    if not ca_key.equals(snv_key) or not ca_key.equals(indel_key):
        print("   ⚠️  Coordinates don't match exactly - will align by merge")
        # Merge all data on coordinates
        merged = ca_coords.copy()
        merged['ca_signal'] = ca_signal
        merged['rt_signal'] = rt_signal
        merged = merged.merge(snv_coords.assign(snv_signal=snv_signal), on=['chr', 'start', 'end'], how='inner')
        merged = merged.merge(indel_coords.assign(indel_signal=indel_signal), on=['chr', 'start', 'end'], how='inner')

        ca_coords = merged[['chr', 'start', 'end']].copy()
        ca_signal = merged['ca_signal']
        rt_signal = merged['rt_signal']
        snv_signal = merged['snv_signal']
        indel_signal = merged['indel_signal']
        print(f"   ✅ Aligned to {len(merged):,} common windows")
    else:
        print("   ✅ Coordinates match perfectly")

    # Create plot(s)
    if args.single_panel:
        plot_single_panel_zscore_overlay(
                            ca_coords, ca_signal, rt_signal, snv_signal, indel_signal,
                            args.out_png,
                            cancer_type=args.cancer_type,
                            smooth_ca=args.smooth_ca, smooth_mut=args.smooth_mut,
                            winsor_pct=args.winsor_pct,
                            mask_zeros=args.mask_zeros,
                            dpi=args.dpi, save_pdf=args.save_pdf,
                            fig_width=args.fig_width,
                            fig_height=args.single_panel_height,
                            title_fontsize=args.title_fontsize,
                            label_fontsize=args.label_fontsize,
                            tick_fontsize=args.tick_fontsize,
                            legend_fontsize=args.legend_fontsize,
                            legend_out=args.legend_out,
                            hide_legends=args.hide_legends,
                            legend_width=args.legend_width,
                            legend_height=args.legend_height)
    else:
        plot_ca_rt_mutations_with_rt(
                            ca_coords, ca_signal, rt_signal, snv_signal, indel_signal,
                            args.out_png,
                            cancer_type=args.cancer_type,
                            smooth_ca=args.smooth_ca, smooth_mut=args.smooth_mut,
                            winsor_pct=args.winsor_pct,
                            transform_input=args.transform_input,
                            transform_output=args.transform_output,
                            ca_count=display_ca_count, rt_count=display_rt_count,
                            mask_zeros=args.mask_zeros,
                            dpi=args.dpi, save_pdf=args.save_pdf,
                            fig_width=args.fig_width,
                            fig_height=args.fig_height,
                            title_fontsize=args.title_fontsize,
                            label_fontsize=args.label_fontsize,
                            tick_fontsize=args.tick_fontsize,
                            legend_fontsize=args.legend_fontsize,
                            legend_out=args.legend_out,
                            hide_legends=args.hide_legends,
                            legend_width=args.legend_width,
                            legend_height=args.legend_height)

    print(f"\n🎉 Complete! Figure saved to: {args.out_png}")

if __name__ == '__main__':
    main()
