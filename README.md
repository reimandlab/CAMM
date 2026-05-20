# CAMM: Chromatin Accessibility to Metastatic Mutagenesis

Code for the manuscript *"Chromatin accessibility of primary cancers informs regional mutagenesis in metastases through multi-scale deep learning"*.

A hierarchical, multi-scale, multi-task neural network that jointly predicts SNV and indel density at 1 Mb, 100 kb, and 10 kb resolution from chromatin accessibility (CA) and replication timing (RT) profiles, trained on metastatic whole-genome data (HMF) and externally validated on primary tumors (PCAWG).

## Data

- **Mutations**: HMF metastatic WGS (6 cancer types: breast, colorectal, prostate, lung, esophagus, skin) and PCAWG primary tumors (validation), GRCh37/hg19, autosomes only.
- **Epigenomes**: 796 TCGA ATAC-seq CA profiles + 96 ENCODE Repli-seq RT profiles.
- **Windows**: non-overlapping 10 kb / 100 kb / 1 Mb after mappability and blacklist filtering.

## Repository layout

```
Code/
  step1/   # Hyperparameter search + main hierarchical MS-MT model
  step2/   # Cross-validation, ablations, baselines, PCAWG validation
  step3/   # Feature importance (permutation + SHAP)
  step4/   # Mutation-enriched windows and cancer-gene annotation
Figure_script/   # Python/R scripts for Figures 1–4
```

### `Code/step1` — model training
- [run_model_hier_multi.py](Code/step1/run_model_hier_multi.py): hierarchical multi-scale (1 Mb → 100 kb → 10 kb), multi-task (SNV + indel) MLP with adaptive feature gating, coarse-to-fine context flow, and uncertainty-weighted loss.
- [optuna_hier_multi_tcga_rt.py](Code/step1/optuna_hier_multi_tcga_rt.py): Optuna hyperparameter search (learning rate, batch size, hidden dim, dropout).

### `Code/step2` — evaluation and baselines
- [run_model_hier_multi_cv.py](Code/step2/run_model_hier_multi_cv.py): repeated K-fold CV for the full MS-MT model.
- [run_model_hier_single_task_cv.py](Code/step2/run_model_hier_single_task_cv.py), [run_model_multitask_single_scale_cv.py](Code/step2/run_model_multitask_single_scale_cv.py): single-task and single-scale ablations.
- [cv_compare_mtl_ms_vs_ms_st.py](Code/step2/cv_compare_mtl_ms_vs_ms_st.py), [cv_compare_mtl_ms_vs_ss.py](Code/step2/cv_compare_mtl_ms_vs_ss.py): paired comparisons (MS-MT vs MS-ST and vs MT-SS).
- [run_baseline_tree_models_randomsplit_independent.py](Code/step2/run_baseline_tree_models_randomsplit_independent.py): Random Forest / Elastic Net baselines, independent per (task, scale).
- [eval_best_model_snv10_all.py](Code/step2/eval_best_model_snv10_all.py), [eval_best_model_indel10_all.py](Code/step2/eval_best_model_indel10_all.py): genome-wide 10 kb predictions and residual outlier (z > 3 / 4) tables.
- [validate_pcawg_kfold_linear_calib.py](Code/step2/validate_pcawg_kfold_linear_calib.py): external validation on PCAWG with linear / per-chromosome calibration.

### `Code/step3` — interpretation
- [feature_importance.py](Code/step3/feature_importance.py): permutation importance (1,000 permutations, empirical p-values) and SHAP attributions (Captum `ShapleyValueSampling`) for 10 kb SNV predictions.

### `Code/step4` — mutation-enriched windows
- [underestimated_windows.py](Code/step4/underestimated_windows.py): build combined tables of underestimated 10 kb windows (z > 4 baseline and optional Tukey-thresholded inputs), add window-end coordinates, intersect with `hg19_genes_gff.bed`, flag OncoKB / CGC cancer genes, and emit per-prefix step1–step6 tables plus downstream-compatible aliases (`*_cancer_genes_only.tsv`, `*_cancer_types_by_gene.tsv`).

### `Figure_script/`
Plotting scripts grouped by figure:
- **Fig. 1** — CA / RT vs mutation density heatmaps ([plot_fig1_matched_atacseq_heatmap.py](Figure_script/plot_fig1_matched_atacseq_heatmap.py), [plot_fig1_matched_repliseq_heatmaps.py](Figure_script/plot_fig1_matched_repliseq_heatmaps.py)).
- **Fig. 2** — MS-MT vs ablations / baselines and PCAWG transfer ([plot_fig2_mtms_dots_bar.R](Figure_script/plot_fig2_mtms_dots_bar.R), [plot_fig2_legend_only.R](Figure_script/plot_fig2_legend_only.R)).
- **Fig. 3** — SHAP-based feature importance bars and pies ([plot_fig3_bar_pie.py](Figure_script/plot_fig3_bar_pie.py), [plot_fig3_all_shap.R](Figure_script/plot_fig3_all_shap.R)).
- **Fig. 4** — Residual analysis and mutation-enriched windows ([plot_fig4_residual_violin_outliers.py](Figure_script/plot_fig4_residual_violin_outliers.py), [plot_fig4_bar_stacked_per_cancer_z.py](Figure_script/plot_fig4_bar_stacked_per_cancer_z.py), [plot_fig4_gene_windows.py](Figure_script/plot_fig4_gene_windows.py), [plot_fig4_zscores.R](Figure_script/plot_fig4_zscores.R)).

## Pipeline

1. **Tune** — `step1/optuna_hier_multi_tcga_rt.py` per cancer type.
2. **Train** — `step1/run_model_hier_multi.py` with the best config.
3. **Evaluate** — `step2/` for CV, ablations, tree/linear baselines, residual extraction, and PCAWG transfer.
4. **Interpret** — `step3/feature_importance.py` for permutation + SHAP.
5. **Annotate** — `step4/underestimated_windows.py` to map mutation-enriched windows to genes and OncoKB / CGC cancer genes.

## Requirements

Python 3.9+ with `torch`, `numpy`, `pandas`, `scikit-learn`, `optuna`, `captum`, `shap`; R with `tidyverse` / `ggplot2` for the R figure scripts.
