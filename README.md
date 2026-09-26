# CAMM: Chromatin Accessibility to Metastatic Mutagenesis

Code, trained checkpoints, and public inputs for the manuscript *"Chromatin accessibility of primary cancers informs regional mutagenesis in metastases through multi-scale deep learning"*.

CAMM is a hierarchical, multi-scale, multi-task neural network for predicting single-nucleotide variant (SNV) and indel density at 1 Mb, 100 kb, and 10 kb resolution from chromatin accessibility (CA) and replication timing (RT). The study uses metastatic whole-genome data from the Hartwig Medical Foundation (HMF) for training and PCAWG primary tumors for external validation.

## Repository layout

```text
Code/
  step1/          # Hyperparameter search and model training
  step2/          # Cross-validation, ablations, baselines, and PCAWG validation
  step3/          # Permutation importance and SHAP attribution
  step4/          # Mutation-enriched windows and gene annotation
Data/
  CA_RT/          # Chromatin accessibility and replication timing features
  PCAWG/          # Public validation mutation-count tables
Model/            # Six cancer-specific PyTorch checkpoints
Figure_script/    # Python and R scripts for Figures 1–4
docs/
  parameters.md   # Required inputs, optional parameters, and implementation notes
requirements.txt  # Python dependency inventory
```

## Installation

Use Python 3.11 for the following setup. The shell commands use macOS/Linux syntax. Training uses CUDA when available and otherwise runs on CPU.

```bash
git clone --depth 1 https://github.com/reimandlab/CAMM.git
cd CAMM
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt "pandas==2.3.3"
```

For R figure scripts, install:

```r
install.packages(c("ggplot2", "dplyr", "readr", "tidyr", "ggnewscale",
                   "cowplot", "reshape2", "patchwork", "reticulate", "fs"))
```

## Data availability

| Location | Contents |
|---|---|
| [Data/CA_RT/](Data/CA_RT/) | CA/RT features at three resolutions: 796 TCGA ATAC-seq profiles and 96 ENCODE Repli-seq profiles |
| [Data/PCAWG/](Data/PCAWG/) | PCAWG SNV and indel validation count tables at three resolutions |

HMF whole-genome data and metastatic sample metadata are controlled access. Requests are submitted through the [HMF data-access procedure](https://www.hartwigmedicalfoundation.nl/data/aanvragen-data/) and require approval and the applicable data access or material transfer agreements. HMF-derived intermediate files are not publicly shared because of data-use restrictions.

Feature tables are tab-separated, optionally gzip-compressed; mutation tables are comma-separated. Both use `chr` and `start` coordinates. Mutation targets are selected by cancer-type column. Input-column handling and the current limitations of cross-scale alignment are described in the [implementation notes](docs/parameters.md#implementation-notes).

Reconstruct the chromosome-split 10 kb feature matrix before using the full data:

```bash
python Data/CA_RT/atac_with_repliseq_10kb/combine_chr_tsv.py Data/CA_RT/atac_with_repliseq_10kb --output Data/CA_RT/atac_with_repliseq.10kb.tsv.gz
```

Some wrappers expect feature filenames beginning with `tcga_atac_with_repliseq`, whereas the bundled coarse files and the reconstructed file above begin with `atac_with_repliseq`. The [parameter reference](docs/parameters.md) lists the filenames expected by each wrapper.

## Trained checkpoints

[Model/](Model/) contains PyTorch checkpoints for breast, colorectal, esophagus, lung, prostate, and skin cancers. Scripts with `--model_dir`, `--best_dir`, or `--model_outdir` look for `best_model.pt` in the supplied directory.

Checkpoint use requires the matching model architecture, feature order, and preprocessing. Per-checkpoint training and preprocessing configurations are not bundled. The current PCAWG validator also differs from the checkpoint architecture; see the [checkpoint compatibility note](docs/parameters.md#pcawg-validation) before using it.

## Analysis and figure workflow

Run scripts from the repository root. Main training requires feature, SNV, and indel paths at all three resolutions, `--ctype` to select the cancer column, and `--outdir` for outputs. The [parameter reference](docs/parameters.md) documents required inputs, optional parameters, defaults, and accepted values for the Python analysis scripts.

Inspect command-line help after installation:

```bash
python Code/step1/run_model_hier_multi.py --help
python -m Code.step3.feature_importance --help
```

The feature-importance script uses module invocation to resolve its model import.

| Stage | Scripts |
|---|---|
| Tune and train | [Optuna search](Code/step1/optuna_hier_multi_tcga_rt.py); [hierarchical multi-task model](Code/step1/run_model_hier_multi.py) |
| Cross-validation | [Full model](Code/step2/run_model_hier_multi_cv.py); [single-task ablation](Code/step2/run_model_hier_single_task_cv.py); [single-scale ablation](Code/step2/run_model_multitask_single_scale_cv.py) |
| Paired comparisons | [Multi-task vs single-task](Code/step2/cv_compare_mtl_ms_vs_ms_st.py); [multi-scale vs single-scale](Code/step2/cv_compare_mtl_ms_vs_ss.py) |
| Random Forest / XGBoost baselines | [Independent task/scale models](Code/step2/run_baseline_tree_models_randomsplit_independent.py) |
| PCAWG validation | [Frozen-model evaluation and calibration](Code/step2/validate_pcawg_kfold_linear_calib.py) |
| Residual analysis | [10 kb SNV predictions](Code/step2/eval_best_model_snv10_all.py); [10 kb indel predictions](Code/step2/eval_best_model_indel10_all.py) |
| Feature importance | [Permutation importance and SHAP](Code/step3/feature_importance.py) |
| Gene annotation | [Mutation-enriched windows and cancer-gene annotation](Code/step4/underestimated_windows.py) |

| Figure | Scripts in `Figure_script/` | Inputs |
|---|---|---|
| 1 | [ATAC-seq heatmaps](Figure_script/plot_fig1_matched_atacseq_heatmap.py); [Repli-seq heatmaps](Figure_script/plot_fig1_matched_repliseq_heatmaps.py) | CA/RT features and HMF mutation tables |
| 2 | [Model comparison](Figure_script/plot_fig2_mtms_dots_bar.R); [legend](Figure_script/plot_fig2_legend_only.R) | Model comparison and baseline summary tables; legend is self-contained |
| 3 | [Feature correlations](Figure_script/plot_fig3_bar_pie.py); [SHAP plots](Figure_script/plot_fig3_all_shap.R) | SHAP tables and selected-feature lists |
| 4 | [Residual distributions](Figure_script/plot_fig4_residual_violin_outliers.py); [per-cancer residual z-scores](Figure_script/plot_fig4_bar_stacked_per_cancer_z.py); [gene windows](Figure_script/plot_fig4_gene_windows.py); [z-scores](Figure_script/plot_fig4_zscores.R) | Residual, gene annotation, and enrichment summary tables |

Run Python figure scripts with `python Figure_script/<filename>.py` and R scripts with `Rscript Figure_script/<filename>.R`, after preparing their required inputs. Some scripts use historical paths, including lowercase `data/`; set the documented path options or edit local path settings to match your files.


## Support

Questions and reproducibility issues can be submitted through [GitHub Issues](https://github.com/reimandlab/CAMM/issues).
