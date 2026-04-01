#!/usr/bin/env Rscript
# Legend-only PDF for Fig2 model styles.

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggnewscale)
  library(cowplot)
})

get_script_dir <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- cmd_args[grepl("^--file=", cmd_args)]
  if (length(file_arg) > 0) {
    return(normalizePath(dirname(sub("^--file=", "", file_arg[1]))))
  }
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(normalizePath(dirname(sys.frames()[[1]]$ofile)))
  }
  normalizePath(".")
}

SCRIPT_DIR <- get_script_dir()
OUT_DIR    <- normalizePath(
  Sys.getenv("FIG_OUT_DIR", unset = file.path(SCRIPT_DIR, "..")),
  mustWork = FALSE
)
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

get_env_num <- function(name, default) {
  value <- suppressWarnings(as.numeric(Sys.getenv(name, unset = as.character(default))))
  if (is.finite(value)) value else default
}

BASE_FAMILY <- Sys.getenv("FIG_FONT_FAMILY", unset = "sans")
BASE_SIZE   <- get_env_num("FIG_BASE_SIZE", 15)
OUTPUT_WIDTH <- get_env_num("FIG_WIDTH", 3.2)
OUTPUT_HEIGHT <- get_env_num("FIG_HEIGHT", 3.2)

MODEL_LEVELS <- c("MTL-MS", "MTL-SS", "MS-ST")
MODEL_LABELS <- c(
  "MTL-MS" = "MS-MT (Full model)",
  "MS-ST" = "MS-ST (Single-task baseline)",
  "MTL-SS" = "MT-SS (Single-scale baseline)"
)
MODEL_FILL <- c(
  "MTL-MS" = "#2166AC",
  "MS-ST" = "#D9D9D9",
  "MTL-SS" = "#9E9E9E"
)
MODEL_OUTLINE <- c(
  "MTL-MS" = "#0B3D6D",
  "MS-ST" = "#7A7A7A",
  "MTL-SS" = "#4D4D4D"
)
MODEL_LINEWIDTH <- c(
  "MTL-MS" = 1.1,
  "MS-ST" = 0.7,
  "MTL-SS" = 0.7
)

CANCER_LEVELS <- c("breast", "colorectal", "esophagus", "lung", "prostate", "skin")
CANCER_COLORS <- c(
  breast = "#E75480",
  colorectal = "#1F78B4",
  esophagus = "#FF8C00",
  lung = "#33A02C",
  prostate = "#6A3D9A",
  skin = "#FFD92F"
)

legend_df <- data.frame(
  scale = factor("100 kbps", levels = "100 kbps"),
  model = factor(MODEL_LEVELS, levels = MODEL_LEVELS),
  R2 = c(0.8, 0.7, 0.6)
)

model_plot <- ggplot(legend_df, aes(x = scale, y = R2, fill = model)) +
  geom_boxplot(aes(color = model, linewidth = model),
               width = 0.6, position = position_dodge(width = 0.75),
               outlier.shape = NA, alpha = 0.7) +
  scale_fill_manual(values = MODEL_FILL, labels = MODEL_LABELS) +
  scale_color_manual(values = MODEL_OUTLINE) +
  scale_linewidth_manual(values = MODEL_LINEWIDTH) +
  guides(
    fill = guide_legend(
      title = NULL,
      ncol = 1,
      byrow = TRUE,
      override.aes = list(
        linewidth = unname(MODEL_LINEWIDTH[MODEL_LEVELS]),
        color = unname(MODEL_OUTLINE[MODEL_LEVELS])
      )
    ),
    color = "none",
    linewidth = "none"
  ) +
  theme_void(base_size = BASE_SIZE, base_family = BASE_FAMILY) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = BASE_SIZE)
  )

cancer_df <- data.frame(
  scale = factor("100 kbps", levels = "100 kbps"),
  cancer_type = factor(CANCER_LEVELS, levels = CANCER_LEVELS),
  R2 = seq(0.6, 0.6 + 0.02 * (length(CANCER_LEVELS) - 1), by = 0.02)
)

cancer_plot <- ggplot(cancer_df, aes(x = scale, y = R2, color = cancer_type)) +
  geom_point(size = 2.8 * BASE_SIZE / 15) +
  scale_color_manual(values = CANCER_COLORS, breaks = CANCER_LEVELS) +
  guides(
    color = guide_legend(title = "Cancer type", ncol = 1, byrow = TRUE)
  ) +
  theme_void(base_size = BASE_SIZE, base_family = BASE_FAMILY) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = BASE_SIZE),
    legend.title = element_text(size = BASE_SIZE)
  )

model_legend <- cowplot::get_legend(model_plot)
cancer_legend <- cowplot::get_legend(cancer_plot)

legend_only <- cowplot::plot_grid(
  model_legend,
  cancer_legend,
  ncol = 1,
  rel_heights = c(1, 1.6)
)

ggsave(file.path(OUT_DIR, "Fig2_legend_only.pdf"),
       legend_only, width = OUTPUT_WIDTH, height = OUTPUT_HEIGHT,
       device = grDevices::cairo_pdf)
