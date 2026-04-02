#!/usr/bin/env Rscript
# Bar plots for MT-MS per-cancer medians from mtl_vs_msst_dot_values_repeat1_medians.csv

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
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
BASE_DIR   <- normalizePath(file.path(SCRIPT_DIR, ".."))
DATA_DIR   <- file.path(BASE_DIR, "data")
OUT_DIR    <- normalizePath(
  Sys.getenv("FIG_OUT_DIR", unset = BASE_DIR),
  mustWork = FALSE
)
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

get_env_num <- function(name, default) {
  value <- suppressWarnings(as.numeric(Sys.getenv(name, unset = as.character(default))))
  if (is.finite(value)) value else default
}

BASE_FAMILY <- Sys.getenv("FIG_FONT_FAMILY", unset = "sans")
BASE_SIZE   <- get_env_num("FIG_BASE_SIZE", 12)
OUTPUT_WIDTH <- get_env_num("FIG_WIDTH", 10)
OUTPUT_HEIGHT <- get_env_num("FIG_HEIGHT", 7)

DOT_FILE <- file.path(DATA_DIR, "mtl_vs_msst_dot_values_repeat1_medians.csv")
BASELINE_FILE <- file.path(DATA_DIR, "RF_baseline_R__summary.csv")
if (!file.exists(DOT_FILE)) stop("Missing dot-value CSV: ", DOT_FILE)
if (!file.exists(BASELINE_FILE)) stop("Missing RF baseline CSV: ", BASELINE_FILE)

load_mtms_dots <- function(dot_path, baseline_path) {
  cancer_levels <- c("breast","lung","colorectal","esophagus","prostate","skin")
  mtms <- read_csv(dot_path, show_col_types = FALSE) %>%
    filter(model == "MTL-MS") %>%
    mutate(
      mutation = toupper(mutation),
      cancer_type = factor(cancer_type, levels = cancer_levels),
      scale = factor(dplyr::recode(scale, "1MB" = "1 Mbps", "100KB" = "100 kbps", "10KB" = "10 kbps", .default = scale),
                     levels = c("1 Mbps","100 kbps","10 kbps")),
      R2 = R2_median_repeat1,
      model = factor("MTL-MS", levels = c("MTL-MS","RF"))
    ) %>%
    select(mutation, cancer_type, scale, R2, model)

  rf_raw <- read_csv(
    baseline_path,
    show_col_types = FALSE,
    col_names = c("idx","ctype","task","scale","r2_rf"),
    skip = 1
  ) %>%
    select(-idx)
  rf <- rf_raw %>%
    mutate(
      mutation = toupper(task),
      cancer_type = factor(ctype, levels = cancer_levels),
      scale = factor(case_when(
        scale == "1mb" ~ "1 Mbps",
        scale == "100kb" ~ "100 kbps",
        scale == "10kb" ~ "10 kbps",
        TRUE ~ toupper(scale)
      ), levels = c("1 Mbps","100 kbps","10 kbps")),
      R2 = as.numeric(r2_rf),
      model = factor("RF", levels = c("MTL-MS","RF"))
    ) %>%
    select(mutation, cancer_type, scale, R2, model)

  bind_rows(mtms, rf)
}

make_plot <- function(df, mut, out_pdf, out_png) {
  sub <- df %>% filter(mutation == !!mut)
  if (nrow(sub) == 0) {
    warning("No MTL-MS dot rows for ", mut)
    return()
  }
  sub <- sub %>% mutate(model = factor(model, levels = c("MTL-MS","RF")))
  y_limits <- c(0, 1.0)
  y_breaks <- seq(0, 1.0, by = 0.1)
  pos <- position_dodge(width = 0.7)
  p <- ggplot(sub, aes(x = scale, y = R2, fill = model)) +
    geom_col(width = 0.6, alpha = 0.95, position = pos) +
    facet_wrap(~ cancer_type, ncol = 3) +
    scale_fill_manual(values = c("MTL-MS" = "#377eb8", "RF" = "#b0b0b0"),
                      name = "Model",
                      labels = c("MTL-MS" = "MT-MS MLP", "RF" = "Random Forest")) +
    scale_y_continuous(breaks = y_breaks, expand = expansion(mult = c(0, 0.02))) +
    coord_cartesian(ylim = y_limits) +
    labs(
      title = paste0(mut, " R\u00B2: MT-MS MLP vs. Random Forest Baseline"),
      x = "Scale",
      y = "R\u00B2"
    ) +
    theme_minimal(base_size = BASE_SIZE, base_family = BASE_FAMILY) +
    theme(
      plot.title = element_text(size = BASE_SIZE + 2, face = "bold"),
      strip.text = element_text(size = BASE_SIZE, face = "bold"),
      legend.position = "bottom",
      panel.grid.minor = element_blank()
    )
  ggsave(out_pdf, p, width = OUTPUT_WIDTH, height = OUTPUT_HEIGHT,
         device = grDevices::cairo_pdf)
  ggsave(out_png, p, width = OUTPUT_WIDTH, height = OUTPUT_HEIGHT, dpi = 300)
}

main <- function() {
  dots <- load_mtms_dots(DOT_FILE, BASELINE_FILE)
  make_plot(dots, "SNV",
            file.path(OUT_DIR, "mtl_ms_barplot_dots_snv.pdf"),
            file.path(OUT_DIR, "mtl_ms_barplot_dots_snv.png"))
  make_plot(dots, "INDEL",
            file.path(OUT_DIR, "mtl_ms_barplot_dots_indel.pdf"),
            file.path(OUT_DIR, "mtl_ms_barplot_dots_indel.png"))
}

if (identical(environment(), globalenv())) {
  try(main())
}
