#!/usr/bin/env Rscript
# SHAP beeswarm plots for all cancer types under feature_importance_v2.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(reticulate)
  library(fs)
})

`%||%` <- function(a, b) if (!is.null(a)) a else b

get_arg_value <- function(args, flag, default = NULL) {
  prefix <- paste0(flag, "=")
  match <- args[grepl(paste0("^", flag, "="), args)]
  if (length(match) > 0) {
    return(sub(prefix, "", match[1]))
  }
  default
}

get_arg_flag <- function(args, flag, default = FALSE) {
  if (flag %in% args) {
    return(TRUE)
  }
  value <- get_arg_value(args, flag, NULL)
  if (is.null(value)) {
    return(default)
  }
  tolower(value) %in% c("1", "true", "yes")
}

matched_codes <- list(
  breast = c("BRCA"),
  colorectal = c("COAD", "READ"),
  esophagus = c("ESCA"),
  lung = c("LUAD", "LUSC"),
  prostate = c("PRAD"),
  skin = c("SKCM")
)
matched_label_color <- "#C62026"

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
script_path <- if (length(file_arg) > 0) file_arg[1] else sys.frames()[[1]]$ofile %||% "."
root <- normalizePath(file.path(dirname(script_path), ".."))
args_trailing <- commandArgs(trailingOnly = TRUE)
tag <- "top5pct"
font_family <- get_arg_value(args_trailing, "--font-family", "Arial")
font_size <- suppressWarnings(as.numeric(get_arg_value(args_trailing, "--font-size", "21")))
if (!is.finite(font_size)) font_size <- 21
vertical <- get_arg_flag(args_trailing, "--vertical", FALSE)
if (length(args_trailing) > 0) {
  for (a in args_trailing) {
    if (grepl("^--tag=", a)) {
      tag <- sub("^--tag=", "", a)
    } else if (!grepl("^--", a)) {
      tag <- a
    }
  }
}

preferred_python <- Sys.getenv("RETICULATE_PYTHON", unset = "")
if (!nzchar(preferred_python)) {
  venv_python <- file.path(root, "..", ".venv_shap_plot", "bin", "python")
  preferred_python <- normalizePath(venv_python, mustWork = FALSE)
  if (!file.exists(preferred_python)) {
    preferred_python <- ""
  }
}
if (nzchar(preferred_python)) {
  reticulate::use_python(preferred_python, required = TRUE)
}

feature_dir <- root
out_dir <- file.path(root, "plots", paste0("feature_importance_", tag))
beeswarm_dir <- file.path(out_dir, "beeswarm")
dir.create(beeswarm_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cts <- dir_ls(feature_dir, type = "directory", recurse = FALSE) %>%
  path_file() %>%
  setdiff(c("plots", "scripts"))

if (!py_module_available("shap")) stop("Python module 'shap' not found.")
if (!py_module_available("matplotlib")) stop("Python module 'matplotlib' not found.")
s <- import("shap")
mpl <- import("matplotlib")
plt <- import("matplotlib.pyplot")
np <- import("numpy")

mpl$rcParams$update(dict(
  `font.family` = "sans-serif",
  `font.sans-serif` = tuple(font_family, "Arial", "Helvetica", "DejaVu Sans"),
  `font.size` = font_size,
  `axes.labelsize` = font_size,
  `xtick.labelsize` = font_size,
  `ytick.labelsize` = font_size,
  `axes.titlesize` = font_size,
  `figure.titlesize` = font_size,
  `legend.fontsize` = font_size,
  `figure.dpi` = 300,
  `savefig.dpi` = 300
))

rotate_beeswarm_vertical <- function(ax, fig, font_size, ct_codes, matched_label_color, np) {
  fig$canvas$draw()

  x_ticks <- py_to_r(ax$get_xticks())
  x_tick_objs <- ax$get_xticklabels()
  x_tick_labels <- vapply(x_tick_objs, function(tick) as.character(tick$get_text()), character(1))
  y_ticks <- py_to_r(ax$get_yticks())
  y_tick_objs <- ax$get_yticklabels()
  y_tick_labels <- vapply(y_tick_objs, function(tick) as.character(tick$get_text()), character(1))
  matched_mask <- toupper(trimws(y_tick_labels)) %in% ct_codes

  for (coll in reticulate::iterate(ax$collections)) {
    offsets <- py_to_r(coll$get_offsets())
    if (is.null(dim(offsets)) || nrow(offsets) == 0 || ncol(offsets) < 2) next
    coll$set_offsets(np$column_stack(tuple(offsets[, 2], offsets[, 1])))
  }

  for (line in reticulate::iterate(ax$lines)) {
    line$set_data(line$get_ydata(), line$get_xdata())
  }

  ax$set_xlim(min(y_ticks) - 0.7, max(y_ticks) + 0.7)
  ax$set_xticks(y_ticks)
  ax$set_xticklabels(y_tick_labels, rotation = 90)
  ax$set_ylim(min(x_ticks), max(x_ticks))
  ax$set_yticks(x_ticks)
  ax$set_yticklabels(x_tick_labels)
  ax$invert_xaxis()
  ax$set_xlabel("")
  ax$set_ylabel("SHAP value", fontsize = font_size)
  ax$axhline(
    y = 0,
    linestyle = "--",
    linewidth = max(1, font_size / 14),
    color = "#7F7F7F",
    alpha = 0.9,
    zorder = 0
  )
  ax$tick_params(axis = "x", labelsize = font_size)
  ax$tick_params(axis = "y", labelsize = font_size)

  xticklabels <- ax$get_xticklabels()
  for (i in seq_along(xticklabels)) {
    tick <- xticklabels[[i]]
    if (isTRUE(matched_mask[i])) {
      tick$set_color(matched_label_color)
      tick$set_fontweight("bold")
    }
    tick$set_fontsize(font_size)
  }
}

for (ct in cts) {
  top_path_legacy <- file.path(feature_dir, ct, paste0(tag, "_significant_features.tsv"))
  top_path_v2 <- file.path(feature_dir, ct, paste0(tag, "_shap_features_in_permutation_significant_10kb.tsv"))
  top_path <- if (file.exists(top_path_legacy)) top_path_legacy else top_path_v2

  shap_path_gz <- file.path(feature_dir, ct, "shap_beeswarm_10kb.tsv.gz")
  shap_path <- if (file.exists(shap_path_gz)) shap_path_gz else file.path(feature_dir, ct, "shap_beeswarm_10kb.tsv")
  if (!file.exists(top_path) || !file.exists(shap_path)) {
    message("Skipping ", ct, ": missing ", tag, " or beeswarm file")
    next
  }

  top_df <- read_tsv(top_path, show_col_types = FALSE)
  if (!all(c("feature", "mean_abs_attr") %in% names(top_df))) {
    message("Skipping ", ct, ": ", tag, " file missing required columns feature/mean_abs_attr")
    next
  }
  shap_long <- read_tsv(shap_path, show_col_types = FALSE)
  if (!all(c("row", "feature", "attr", "value") %in% names(shap_long))) {
    message("Skipping ", ct, ": beeswarm file missing row/feature/attr/value")
    next
  }

  feat_order <- top_df %>% arrange(desc(mean_abs_attr)) %>% pull(feature)
  shap_filt <- shap_long %>% filter(feature %in% feat_order) %>% mutate(row = as.integer(row))
  if (nrow(shap_filt) == 0) {
    message("Skipping ", ct, ": no overlap between ", tag, " and beeswarm features")
    next
  }
  label_map <- setNames(substr(feat_order, 1, 4) |> toupper(), feat_order)
  feat_labels <- label_map[feat_order]

  shap_wide <- shap_filt %>%
    select(row, feature, attr) %>%
    pivot_wider(names_from = feature, values_from = attr)
  vals_wide <- shap_filt %>%
    select(row, feature, value) %>%
    pivot_wider(names_from = feature, values_from = value)

  shap_mat <- shap_wide %>% select(all_of(feat_order)) %>% as.matrix()
  vals_mat <- vals_wide %>% select(all_of(feat_order)) %>% as.matrix()
  shap_mat[is.na(shap_mat)] <- 0
  vals_mat[is.na(vals_mat)] <- 0
  vals_mat <- scale(vals_mat, center = TRUE, scale = TRUE)

  expl <- s$Explanation(
    values = shap_mat,
    base_values = 0.0,
    data = vals_mat,
    feature_names = feat_labels
  )

  s$plots$beeswarm(
    expl,
    max_display = length(feat_order),
    show = FALSE
  )

  ax <- plt$gca()
  fig <- plt$gcf()
  ct_codes <- matched_codes[[ct]] %||% toupper(ct)
  if (isTRUE(vertical)) {
    fig$set_size_inches(max(10, 0.42 * length(feat_order)), 8.5)
    rotate_beeswarm_vertical(ax, fig, font_size, ct_codes, matched_label_color, np)
  } else {
    fig$set_size_inches(10, max(6.5, 0.42 * length(feat_order)))
    ax$set_xlabel("SHAP value", fontsize = font_size)
    ax$tick_params(axis = "x", labelsize = font_size)
    ax$tick_params(axis = "y", labelsize = font_size)
    yticklabels <- ax$get_yticklabels()
    for (tick in yticklabels) {
      tick_text <- toupper(trimws(tick$get_text()))
      if (tick_text %in% ct_codes) {
        tick$set_color(matched_label_color)
        tick$set_fontweight("bold")
      }
      tick$set_fontsize(font_size)
    }
  }
  if (length(fig$axes) > 1) {
    cbar_ax <- fig$axes[[length(fig$axes)]]
    cbar_ax$tick_params(labelsize = font_size)
    cbar_ax$set_ylabel("Feature value", fontsize = font_size)
  }

  out_path <- file.path(beeswarm_dir, paste0(ct, "_", tag, "_shap_beeswarm.png"))
  plt$savefig(out_path, bbox_inches = "tight", dpi = 300)
  plt$close()
  message("Saved ", out_path)
}
