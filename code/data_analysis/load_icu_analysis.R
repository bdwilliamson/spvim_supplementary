#!/usr/local/bin/Rscript
## load results of the ICU data analysis

## ---------------------------------------------------
## load required libraries, set up arguments
## ---------------------------------------------------
library("argparse")
library("dplyr")
library("tidyr")
library("ggplot2")
library("cowplot")
theme_set(theme_cowplot())
library("readr")

code_dir <- "code/"
output_dir <- "sims/results/"
data_dir <- "icu_data/"
plots_dir <- "sims/_plots/"
source(paste0(code_dir, "map_icu_features_to_names.R"))

parser <- ArgumentParser()
parser$add_argument("--vimp-measure", default = c("auc"),
                    help = "variable importance measures to use")
parser$add_argument("--estimator-type", default = c("tree", "nn"),
                    help = "estimator types to use")
args <- parser$parse_args()

## feature names
dataset_nms <- get_feature_names(all_summaries = FALSE, pre_filter_missing = TRUE)
dataset_nms$feature_nms
dataset_nms$outcome_nm

## dataset
icu_data <- list()
icu_data$x_train <- read_csv(paste0(data_dir, "icu_data_processed_xtrain.csv"), col_names = c("Row index", dataset_nms$feature_nms), skip = 1)
icu_data$y_train <- read_csv(paste0(data_dir, "icu_data_processed_ytrain.csv"), col_names = c("Row index", dataset_nms$outcome_nm), skip = 1)
icu_data$x_test <- read_csv(paste0(data_dir, "icu_data_processed_xtest.csv"), col_names = c("Row index", dataset_nms$feature_nms), skip = 1)
icu_data$y_test <- read_csv(paste0(data_dir, "icu_data_processed_ytest.csv"), col_names = c("Row index", dataset_nms$outcome_nm), skip = 1)

## descriptive statistics (missing data)
icu_data$x_train %>%
    mutate_all(.funs = is.na) %>%
    summarize_all(.funs = mean) %>%
    print(width = Inf)

## ---------------------------------------------------
## load results from the ICU analysis
## ---------------------------------------------------
for (i in 1:length(args$vimp_measure)) {
  for (j in 1:length(args$estimator_type)) {
    eval(parse(text = paste0("icu_results_", args$vimp_measure[i], "_", args$estimator_type[j], " <- read_csv(paste0(output_dir, 'icu_data_analysis_measure_', args$vimp_measure[i], '_est_', args$estimator_type[j], '.csv')) %>% select(-X1) %>% mutate(est_type = as.character(args$estimator_type[j]))")))
    eval(parse(text = paste0("icu_lime_", args$vimp_measure[i], "_", args$estimator_type[j], " <- read_csv(paste0(output_dir, 'icu_lime_ests_measure_', args$vimp_measure[i], '_est_', args$estimator_type[j], '.csv')) %>% select(-X1) %>% mutate(est_type = as.character(args$estimator_type[j]))")))
    eval(parse(text = paste0("icu_vimp_", args$vimp_measure[i], "_", args$estimator_type[j], " <- read_csv(paste0(output_dir, 'icu_vim_ests_measure_', args$vimp_measure[i], '_est_', args$estimator_type[j], '.csv')) %>% select(-X1) %>% mutate(est_type = as.character(args$estimator_type[j]))")))
  }
}
## ---------------------------------------------------
## create plots of VIMs and SHAP values
## ---------------------------------------------------
make_xlab <- function(vimp_measure, measure_type) {
    param <- ifelse(measure_type == "spvim", "SPVIM", "Conditional VIM")
    if (vimp_measure == "r_squared") {
        return(bquote(.(paste0(param, " estimates:"))~R^2))
    } else if (vimp_measure == "auc") {
        return(bquote(.(paste0(param, " estimates:"))~AUC))
    } else {

    }
}
make_shap_xlab <- function(vimp_measure) {
    if (vimp_measure == "r_squared") {
        return(bquote("Mean absolute SHAP values: "~"MSE"))
    } else if (vimp_measure == "auc") {
      return(bquote("Mean absolute SHAP values"))
    } else {

    }
}

fig_width <- 50
fig_height <- 50
point_size <- 3
axis_font_size <- 18
main_font_size <- 20
lgnd_pos <- c(0, 0)
num_features <- 20
for (i in 1:length(args$vimp_measure)) {
    ## create individual plots of each VIM
    eval(parse(text = paste0("current_vimp_results_tree <- icu_results_all_", args$vimp_measure[i], "_tree %>% tibble::add_column(estimator_type = 'Boosted trees')")))
    if (length(args$estimator_type) == 2) {
      eval(parse(text = paste0("current_vimp_results_nn <- icu_results_all_", args$vimp_measure[i], "_nn %>% tibble::add_column(estimator_type = 'Neural networks')")))
      current_vimp_results <- dplyr::bind_rows(current_vimp_results_tree, current_vimp_results_nn)
    } else {
      current_vimp_results <- current_vimp_results_tree
    }
    ## create groups ordered by the estimate, and truncate estimates and CIs at zero
    current_vimp_results <- current_vimp_results %>%
      mutate(feature_nm = rep(dataset_nms$feature_nms,
                              length(unique(current_vimp_results$measure))*length(unique(current_vimp_results$estimator_type))))
    spvim_results <- current_vimp_results %>%
      filter(measure == "auc") %>%
      group_by(estimator_type) %>%
      mutate(cil_0 = ifelse(cil < 0, 0, cil), ciu_0 = ifelse(ciu < 0, 0, ciu),
             est_0 = ifelse(est < 0, 0, est)) %>%
      mutate(ord_nms = forcats::fct_reorder(feature_nm, est))
    vimp_results <- current_vimp_results %>%
        filter(measure == "vimp_auc") %>%
        group_by(estimator_type) %>%
        mutate(cil_0 = ifelse(cil < 0, 0, cil), ciu_0 = ifelse(ciu < 0, 0, ciu),
               est_0 = ifelse(est < 0, 0, est)) %>%
        mutate(ord_nms = forcats::fct_reorder(feature_nm, est))
    shap_results <- current_vimp_results %>%
      filter(measure == "mean_abs_shap") %>%
      group_by(estimator_type) %>%
      mutate(ord_nms = forcats::fct_reorder(feature_nm, est))
    lime_results <- current_vimp_results %>%
      filter(measure == "mean_lime_select") %>%
      group_by(estimator_type) %>%
      mutate(ord_nms = forcats::fct_reorder(feature_nm, est))
    # individual plot of SPVIM estimates
    spvim_plot <- spvim_results %>%
        arrange(desc(est)) %>%
        ggplot(aes(x = est_0, y = ord_nms, group = estimator_type, shape = estimator_type, color = estimator_type)) +
        geom_errorbarh(aes(xmin = cil_0, xmax = ciu_0)) +
        geom_point(size = point_size) +
        ggtitle("SPVIM") +
        xlab(make_xlab(args$vimp_measure[i], "spvim")) +
        ylab("Feature") +
        labs(shape = "Estimator type", color = "Estimator type") +
        theme(legend.position = c(0.4, 0.2),
              axis.line = element_blank(),
              panel.border = element_rect(fill = NA, color = "black", linetype = 1, size = 1),
              plot.title = element_text(size = main_font_size),
              axis.title = element_text(size = axis_font_size),
              axis.text = element_text(size = axis_font_size),
              legend.text = element_text(size = axis_font_size),
              legend.title = element_text(size = axis_font_size),
              text = element_text(size = main_font_size))
    # individual plot of SHAP results
    shap_plot <- shap_results %>%
        arrange(desc(est)) %>%
        ggplot(aes(x = est, y = ord_nms, group = estimator_type, shape = estimator_type, color = estimator_type)) +
        geom_point(size = point_size) +
        ggtitle("Mean absolute SHAP") +
        xlab(make_shap_xlab(args$vimp_measure[i])) +
        scale_x_log10(breaks = c(1e-4, 1e-3, 1e-2, 1e-1, 1),
                      labels = c(expression(paste("1 x ", 10^-4, sep = "")),
                                 expression(paste("1 x ", 10^-3, sep = "")),
                                 expression(paste("1 x ", 10^-2, sep = "")),
                                 expression(paste("1 x ", 10^-1, sep = "")),
                                 "1")) +
        ylab("Feature") +
        guides(shape = FALSE, color = FALSE) +
        theme(axis.line = element_blank(),
              panel.border = element_rect(fill = NA, color = "black", linetype = 1, size = 1),
              plot.title = element_text(size = main_font_size),
              axis.title = element_text(size = axis_font_size),
              axis.text = element_text(size = axis_font_size),
              legend.text = element_text(size = axis_font_size),
              legend.title = element_text(size = axis_font_size),
              legend.position = lgnd_pos,
              text = element_text(size = main_font_size))
    # individual plot of VIM estimates
    vimp_plot <- vimp_results %>%
      arrange(desc(est)) %>%
      ggplot(aes(x = est_0, y = ord_nms, group = estimator_type,
                 shape = estimator_type, color = estimator_type)) +
      geom_errorbarh(aes(xmin = cil_0, xmax = ciu_0)) +
      geom_point(size = point_size) +
      ggtitle("Conditional VIM") +
      xlab(make_xlab(args$vimp_measure[i], "vimp")) +
      ylab("Feature") +
      guides(shape = FALSE, color = FALSE) +
      theme(axis.line = element_blank(),
            panel.border = element_rect(fill = NA, color = "black", linetype = 1, size = 1),
            plot.title = element_text(size = main_font_size),
            axis.title = element_text(size = axis_font_size),
            axis.text = element_text(size = axis_font_size),
            legend.text = element_text(size = axis_font_size),
            legend.title = element_text(size = axis_font_size),
            text = element_text(size = main_font_size))
    # individual plot of LIME proportions
    lime_plot <- lime_results %>%
      arrange(desc(est)) %>%
      ggplot(aes(x = est, y = ord_nms, group = estimator_type,
                 shape = estimator_type, color = estimator_type)) +
      geom_point(size = point_size) +
      ggtitle("LIME") +
      xlab("Prop. test instances feature selected") +
      ylab("Feature") +
      guides(shape = FALSE, color = FALSE) +
      theme(axis.line = element_blank(),
            panel.border = element_rect(fill = NA, color = "black", linetype = 1, size = 1),
            plot.title = element_text(size = main_font_size),
            axis.title = element_text(size = axis_font_size),
            axis.text = element_text(size = axis_font_size),
            legend.text = element_text(size = axis_font_size),
            legend.title = element_text(size = axis_font_size),
            text = element_text(size = main_font_size))


    ggsave(filename = paste0(plots_dir, "icu_analysis_", args$vimp_measure[i], ".png"),
           plot = plot_grid(spvim_plot, shap_plot, vimp_plot, lime_plot,
                            nrow = 1, labels = "AUTO"),
           width = fig_width, height = fig_height, units = 'cm')
    ggsave(filename = paste0(plots_dir, "icu_analysis_", args$vimp_measure[i], "_wide.png"),
           plot = plot_grid(spvim_plot, shap_plot, vimp_plot, lime_plot,
                            nrow = 1, labels = "AUTO"),
           width = fig_width * 2 / 3, height = fig_height * 0.9, units = 'cm', limitsize = FALSE)
}

## correlation between ranks given by SHAP and trees
icu_results_auc <- dplyr::bind_rows(icu_results_all_auc_nn, icu_results_all_auc_tree)

icu_results_auc_wide <- icu_results_auc %>%
  group_by(measure, est_type) %>%
  select(measure, est, est_type, feature) %>%
  pivot_wider(names_from = est_type, values_from = est)
icu_results_auc_wide %>%
  summarize(kendall_tau = cor(nn, tree, method = "kendall"))
