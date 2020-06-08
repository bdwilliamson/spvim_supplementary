#!/usr/local/bin/Rscript
# load results of the shapley sim

# ---------------------------------------------------
# load required libraries, set up arguments
# ---------------------------------------------------
library("argparse")
library("dplyr")
library("data.table")
library("tidyr")
library("ggplot2")
library("cowplot")
theme_set(theme_cowplot())
library("readr")

parser <- ArgumentParser()
parser$add_argument("--sim-name", default = "continuous-twohundred",
                    help = "the name of the simulation")
parser$add_argument("--nreps-total", type = "double", default = 1000,
                    help = "number of replicates in total")
parser$add_argument("--nreps-per-job", type = "double", default = 5,
                    help = "number of replicates for each job")
parser$add_argument("--est-type", default = "tree", help = "estimator type")
parser$add_argument("--cond-mean", default = "nonlinear", help = "conditional mean")
args <- parser$parse_args()

dir_postfix <- ifelse(args$cond_mean == "linear", "-lm/", "/")
dir_prefix <- strsplit(args$sim_name, "-", fixed = TRUE)[[1]][2]
output_dir <- paste0("./", dir_prefix, "-var-output", dir_postfix)
code_dir <- "./"
truth_dir <- "./"
plots_dir <- "./_plots/"
source(paste0(code_dir, "utils.R"))

# read in the truths
truths_init <- data.frame(truth = readRDS(paste0(truth_dir, "true_shapley_vals_",
                                                 ifelse(args$cond_mean == "linear", "lm_", ""),
                                                 "with_7_nonnoise_vars.rds"))[-1])
truths <- truths_init %>%
  mutate(feature = c(1, 3, 5, 11, 12, 13, 14))
true_mps <- NA
truths <- dplyr::bind_rows(truths, tibble(truth = 0, feature = 6)) %>%
  mutate(true_rank = rank(-abs(truth)))

# ---------------------------------------------------
# read in the data
# ---------------------------------------------------
if (args$cond_mean == "nonlinear") {
  ns <- c(500, 1000, 2000, 3000, 4000)
} else {
  ns <- c(500, 1000, 2000)
}
num_ns <- length(ns)

# total number of jobs for each sample size should be:
args$nreps_total/args$nreps_per_job * num_inds
# total number of reps for each index, sample size should be 1000 (2000 with SHAP values)

# names of files to read in
est_txt <- ifelse(args$est_type == "lm", "_est_lm_", "_est_tree_")
condmean_txt <- ifelse(args$cond_mean == "linear", "condmean_linear", "condmean_nonlinear")
output_nms <- paste0(args$sim_name, est_txt, condmean_txt, "_output_",
                     1:(args$nreps_total/args$nreps_per_job * num_ns * num_inds) - 1, postfix)
# list of output
max_ncols <- 20
output_lst <- lapply(paste0(output_dir, output_nms), function(x) read_func(x, nreps_per_job = args$nreps_per_job, max_ncol = max_ncols))
# remove nas
output_lst_nona <- output_lst[!is.na(output_lst)]
# make it a matrix
output_tib <- as_tibble(data.table::rbindlist(output_lst_nona)) %>%
    select(-X1) %>%
  mutate(feature = s + 1) %>%
  left_join(truths, by = "feature") %>%
  select(mc_id, measure, n, s, feature, truth, true_rank, est, se, cil, ciu, p_value, hyp_test, num_subsets_sampled, tidyselect::starts_with("mp_"))

saveRDS(output_tib, paste0(output_dir, "output_tib", est_txt, condmean_txt, ".rds"))
# ---------------------------------------------------
# compute performance metrics:
# bias*sqrt(n), var*n, mse*n, coverage
# ---------------------------------------------------
output_tib <- readRDS(paste0(output_dir, "output_tib", est_txt, condmean_txt, ".rds"))
output_tib %>%
    filter(measure != "mean_abs_shap") %>%
    group_by(n, feature) %>%
    summarize(n_row = n(), mn_num_subsets = mean(num_subsets_sampled)) %>%
  print(n = Inf)

round_num <- 2
raw_performance <- output_tib %>%
    mutate(bias = (est - truth)*sqrt(n),
           mse = (est - truth)^2*n, cover = (cil <= truth & ciu >= truth),
           round_cil = round(cil, round_num), round_ciu = round(ciu, round_num),
           round_truth = round(truth, round_num),
           rounded_cover = (round_cil <= round_truth & round_ciu >= round_truth))

# average over everything
average_performance_init <- raw_performance %>%
    group_by(n, feature, measure) %>%
    select(-truth, -se, -cil, -ciu) %>%
    summarize(bias = mean(bias, na.rm = TRUE), var = var(est, na.rm = TRUE),
              mse = mean(mse, na.rm = TRUE), cover = mean(cover, na.rm = TRUE),
              rounded_cover = mean(rounded_cover, na.rm = TRUE),
              sd = sd(est, na.rm = TRUE),
              power = mean(hyp_test)) %>%
    ungroup()
average_performance <- average_performance_init %>%
    group_by(n, feature, measure) %>%
    filter(!is.na(n)) %>%
    mutate(sd = sd*sqrt(n)/sqrt(args$nreps_total), var = var*n) %>%
    ungroup()
# break up into directly important vars (1, 3, 5) plus pure noise (6)
# and indirectly important vars (11, 12, 13, 14)
directly_important_perf <- average_performance %>%
  filter(feature %in% c(1, 3, 5, 6))
indirectly_important_perf <- average_performance %>%
  filter(feature %in% c(11, 12, 13, 14))
# without shap
directly_important_perf_noshap <- average_performance %>%
  filter(feature %in% c(1, 3, 5, 6), measure != "mean_abs_shap")
indirectly_important_perf_noshap <- average_performance %>%
  filter(feature %in% c(11, 12, 13, 14), measure != "mean_abs_shap")


# performance based on ranks
performance_rank <- output_tib %>%
  group_by(mc_id, measure, n) %>%
  summarize(spearman_cor = cor(est, truth, method = "spearman"),
            pearson_cor = cor(est, truth, method = "pearson"),
            kendall_tau = cor(est, truth, method = "kendall")) %>%
  ungroup() %>%
  group_by(measure, n) %>%
  summarize(spearman = mean(spearman_cor, na.rm = TRUE), pearson = mean(pearson_cor, na.rm = TRUE), kendall = mean(kendall_tau, na.rm = TRUE))
# ---------------------------------------------------
# make plots
# ---------------------------------------------------
point_size <- 5
text_size <- 20
title_text_size <- 30
y_lim_bias <- c(-2, 2)
y_lim_mse <- c(0, 5)
y_lim_var <- c(0, 4)
dodge_x <- 300
dodge_x_large <- 400
right_pad <- 10
# doing importance for 1, 3, 5, 6 (null), 11, 12, 13, 14 (nearly null)
non_null_point_vals <- c(16, 13, 18, 15, 17, 7)
null_point_vals <- 9
point_vals <- c(non_null_point_vals[1:3], null_point_vals, non_null_point_vals[4:6], null_point_vals)
legend_pos <- c(0.7, 0.8)

# make two plots: one for the directly important vars (s = 1, 3, 5) and null var (6)
# and one for the indirectly important (11, 12, 13, 14)
bias_plot <- directly_important_perf_noshap %>%
    ggplot(aes(x = n, y = bias, group = factor(paste(n, feature, sep = "_")),
               shape = factor(feature))) +
    geom_point(position = position_dodge(width = dodge_x), size = point_size) +
    geom_errorbar(aes(ymin = bias - 1.96*sd, ymax = bias + 1.96*sd), width = rep(200, nrow(directly_important_perf_noshap)),
                  position = position_dodge(width = dodge_x), size = 0.3*point_size) +
    ylab(expression(paste(sqrt(n), "x empirical ", bias[n]))) +
    xlab("n") +
    ggtitle(expression(bold(paste("Empirical bias scaled by ", sqrt(n))))) +
    scale_shape_manual(name = "Group", values = point_vals[1:4]) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    theme(legend.position = c(0.7, 0.2), text = element_text(size = text_size),
          axis.text = element_text(size = text_size),
          plot.title = element_text(size = text_size),
          plot.margin = unit(c(0, right_pad, 0, 0), "mm"))
bias_plot_indirect <- indirectly_important_perf_noshap %>%
  ggplot(aes(x = n, y = bias, group = factor(paste(n, feature, sep = "_")),
             shape = factor(feature))) +
  geom_point(position = position_dodge(width = dodge_x), size = point_size) +
  geom_errorbar(aes(ymin = bias - 1.96*sd, ymax = bias + 1.96*sd), width = rep(200, nrow(indirectly_important_perf_noshap)),
                position = position_dodge(width = dodge_x), size = 0.3*point_size) +
  ylab(expression(paste(sqrt(n), "x empirical ", bias[n]))) +
  xlab("n") +
  ggtitle(expression(bold(paste("Empirical bias scaled by ", sqrt(n), " (indirect)")))) +
  scale_shape_manual(name = "Group", values = point_vals[5:8]) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  theme(legend.position = c(0.7, 0.2), text = element_text(size = text_size),
        axis.text = element_text(size = text_size),
        plot.title = element_text(size = text_size),
        plot.margin = unit(c(0, right_pad, 0, 0), "mm"))

# coverage
cover_plot <- directly_important_perf_noshap %>%
    ggplot(aes(x = n, y = rounded_cover, group = factor(paste(n, feature, sep = "_")),
               shape = factor(feature))) +
    geom_point(position = position_dodge(width = dodge_x), size = point_size) +
    ylim(c(0, 1)) +
    ylab("Empirical coverage") +
    xlab("n") +
    ggtitle("Coverage") +
    scale_shape_manual(name = "Group", values = point_vals[1:4]) +
    geom_hline(yintercept = 0.95, linetype = "dashed", color = "red") +
    guides(shape = FALSE) +
    theme(text = element_text(size = text_size),
          axis.text = element_text(size = text_size),
          plot.title = element_text(size = text_size),
          plot.margin = unit(c(0, right_pad, 0, 0), "mm"))
cover_plot_indirect <- indirectly_important_perf_noshap %>%
  ggplot(aes(x = n, y = rounded_cover, group = factor(paste(n, feature, sep = "_")),
             shape = factor(feature))) +
  geom_point(position = position_dodge(width = dodge_x), size = point_size) +
  ylim(c(0, 1)) +
  ylab("Empirical coverage") +
  xlab("n") +
  ggtitle("Coverage (indirect)") +
  scale_shape_manual(name = "Group", values = point_vals[5:8]) +
  geom_hline(yintercept = 0.95, linetype = "dashed", color = "red") +
  guides(shape = FALSE) +
  theme(text = element_text(size = text_size),
        axis.text = element_text(size = text_size),
        plot.title = element_text(size = text_size),
        plot.margin = unit(c(0, right_pad, 0, 0), "mm"))

# variance
variance_plot <- directly_important_perf_noshap %>%
    ggplot(aes(x = n, y = var, group = factor(paste(n, feature, sep = "_")),
               shape = factor(feature))) +
    geom_point(position = position_dodge(width = dodge_x), size = point_size) +
    ylab(expression(paste(n, "x empirical ", var[n]))) +
    xlab("n") +
    ggtitle(expression(bold(paste("Empirical variance scaled by ", n)))) +
    scale_shape_manual(name = "Group", values = point_vals[1:4]) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    guides(shape = FALSE) +
    theme(text = element_text(size = text_size),
          axis.text = element_text(size = text_size),
          plot.title = element_text(size = text_size),
          plot.margin = unit(c(0, right_pad, 0, 0), "mm"))
variance_plot_indirect <- indirectly_important_perf_noshap %>%
  ggplot(aes(x = n, y = var, group = factor(paste(n, feature, sep = "_")),
             shape = factor(feature))) +
  geom_point(position = position_dodge(width = dodge_x), size = point_size) +
  ylab(expression(paste(n, "x empirical ", var[n]))) +
  xlab("n") +
  ggtitle(expression(bold(paste("Empirical variance scaled by ", n, " (indirect)")))) +
  scale_shape_manual(name = "Group", values = point_vals[5:8]) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  guides(shape = FALSE) +
  theme(text = element_text(size = text_size),
        axis.text = element_text(size = text_size),
        plot.title = element_text(size = text_size),
        plot.margin = unit(c(0, right_pad, 0, 0), "mm"))

# mse
mse_plot <- directly_important_perf_noshap %>%
    ggplot(aes(x = n, y = mse, group = factor(paste(n, feature, sep = "_")),
               shape = factor(feature))) +
    geom_point(position = position_dodge(width = dodge_x), size = point_size) +
    ylab(expression(paste(n, "x empirical ", MSE[n]))) +
    xlab("n") +
    ggtitle(expression(bold(paste("Empirical mean squared error scaled by ", n)))) +
    scale_shape_manual(name = "Feature", values = point_vals[1:4]) +
    theme(legend.position = legend_pos,
          text = element_text(size = text_size),
          axis.text = element_text(size = text_size),
          plot.title = element_text(size = text_size),
          plot.margin = unit(c(0, right_pad, 0, 0), "mm"))
mse_plot_indirect <- indirectly_important_perf_noshap %>%
  ggplot(aes(x = n, y = mse, group = factor(paste(n, feature, sep = "_")),
             shape = factor(feature))) +
  geom_point(position = position_dodge(width = dodge_x), size = point_size) +
  ylab(expression(paste(n, "x empirical ", MSE[n]))) +
  xlab("n") +
  ggtitle(expression(bold(paste("Empirical mean squared error scaled by ", n, " (indirect)")))) +
  scale_shape_manual(name = "Feature", values = point_vals[5:8]) +
  theme(legend.position = legend_pos,
        text = element_text(size = text_size),
        axis.text = element_text(size = text_size),
        plot.title = element_text(size = text_size),
        plot.margin = unit(c(0, right_pad, 0, 0), "mm"))

power_plot <- directly_important_perf_noshap %>%
    ggplot(aes(x = n, y = power, group = factor(paste(n, feature, sep = "_")),
               shape = factor(feature))) +
    geom_point(position = position_dodge(width = dodge_x), size = point_size) +
    ylab("Empirical power") +
    xlab("n") +
    scale_shape_manual(name = "Group", values = point_vals[1:4]) +
    ggtitle(expression(bold(paste("Proportion of tests rejected", sep = "")))) +
    geom_hline(yintercept = 0.05, linetype = "dashed", color = "red") +
    ylim(c(0, 1)) +
    guides(shape = FALSE) +
    theme(text = element_text(size = text_size),
          axis.text = element_text(size = text_size),
          plot.title = element_text(size = text_size),
          plot.margin = unit(c(0, right_pad, 0, 0), "mm"))
power_plot_indirect <- indirectly_important_perf_noshap %>%
  ggplot(aes(x = n, y = power, group = factor(paste(n, feature, sep = "_")),
             shape = factor(feature))) +
  geom_point(position = position_dodge(width = dodge_x), size = point_size) +
  ylab("Empirical power") +
  xlab("n") +
  scale_shape_manual(name = "Group", values = point_vals[5:8]) +
  ggtitle(expression(bold(paste("Proportion of tests rejected (indirect)", sep = "")))) +
  geom_hline(yintercept = 0.05, linetype = "dashed", color = "red") +
  ylim(c(0, 1)) +
  guides(shape = FALSE) +
  theme(text = element_text(size = text_size),
        axis.text = element_text(size = text_size),
        plot.title = element_text(size = text_size),
        plot.margin = unit(c(0, right_pad, 0, 0), "mm"))

# correlation between estimated and true ranks
rank_plot <- performance_rank %>%
  filter(measure != "<NA>") %>%
  pivot_longer(cols = -c(measure, n),
               names_to = "cor_type", values_to = "cor") %>%
  group_by(measure, n, cor_type) %>%
  filter(cor_type == "kendall") %>%
  ggplot(aes(x = n, y = cor, group = factor(paste(measure, cor_type, sep = "_")),
             shape = factor(measure, levels = c("mean_abs_shap", "r_squared"), labels = c("Mean abs. SHAP", "SPVIM")))) +
             # shape = factor(cor_type, levels = c("spearman", "pearson", "kendall"), labels = c("Spearman", "Pearson", "Kendall's tau")),
             # color = factor(measure, levels = c("mean_abs_shap", "r_squared"), labels = c("Mean abs. SHAP", "SPVIM")))) +
  geom_point(position = position_dodge(width = dodge_x), size = point_size) +
  ylab("Estimated correlation") +
  xlab("n") +
  ggtitle("Correlation between true and estimated values") +
  # labs(shape = "Correlation type", color = "Measure") +
  labs(shape = "Measure") +
  ylim(c(0, 1)) +
  theme(legend.position = c(0.7, 0.3), text = element_text(size = text_size),
        axis.text = element_text(size = text_size),
        plot.title = element_text(size = text_size),
        plot.margin = unit(c(0, right_pad, 0, 0), "mm"))

plot_grid(mse_plot, cover_plot, power_plot, rank_plot,
          mse_plot_indirect, cover_plot_indirect, power_plot_indirect,
          nrow = 2, ncol = 4, labels = "AUTO")
# estimates
shap_plot <- output_tib %>%
    filter(measure == "mean_abs_shap") %>%
    group_by(n, feature) %>%
    summarize(mn_shap = mean(est), sd = sd(est, na.rm = TRUE)) %>%
    mutate(sd = sd*sqrt(n)/sqrt(args$nreps_total)) %>%
    ggplot(aes(x = n, y = mn_shap, group = factor(paste(n, feature, sep = "_")),
               shape = factor(feature))) +
    geom_errorbar(aes(ymin = mn_shap - 1.96 * sd, ymax = mn_shap + 1.96 * sd),
                  position = position_dodge(width = dodge_x)) +
    geom_point(position = position_dodge(width = dodge_x), size = point_size) +
    ylab(expression(paste("Mean absolute SHAP value"))) +
    xlab("n") +
    ggtitle(expression(bold("Mean absolute SHAP value"))) +
    scale_shape_manual(name = "Feature", values = point_vals) +
    guides(shape = FALSE) +
    ylim(c(-0.1, 1.75)) +
    theme(text = element_text(size = text_size),
          axis.text = element_text(size = text_size),
          plot.title = element_text(size = text_size),
          plot.margin = unit(c(0, right_pad, 0, 0), "mm"))

est_spvim_plot <- output_tib %>%
  filter(measure != "mean_abs_shap") %>%
  group_by(n, feature) %>%
  summarize(mn_est = mean(est), sd = sd(est, na.rm = TRUE)) %>%
  mutate(sd = sd*sqrt(n)/sqrt(args$nreps_total)) %>%
  ggplot(aes(x = n, y = mn_est, group = factor(paste(n, feature, sep = "_")),
             shape = factor(feature))) +
  geom_errorbar(aes(ymin = mn_est - 1.96 * sd, ymax = mn_est + 1.96 * sd),
                position = position_dodge(width = dodge_x)) +
  geom_point(position = position_dodge(width = dodge_x), size = point_size) +
  ylab(expression(paste("Estimated SPVIM value"))) +
  xlab("n") +
  ggtitle(expression(bold("Estimated SPVIM value"))) +
  scale_shape_manual(name = "Feature", values = point_vals) +
  guides(shape = FALSE) +
  ylim(c(-0.075, 0.5)) +
  theme(text = element_text(size = text_size),
        axis.text = element_text(size = text_size),
        plot.title = element_text(size = text_size),
        plot.margin = unit(c(0, right_pad, 0, 0), "mm"))

plot_grid(est_spvim_plot, shap_plot)
ggsave(paste0(plots_dir, args$sim_name, "_separated_results", est_txt, condmean_txt, ".png"),
       plot = plot_grid(mse_plot, cover_plot, power_plot, rank_plot,
                        mse_plot_indirect, cover_plot_indirect, power_plot_indirect,
                        nrow = 2, ncol = 4, labels = "AUTO"),
       width = 100, height = 30, units = "cm")
ggsave(paste0(plots_dir, args$sim_name, "_est_spvim_shap", est_txt, condmean_txt, ".png"),
       plot = plot_grid(est_spvim_plot, shap_plot, labels = "AUTO"),
       width = 30, height = 15, units = "cm")
# ---------------------------------------------------
# look at average estimates
# ---------------------------------------------------
# SHAP values
output_tib %>%
    filter(measure == "mean_abs_shap") %>%
    group_by(n, feature) %>%
    summarize(mn_shap = mean(est)) %>%
      print(n = Inf)

# SPVIM values
output_tib %>%
  group_by(n, feature, measure) %>%
  filter(measure != "mean_abs_shap") %>%
  summarize(mn = mean(est), truth = mean(truth)) %>%
  print(n = Inf)

# look at CI width
output_tib %>%
  filter(measure != "mean_abs_shap") %>%
  mutate(width = ciu - cil) %>%
  group_by(n, feature) %>%
  summarize(mn_width = mean(width, na.rm = TRUE)) %>%
  print(n = Inf)
output_tib %>%
  filter(measure != "mean_abs_shap") %>%
  group_by(n, feature) %>%
  summarize(mn_se = mean(se, na.rm = TRUE)) %>%
  print(n = Inf)
