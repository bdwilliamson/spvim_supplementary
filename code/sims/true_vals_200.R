# compute true values for 200-variable sim
output_dir <- "code/sims/"
source(paste0(output_dir, "sim_truth_utils.R"))
# set up Sigma, n
n <- 1e4
nsim <- 1e3
nsim_outer <- 1e2
p <- 7
cor <- c(0.7, 0.3, 0.05)
Sigma <- diag(1, nrow = p, ncol = p)
Sigma[1, 4] <- Sigma[4, 1] <- cor[1]
Sigma[2, 5] <- Sigma[5, 2] <- Sigma[2, 6] <- Sigma[6, 2] <- cor[2]
Sigma[3, 7] <- Sigma[7, 3] <- cor[3]

# get the full R-squared and variance of the outcome
set.seed(4747)
x <- gen_x(1e7, Sigma)
y <- gen_y(x)
var <- mean((y - mean(y))^2)
full_r2 <- 1 - mean((y - full_conditional_mean(x))^2)/var
full_r2
# the null r2 is zero
null_r2 <- 0

# get the rest of the R-squareds
all_subsets <- get_powerset(1:p)
sampling_weights <- unlist(lapply(all_subsets, function(s, p) {
    if (length(s) == 0) {
        return(1)
    } else if (length(s) == p) {
        return(1)
    } else {
        return(choose(p - 2, length(s) - 1) ^ (-1))
    }
}, p = p))
smaller_n <- 2e3
S <- all_subsets[-c(1, length(all_subsets))]
true_W <- diag(sampling_weights / sum(sampling_weights))
intermediate_r2s <- rep(list(vector("numeric", nsim_outer)), length(S))
set.seed(4747)
for (s in 1:length(S)) {
    for (i in 1:nsim_outer) {
        this_x <- gen_x(smaller_n, Sigma)
        this_y <- gen_y(this_x)
        this_conditional_mean <- conditional_mean_s(this_x, S[[s]], Sigma, nsim)
        intermediate_r2s[[s]][i] <- 1 - mean((this_y - this_conditional_mean)^2)/var
    }
}
# compute the Shapley values using all of the R-squareds
intermediate_r2 <- unlist(lapply(intermediate_r2s, mean))
all_r2s <- c(null_r2, abs(intermediate_r2), full_r2)
## do wls
Z <- cbind(1, do.call(rbind, lapply(all_subsets, get_z, p = p)))
mod <- lm(all_r2s ~ Z - 1, weights = sampling_weights)
est <- coef(mod)
est
v <- matrix(all_r2s)
A_W <- sqrt(true_W) %*% Z
v_W <- sqrt(true_W) %*% v
G <- rbind(c(1, rep(0, p)), rep(1, p + 1) - c(1, rep(0, p)))
c_n <- matrix(c(null_r2, full_r2 - null_r2), ncol = 1)
kkt_matrix_11 <- 2 * t(A_W) %*% A_W
kkt_matrix_12 <- t(G)
kkt_matrix_21 <- G
kkt_matrix_22 <- matrix(0, nrow = dim(kkt_matrix_21)[1],  ncol = dim(kkt_matrix_12)[2])
kkt_matrix <- rbind(cbind(kkt_matrix_11, kkt_matrix_12), cbind(kkt_matrix_21, kkt_matrix_22))
ls_matrix <- rbind(2 * t(A_W) %*% v_W, c_n)
ls_solution <- solve(kkt_matrix) %*% ls_matrix
est2 <- ls_solution[1:(p + 1), , drop = FALSE]
cbind(est, est2)
for (i in 1:p) {
    print(paste0("Estimate for j = ", i, " is ", round(est[i + 1], 3)))
}
sum(est)
sum(est2)
full_r2
saveRDS(all_r2s, file = paste0(output_dir, "true_r2s_with_7_nonnoise_vars.rds"))
saveRDS(intermediate_r2s, file = paste0(output_dir, "sampled_r2s_with_7_nonnoise_vars.rds"))
saveRDS(est, file = paste0(output_dir, "true_shapley_vals_with_7_nonnoise_vars.rds"))
