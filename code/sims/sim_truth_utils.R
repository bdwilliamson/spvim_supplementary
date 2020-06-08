# sim_truth_utils.R
# utility functions for sim truths
gen_x <- function(n, Sigma) {
    x <- MASS::mvrnorm(n = n, mu = rep(0, nrow(Sigma)), Sigma = Sigma)
    return(x)
}
# 7 variables only (three direct, four indirect)
full_conditional_mean <- function(x) {
    return(sign(x[, 1]) + step_function(x[, 2]) + wiggle(x[, 3]))
}
get_betahat <- function(x, y) {
    return(solve(t(x)%*%x)%*%t(x)%*%as.matrix(y))
}
lm_conditional_mean <- function(x, beta) {
    return(x%*%as.matrix(beta))
}
gen_y <- function(x) {
    y <- full_conditional_mean(x) + rnorm(nrow(x), 0, 1)
    return(y)
}
gen_y_lm <- function(x, beta) {
    y <- lm_conditional_mean(x, beta) + rnorm(nrow(x), 0, 1)
    return(y)
}
step_function <- function(x) {
    return(0 + (-1) * 6 * (x <= -4) + (-1) * 4 * (x > -4) * (x <= -2) + (-1) * 2 * (x > -2) * (x <= 0) + 2 * (x > 2) * (x <= 4) + 4 * (x > 4))
}
wiggle <- function(x) {
    return(0 + (-1) * (x <= -4) + (-1) * (x > -2) * (x <= 0) + (-1) * (x > 2) * (x <= 4) + 1 * (x > -4) * (x <= -2) + 1 * (x > 0) * (x <= 2) + 1 * (x > 4))    
}
get_sigmas <- function(s, Sigma) {
    Sigma_11 <- Sigma[s, s]
    Sigma_12 <- Sigma[s, -s, drop = FALSE]
    Sigma_21 <- Sigma[-s, s, drop = FALSE]
    Sigma_22 <- Sigma[-s, -s, drop = FALSE]
    Sigma_22_inv <- solve(Sigma_22)
    return(list(sig1 = Sigma_11, sig12 = Sigma_12,
                sig21 = Sigma_21, sig22inv = Sigma_22_inv))
}

conditional_mean_s <- function(x, s, Sigma, nsim) {
    # get Sigma list
    sigma_lst <- get_sigmas(-s, Sigma)
    # generate newx based on sigma list
    samp_x <- array(data = NA, dim = c(nrow(x), ncol(x), nsim))
    mean_matrix <- t(sigma_lst$sig12 %*% sigma_lst$sig22inv%*%t(as.matrix(x[, s])))
    sigma_matrix <- sigma_lst$sig1 - sigma_lst$sig12 %*% sigma_lst$sig22inv %*% sigma_lst$sig21
    samp_x[, s, ] <- x[, s]
    for (i in 1:nrow(x)) {
        samp_x[i, -s, ] <- t(MASS::mvrnorm(n = nsim, mu = mean_matrix[i, ],
                                       Sigma = sigma_matrix))
    }

    # get conditional mean based on what s is
    samp_f1 <- apply(samp_x, 3, function(x) sign(x[, 1, drop = FALSE]))
    samp_f2 <- apply(samp_x, 3, function(x) step_function(x[, 2, drop = FALSE]))
    samp_f3 <- apply(samp_x, 3, function(x) wiggle(x[, 3, drop = FALSE]))
    mean_filtered <- rowMeans(samp_f1) + rowMeans(samp_f2) + rowMeans(samp_f3)
    return(mean_filtered)
}
shapley_val_s_lm <- function(x, y, s, Sigma, nsim, var) {
    # get Sigma list
    sigma_lst <- get_sigmas(-s, Sigma)
    # generate newx based on sigma list
    samp_x <- array(data = NA, dim = c(nrow(x), ncol(x), nsim))
    mean_matrix <- t(sigma_lst$sig12 %*% sigma_lst$sig22inv%*%t(as.matrix(x[, s])))
    sigma_matrix <- sigma_lst$sig1 - sigma_lst$sig12 %*% sigma_lst$sig22inv %*% sigma_lst$sig21
    samp_x[, s, ] <- x[, s]
    for (i in 1:nrow(x)) {
        samp_x[i, -s, ] <- t(MASS::mvrnorm(n = nsim, mu = mean_matrix[i, ],
                                           Sigma = sigma_matrix))
    }
    all_betahat_s <- apply(samp_x, 3, function(x, y) get_betahat(x, y), y = y)
    betahat_s <- rowMeans(all_betahat_s)
    all_conditional_means <- apply(samp_x, 3, function(x, betahat) x%*%as.matrix(betahat),
                                   betahat = betahat_s)
    all_r2s <- 1 - colMeans((sweep(all_conditional_means, MARGIN = 1, STATS = y, FUN = "-"))^2)/var
    return(mean(all_r2s))
}

# get all possible subsets
get_powerset <- function(set) { 
    n <- length(set)
    masks <- 2^(1:n-1)
    lapply( 1:2^n-1, function(u) set[ bitwAnd(u, masks) != 0 ] )
}
get_z <- function(s, p) {
    max_subset <- 1:p
    z <- rep(0, p)
    z[match(s, max_subset)] <- 1
    return(z)
}
