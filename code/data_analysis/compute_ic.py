## compute influence function for a single measure of predictiveness


def compute_ic(y, preds, measure):
    """
    Compute IC based on the given expected reward

    @param y: the outcome
    @param preds: the predictions based on the current subset of features
    @param measure: the expected reward

    @return an n-vector of the IC for the given expected reward
    """
    import numpy as np
    import sklearn.metrics as skm

    ## if auc, do one thing; if r_squared, do another
    if measure == "auc":
        p_1 = np.mean(y)
        p_0 = 1 - p_1

        sens = np.array([np.mean(preds[(y == 0).reshape(preds.shape)] < x) for x in preds])
        spec = np.array([np.mean(preds[(y == 1).reshape(preds.shape)] > x) for x in preds])

        contrib_1 = (y == 1).reshape(preds.shape) / p_1 * sens
        contrib_0 = (y == 0).reshape(preds.shape) / p_0 * spec

        auc = skm.roc_auc_score(y_true = y, y_score = preds, average = "micro")
        return contrib_1 + contrib_0 - ((y == 0).reshape(preds.shape) / p_0 + (y == 1).reshape(preds.shape) / p_1) * auc
    elif measure == "r_squared":
        mse = skm.mean_squared_error(y_true = y, y_pred = preds)
        var = np.mean((y - np.mean(y)) ** 2)
        d_mse = (y.reshape(preds.shape) - preds)**2 - mse
        d_var = (y.reshape(preds.shape) - np.mean(y)) ** 2 - var
        grad = np.array([1. / var, -mse / (var ** 2)])
        return np.dot(grad, np.vstack((d_mse, d_var)))
