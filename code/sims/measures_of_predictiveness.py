## user-defined measure functions


def cv_predictiveness(data, S, measure, pred_func, V = 5, stratified = True, na_rm = False, type = "regression", ensemble = False, run_cv = False):
    """
    Compute a cross-validated measure of predictiveness based on the data
    and the chosen measure

    @param data: dataset
    @param S: the covariates to fit
    @param measure: measure of predictiveness
    @param pred_func: function that fits to the data
    @param V: the number of CV folds
    @param stratified: should the folds be stratified?
    @param na_rm: should we do a complete-case analysis (True) or not (False)
    @param type: is this regression (use predict) or classification (use predict_proba)?
    @param ensemble: is this an ensemble (True) or not (False)?

    @return cross-validated measure of predictiveness, along with preds and ics
    """
    import numpy as np
    from compute_ic import compute_ic
    import utils as uts
    from data_generator import Dataset
    ## if na_rm = True, do a complete-case analysis
    if na_rm:
        xs = data.x_train[:, S]
        cc = np.sum(np.isnan(xs), axis = 1) == 0
        newdata = Dataset(x_train = data.x_train[cc, :], y_train = data.y_train[cc])
    else:
        cc = np.repeat(True, data.x_train.shape[0])
        newdata = data
    ## set up CV folds
    folds = uts.make_folds(newdata, V, stratified = stratified)
    ## do CV
    preds = np.empty((data.y_train.shape[0],))
    preds.fill(np.nan)
    ics = np.empty((data.y_train.shape[0],))
    ics.fill(np.nan)
    # preds = np.empty((newdata.y_train.shape[0],))
    vs = np.empty((V,))
    # ics = np.empty((newdata.y_train.shape[0],))
    cc_cond = np.flatnonzero(cc)
    for v in range(V):
        fold_cond = np.flatnonzero(folds == v)
        x_train, y_train = newdata.x_train[folds != v, :], newdata.y_train[folds != v]
        x_test, y_test = newdata.x_train[folds == v, :], newdata.y_train[folds == v]
        pred_func.fit(x_train[:, S], np.ravel(y_train))
        if ensemble:
            preds_v = np.mean(pred_func.transform(x_test[:, S]))
        else:
            if type == "classification":
                preds_v = pred_func.predict_proba(x_test[:, S])[:, 1]
            else:
                preds_v = pred_func.predict(x_test[:, S])
        preds[cc_cond[fold_cond]] = preds_v
        vs[v] = measure(y_test, preds_v)
        ics[cc_cond[fold_cond]] = compute_ic(y_test, preds_v, measure.__name__)
    return np.mean(vs), preds, ics, folds


def auc(y, preds, *args, **kwargs):
    """
    Compute AUC for a given set of predictions and outcomes

    @param y: the outcome
    @param preds: the predictions based on a given subset of features

    @return the AUC
    """
    import sklearn.metrics as skm

    if len(preds.shape) == 2:
        if preds.shape[1] > 1:
            return [skm.roc_auc_score(y_true = y, y_score = preds[:, i], average = "micro") for i in range(preds.shape[1])]
        else:
            return skm.roc_auc_score(y_true = y, y_score = preds, average = "micro")
    else:
        return skm.roc_auc_score(y_true = y, y_score = preds, average = "micro")


def r_squared(y, preds):
    """
    Compute R^s for a given set of predictions and outcomes

    @param y: the outcome
    @param preds: the predictions based on a given subset of features

    @return the R^2
    """
    import sklearn.metrics as skm
    import numpy as np
    var = np.mean((y - np.mean(y)) ** 2)

    if len(preds.shape) == 2:
        if preds.shape[1] > 1:
            return [1. - skm.mean_squared_error(y_true = y, y_pred = preds[:, i]) / var for i in range(preds.shape[1])]
        else:
            return 1. - skm.mean_squared_error(y_true = y, y_pred = preds) / var
    else:
        return 1. - skm.mean_squared_error(y_true = y, y_pred = preds) / var
