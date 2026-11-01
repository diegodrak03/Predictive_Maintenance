import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import cross_val_predict, cross_val_score

from pm import config, split


def cv_pr_auc(pipeline, X, y, groups):
    """
    PR-AUC (average precision) of each fold of the training-set cross-validation.

    Returns:
        np.ndarray: One score per fold, in fold order.
    """
    return cross_val_score(pipeline, X, y, cv=split.cv_splitter(), groups=groups,
                           scoring="average_precision")


def oof_proba(pipeline, X, y, groups):
    """
    Out-of-fold failure probabilities: each process is predicted by the model trained
    on the other folds.

    Returns:
        np.ndarray: Probability of machine failure for each row of X.
    """
    return cross_val_predict(pipeline, X, y, cv=split.cv_splitter(), groups=groups,
                             method="predict_proba")[:, 1]


def threshold_metrics(y, proba, threshold):
    """
    Precision, recall, detected, missed and false alarms when every process with
    probability >= threshold is flagged as a failure.

    Returns:
        dict: The metrics for that threshold.
    """
    y = np.asarray(y) == 1
    pred = np.asarray(proba) >= threshold

    detected = int((pred & y).sum())
    missed = int((~pred & y).sum())
    false_alarms = int((pred & ~y).sum())

    flagged = detected + false_alarms

    return {
        "threshold": threshold,
        "precision": detected / flagged if flagged else np.nan,
        "recall": detected / (detected + missed),
        "detected": detected,
        "missed": missed,
        "false_alarms": false_alarms,
    }


def threshold_table(y, proba, thresholds=config.THRESHOLDS_TO_REPORT):
    """
    threshold_metrics for each threshold, one row per threshold.
    """
    return pd.DataFrame([threshold_metrics(y, proba, t) for t in thresholds])


def best_threshold(y, proba, min_precision=config.MIN_PRECISION):
    """
    Threshold with the highest recall among those with precision >= min_precision
    (decisions.md, section 16).

    Returns:
        float: The threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y, proba)

    # precision and recall have one more element than thresholds: drop the last one
    valid = precision[:-1] >= min_precision
    best = np.argmax(recall[:-1][valid])

    return thresholds[valid][best]


def recall_by_mode(df, pred):
    """
    Recall per failure mode, counting only processes that are machine failures,
    plus a row for failures with no mode flagged.

    Args:
        df (pd.DataFrame): The rows that were predicted, with the label and mode columns.
        pred (array-like): Boolean prediction for each row of df.

    Returns:
        pd.DataFrame: mode, failures, detected and recall.
    """
    pred = np.asarray(pred)
    failure = (df[config.LABEL] == 1).to_numpy()
    no_mode = failure & (df[config.FAILURE_MODE_COLUMNS].sum(axis=1) == 0).to_numpy()

    rows = []
    for mode in config.FAILURE_MODE_COLUMNS:
        is_mode = (df[mode] == 1).to_numpy() & failure
        rows.append({"mode": config.FAILURE_MODE_NAMES[mode],
                     "failures": int(is_mode.sum()),
                     "detected": int((pred & is_mode).sum())})

    rows.append({"mode": "No mode flagged",
                 "failures": int(no_mode.sum()),
                 "detected": int((pred & no_mode).sum())})

    by_mode = pd.DataFrame(rows)
    by_mode["recall"] = by_mode["detected"] / by_mode["failures"].replace(0, np.nan)

    return by_mode


def pr_auc(y, proba):
    """
    PR-AUC (average precision) of a set of predictions.
    """
    return average_precision_score(y, proba)


def fit_cv_folds(pipeline, X, y, groups):
    """
    Train a copy of the pipeline on the training part of each cross-validation split.

    Returns:
        list: (fitted pipeline, X_val, y_val) for each fold, in fold order.
    """
    folds = []
    for train_idx, val_idx in split.cv_splitter().split(X, y, groups):
        model = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
        folds.append((model, X.iloc[val_idx], y.iloc[val_idx]))
    return folds


def cv_permutation_importance(folds, n_repeats=10):
    """
    Permutation importance (drop in average precision) computed on the validation fold
    of each split. Type is permuted as a single column, before one-hot encoding.

    Args:
        folds (list): Output of fit_cv_folds.

    Returns:
        pd.DataFrame: One row per input column, one column per fold, plus mean and std
        of the fold means, sorted by mean.
    """
    per_fold = {}
    for i, (model, X_val, y_val) in enumerate(folds, start=1):
        result = permutation_importance(model, X_val, y_val, scoring="average_precision",
                                        n_repeats=n_repeats, random_state=config.SEED, n_jobs=-1)
        per_fold[f"fold_{i}"] = pd.Series(result.importances_mean, index=X_val.columns)

    table = pd.DataFrame(per_fold)
    table.insert(0, "std", table.std(axis=1, ddof=0))
    table.insert(0, "mean", table.drop(columns="std").mean(axis=1))
    return table.sort_values("mean", ascending=False).rename_axis("feature").reset_index()


def cv_partial_dependence(folds, feature, grid):
    """
    Partial dependence of the predicted failure probability on one feature: the feature is
    set to each grid value in every row of the validation fold, and the predicted
    probabilities are averaged. Other columns are left unchanged, so combinations that
    cannot happen physically are also evaluated.

    Args:
        folds (list): Output of fit_cv_folds.
        feature (str): Input column to vary.
        grid (array-like): Values of the feature.

    Returns:
        pd.DataFrame: One row per grid value, one column per fold, plus the mean.
    """
    table = pd.DataFrame({feature: grid})
    for i, (model, X_val, _y_val) in enumerate(folds, start=1):
        X_mod = X_val.copy()
        values = []
        for value in grid:
            X_mod[feature] = value
            values.append(model.predict_proba(X_mod)[:, 1].mean())
        table[f"fold_{i}"] = values
    table["mean"] = table.filter(like="fold_").mean(axis=1)
    return table
