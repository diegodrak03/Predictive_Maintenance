import numpy as np
import pandas as pd
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
