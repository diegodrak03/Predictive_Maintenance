import matplotlib.pyplot as plt
import pandas as pd

from pm import config, data, evaluation, models, plots, split


COLUMN_SETS = {
    "basic": config.BASIC_NUMERICAL_COLUMNS,
    "basic + physical": config.SELECTED_NUMERICAL_COLUMNS,
}


def run_experiments(train):
    """
    PR-AUC of the 12 experiments: 3 models x with and without class weighting x 2 column sets.
    """
    y = train[config.LABEL]
    spw = (y == 0).sum() / (y == 1).sum()

    experiments = [
        ("logistic", "none", lambda cols: models.build_logistic_pipeline(cols)),
        ("logistic", "balanced", lambda cols: models.build_logistic_pipeline(cols, class_weight="balanced")),
        ("random forest", "none", lambda cols: models.build_random_forest_pipeline(cols)),
        ("random forest", "balanced", lambda cols: models.build_random_forest_pipeline(cols, class_weight="balanced")),
        ("xgboost", "none", lambda cols: models.build_xgboost_pipeline(cols)),
        ("xgboost", f"scale_pos_weight={spw:.1f}", lambda cols: models.build_xgboost_pipeline(cols, scale_pos_weight=spw)),
    ]

    rows = []
    for set_name, numeric_columns in COLUMN_SETS.items():
        X, y, groups = data.get_xy(train, numeric_columns)
        for model_name, weighting, build in experiments:
            scores = evaluation.cv_pr_auc(build(numeric_columns), X, y, groups)
            row = {"model": model_name, "weighting": weighting, "columns": set_name,
                   "pr_auc_mean": scores.mean(), "pr_auc_std": scores.std()}
            row.update({f"fold_{i}": s for i, s in enumerate(scores, start=1)})
            rows.append(row)
            print(f"{model_name:14} {weighting:22} {set_name:17} {scores.mean():.3f} +- {scores.std():.3f}")

    return pd.DataFrame(rows)


def data_figures(df):
    """
    Exploratory figures of the whole processed dataset (no model involved).
    """
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(df["uid"], df["tool_wear_min"], color=plots.BLUE, linewidth=0.8)
    ax.set_xlabel("UID (process order) [-]")
    ax.set_ylabel("Tool wear [min]")
    ax.set_title("Tool wear vs UID: each sawtooth is one tool cycle")
    plots.style_axes(ax)
    plots.save_figure(fig, "tool_wear_vs_uid")

    failure = df[config.LABEL] == 1
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df.loc[~failure, "rotational_speed_rpm"], df.loc[~failure, "torque_nm"],
               s=6, alpha=0.3, color=plots.BLUE, label="No failure")
    ax.scatter(df.loc[failure, "rotational_speed_rpm"], df.loc[failure, "torque_nm"],
               s=14, alpha=0.9, color=plots.ORANGE, label="Failure")
    ax.set_xlabel("Rotational speed [rpm]")
    ax.set_ylabel("Torque [Nm]")
    ax.set_title("Torque vs rotational speed")
    ax.legend(frameon=False)
    plots.style_axes(ax)
    plots.save_figure(fig, "torque_vs_speed")

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(df["uid"], df["temp_diff_k"], color=plots.BLUE, linewidth=0.8)
    ax.axhline(config.HDF_TEMP_DIFF_K, color=plots.ORANGE, linestyle="--", linewidth=1.2,
               label=f"HDF threshold ({config.HDF_TEMP_DIFF_K} K)")
    ax.set_xlabel("UID (process order) [-]")
    ax.set_ylabel("Process - air temperature [K]")
    ax.set_title("Temperature difference vs UID")
    ax.legend(frameon=False, loc="upper right")
    plots.style_axes(ax)
    plots.save_figure(fig, "temp_diff_vs_uid")


def main():
    df = data.load_processed()
    train, _test = split.split_train_test(df)

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Random baseline (failure rate): {train[config.LABEL].mean():.3f}\n")
    cv_results = run_experiments(train)
    cv_results.to_csv(config.METRICS_DIR / "cv_results.csv", index=False)

    X, y, groups = data.get_xy(train, config.SELECTED_NUMERICAL_COLUMNS)
    proba = evaluation.oof_proba(models.build_selected_model(), X, y, groups)

    oof = train[["uid", config.GROUPS, config.LABEL] + config.FAILURE_MODE_COLUMNS + ["tool_wear_min"]].copy()
    oof["proba"] = proba
    oof.to_csv(config.METRICS_DIR / "oof_predictions.csv", index=False)

    print(f"\nOut-of-fold PR-AUC of the selected model: {evaluation.pr_auc(y, proba):.3f}")
    print(f"Best threshold with precision >= {config.MIN_PRECISION}: {evaluation.best_threshold(y, proba):.3f}")

    thresholds = evaluation.threshold_table(y, proba)
    thresholds.to_csv(config.METRICS_DIR / "thresholds.csv", index=False)
    print("\n" + thresholds.round(3).to_string(index=False))

    by_mode = evaluation.recall_by_mode(train, proba >= config.SELECTED_THRESHOLD)
    by_mode.to_csv(config.METRICS_DIR / "recall_by_mode.csv", index=False)
    print("\n" + by_mode.round(3).to_string(index=False))

    data_figures(df)
    plots.save_figure(plots.proba_histogram(y, proba), "oof_proba_histogram")
    plots.save_figure(plots.pr_curve(y, proba), "pr_curve_oof")

    print(f"\nSaved metrics to {config.METRICS_DIR} and figures to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
