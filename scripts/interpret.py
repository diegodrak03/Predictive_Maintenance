import numpy as np
import pandas as pd

from pm import config, data, evaluation, models, plots, split


# Grids cover the full training range (PWF lives at both extremes of power); reference lines are the generator's rules
PARTIAL_DEPENDENCE = [
    ("tool_wear_min", "Tool wear [min]", [(200, "TWF window starts (200 min)")]),
    ("power_w", "Power [W]", [(3500, "PWF limits (3500 and 9000 W)"), (9000, None)]),
    ("temp_diff_k", "Process - air temperature [K]", [(config.HDF_TEMP_DIFF_K, f"HDF threshold ({config.HDF_TEMP_DIFF_K} K)")]),
]


def missed_failures(oof):
    """
    Machine failures that the selected model misses out-of-fold at the selected threshold.
    """
    missed = oof[(oof[config.LABEL] == 1) & (oof["proba"] < config.SELECTED_THRESHOLD)].copy()

    names = [config.FAILURE_MODE_NAMES[m] for m in config.FAILURE_MODE_COLUMNS]
    flags = missed[config.FAILURE_MODE_COLUMNS].to_numpy() == 1
    missed["modes"] = [", ".join(np.array(names)[row]) or "none" for row in flags]

    return missed[["uid", config.GROUPS, "tool_wear_min", "modes", "proba"]].sort_values("uid")


def main():
    oof_path = config.METRICS_DIR / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"{oof_path} not found: run scripts/run_cv.py first")

    df = data.load_processed()
    train, _test = split.split_train_test(df)
    X, y, groups = data.get_xy(train, config.SELECTED_NUMERICAL_COLUMNS)

    folds = evaluation.fit_cv_folds(models.build_selected_model(), X, y, groups)

    importance = evaluation.cv_permutation_importance(folds)
    importance.to_csv(config.METRICS_DIR / "permutation_importance.csv", index=False)
    print("Permutation importance (drop in PR-AUC on the validation fold, mean of 5 folds):")
    print(importance.round(3).to_string(index=False))

    fig = plots.bar_chart(importance["feature"], importance["mean"], importance["std"],
                          xlabel="Drop in PR-AUC when the column is permuted [-]",
                          title="Permutation importance (validation folds, mean +- std)")
    plots.save_figure(fig, "permutation_importance")

    for feature, xlabel, references in PARTIAL_DEPENDENCE:
        grid = np.linspace(X[feature].min(), X[feature].max(), 80)
        table = evaluation.cv_partial_dependence(folds, feature, grid)
        table.to_csv(config.METRICS_DIR / f"partial_dependence_{feature}.csv", index=False)
        fig = plots.partial_dependence_plot(table, feature, xlabel,
                                            f"Partial dependence on {feature}", references)
        plots.save_figure(fig, f"partial_dependence_{feature}")

    oof = pd.read_csv(oof_path)
    missed = missed_failures(oof)
    missed.to_csv(config.METRICS_DIR / "missed_failures.csv", index=False)

    print(f"\nMissed failures (out-of-fold, threshold {config.SELECTED_THRESHOLD}): {len(missed)}")
    print(missed["modes"].value_counts().to_string())
    print("\n" + missed.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
