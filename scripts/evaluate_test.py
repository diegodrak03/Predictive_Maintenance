import pandas as pd

from pm import config, data, evaluation, models, plots, split


def summary_row(experiment, role, train, test, summary, by_mode):
    """
    One row of test_results.csv: the metrics plus recall per failure mode.
    """
    row = {"experiment": experiment, "role": role,
           "train_processes": len(train), "test_processes": len(test),
           "test_failures": int(test[config.LABEL].sum())}
    row.update(summary)
    row.update({f"recall_{m}": r for m, r in zip(by_mode["mode"], by_mode["recall"])})
    return row


def print_result(title, summary, by_mode):
    print(f"\n=== {title}")
    print(f"PR-AUC {summary['pr_auc']:.3f} | threshold {summary['threshold']} | "
          f"precision {summary['precision']:.3f} | recall {summary['recall']:.3f} | "
          f"detected {summary['detected']} | missed {summary['missed']} | "
          f"false alarms {summary['false_alarms']}")
    print(by_mode.round(3).to_string(index=False))


def main():
    df = data.load_processed()
    train, test = split.split_train_test(df)

    rows = []
    by_mode_tables = []

    # 1. Main result: selected model and threshold, fixed before this point. Test set used once.
    proba, summary, by_mode = evaluation.evaluate_holdout(
        models.build_selected_model(), train, test, config.SELECTED_NUMERICAL_COLUMNS)
    print_result("MAIN: selected model on the frozen test set", summary, by_mode)
    rows.append(summary_row("main", "main", train, test, summary, by_mode))
    by_mode_tables.append(by_mode.assign(experiment="main"))

    _X, y_test, _groups = data.get_xy(test, config.SELECTED_NUMERICAL_COLUMNS)
    plots.save_figure(plots.pr_curve(y_test, proba, title="Precision-recall curve (frozen test set)"), "pr_curve_test")
    plots.save_figure(plots.proba_histogram(y_test, proba, title="Predicted probabilities by class (frozen test set)"),
                      "test_proba_histogram")

    cv_mean, cv_std = config.CV_PR_AUC
    logistic_mean, logistic_std = config.CV_LOGISTIC_PR_AUC
    print(f"\nCross-validation reference: PR-AUC {cv_mean} +- {cv_std}, "
          f"precision {config.CV_PRECISION}, recall {config.CV_RECALL} at {config.SELECTED_THRESHOLD}")
    print(f"Test minus CV: PR-AUC {summary['pr_auc'] - cv_mean:+.3f}, "
          f"precision {summary['precision'] - config.CV_PRECISION:+.3f}, "
          f"recall {summary['recall'] - config.CV_RECALL:+.3f}")

    criteria = {
        f"recall >= {config.MIN_RECALL}": summary["recall"] >= config.MIN_RECALL,
        f"precision >= {config.MIN_PRECISION}": summary["precision"] >= config.MIN_PRECISION,
        f"PR-AUC clearly above logistic regression (CV {logistic_mean} +- {logistic_std})":
            summary["pr_auc"] > logistic_mean + 3 * logistic_std,
    }
    print("\nSuccess criterion (decisions.md, section 17):")
    for name, passed in criteria.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"  => {'MET' if all(criteria.values()) else 'NOT MET'}")
    print("Note: the test set has no RNF failures, so test recall may be slightly optimistic.")

    # 2. Secondary: temporal drift. Same model and threshold, trained on the first 96 cycles.
    temporal_train, temporal_test = split.temporal_split(df)
    _proba, summary, by_mode = evaluation.evaluate_holdout(
        models.build_selected_model(), temporal_train, temporal_test, config.SELECTED_NUMERICAL_COLUMNS)
    print_result("SECONDARY: temporal drift (last 24 cycles)", summary, by_mode)
    rows.append(summary_row("temporal drift", "secondary", temporal_train, temporal_test, summary, by_mode))
    by_mode_tables.append(by_mode.assign(experiment="temporal drift"))

    # 3. Secondary: "before the process" model. Cross-validation first, then the test set.
    before = config.BEFORE_PROCESS_NUMERICAL_COLUMNS
    X, y, groups = data.get_xy(train, before)
    scores = evaluation.cv_pr_auc(models.build_selected_model(before), X, y, groups)
    print(f"\n=== SECONDARY: before the process ({', '.join(before)} + type)")
    print(f"Cross-validation PR-AUC: {scores.mean():.3f} +- {scores.std():.3f} "
          f"(main model {cv_mean} +- {cv_std})")

    _proba, summary, by_mode = evaluation.evaluate_holdout(
        models.build_selected_model(before), train, test, before)
    print_result("SECONDARY: before the process, frozen test set", summary, by_mode)
    row = summary_row("before the process", "secondary", train, test, summary, by_mode)
    row.update({"cv_pr_auc_mean": scores.mean(), "cv_pr_auc_std": scores.std()})
    rows.append(row)
    by_mode_tables.append(by_mode.assign(experiment="before the process"))

    rows[0].update({"cv_pr_auc_mean": cv_mean, "cv_pr_auc_std": cv_std})
    print(f"Drop versus main model on test: PR-AUC {summary['pr_auc'] - rows[0]['pr_auc']:+.3f}, "
          f"recall {summary['recall'] - rows[0]['recall']:+.3f}")

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.METRICS_DIR / "test_results.csv", index=False)
    pd.concat(by_mode_tables).to_csv(config.METRICS_DIR / "test_recall_by_mode.csv", index=False)
    print(f"\nSaved test_results.csv and test_recall_by_mode.csv to {config.METRICS_DIR}")


if __name__ == "__main__":
    main()
