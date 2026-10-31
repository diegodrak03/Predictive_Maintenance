from sklearn.model_selection import StratifiedGroupKFold

from pm import config


def split_train_test(df):
    """
    Split into train and test by whole tool cycles, keeping the failure rate similar.
    The test set is about 20 % of the cycles.
    """
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    train_idx, test_idx = next(sgkf.split(df, df["machine_failure"], groups=df["tool_cycle"]))

    return df.iloc[train_idx], df.iloc[test_idx]





