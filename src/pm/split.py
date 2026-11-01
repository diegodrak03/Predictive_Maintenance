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


def temporal_split(df, test_fraction = 0.2):
    """
    Splits into train and tests by whole tool cycles.
    The test set is the last "test fraction" of the cycles
    """
    n_cycles = df["tool_cycle"].max() + 1

    n_test_cycles = round(test_fraction * n_cycles)

    first_test_cycle = n_cycles - n_test_cycles

    is_test = df["tool_cycle"] >= first_test_cycle

    return df[~is_test] , df[is_test] 



def cv_splitter():
    """
    Cross-validation splitter used on the training set: 5 folds by whole tool cycles,
    keeping the failure rate similar. Always pass it as cv, with groups=tool_cycle.
    """
    return StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
