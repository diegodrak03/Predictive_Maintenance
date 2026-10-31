import pandas as pd

from pm import config


def load_raw():
    """
    Load the raw data from the CSV file specified in the config.

    Returns:
        pd.DataFrame: The raw data as a pandas DataFrame.
    """
    return pd.read_csv(config.RAW_DATA)

def rename_columns(df):
    """
    Rename the columns of the DataFrame to more descriptive names.

    Args:
        df (pd.DataFrame): The DataFrame with original column names.

    Returns:
        pd.DataFrame: The DataFrame with renamed columns.
    """
    df = df.rename(columns=config.COLUMN_RENAMES, errors="raise")
    return df

def add_tool_cycle(df):
    """
    Add a new column 'tool_cycle' to the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame with the renamed columns.

    Returns:
        pd.DataFrame: The DataFrame with the new 'tool_cycle' column.
    """
    df = df.sort_values(by=["uid"])

    df["tool_cycle"] = (df["tool_wear_min"].diff() < 0).cumsum()

    return df

if __name__ == "__main__":
    df = load_raw()
    df = rename_columns(df)
    df = add_tool_cycle(df)
    print(df["tool_cycle"].max())
    min_tool_wear = df.groupby("tool_cycle")["tool_wear_min"].min()
    print()
    print(min_tool_wear.describe())

    summary = df.groupby("tool_cycle").agg(
    n_processes=("uid", "count"),
    min_wear=("tool_wear_min", "min"),
    max_wear=("tool_wear_min", "max"),
    failures=("machine_failure", "sum"),
    )
    print()
    print(summary.tail(25))

    n_cycles = df["tool_cycle"].max() + 1
    last = df[df["tool_cycle"] >= int(n_cycles * 0.8)]
    print()
    print(len(last), last["machine_failure"].sum())
    print()
    print("TWF:", last["tool_wear_failure"].sum())
    print("HDF:", last["heat_dissipation_failure"].sum())
    print("PWF:", last["power_failure"].sum())
    print("OSF:", last["overstrain_failure"].sum())
    print("RNF:", last["random_failure"].sum())
    