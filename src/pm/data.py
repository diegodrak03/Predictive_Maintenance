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
        df (pd.DataFrame): The DataFrame with original columns.

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