
from pm import config, data, features

def main():

    df = data.load_raw()

    df = data.rename_columns(df)

    df = data.add_tool_cycle(df)

    df = features.add_physics_features(df)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(config.PROCESSED_DATA, index=False)

    print(f"Saved CSV to {config.PROCESSED_DATA}")
    print()
    rows, columns = df.shape
    print(f"{rows} rows X {columns} columns")

    print(df["power_w"].describe())


if __name__ == "__main__":
    main()