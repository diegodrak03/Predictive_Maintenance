from ucimlrepo import fetch_ucirepo 
from pm.config import RAW_DATA


def main():

    if RAW_DATA.exists():
        print("Data already exists. Skipping download.")

    else:
        # fetch dataset 
        dataset = fetch_ucirepo(id=601)


        # data (as pandas dataframes) 
        df = dataset.data.original

        RAW_DATA.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(RAW_DATA, index=False)

        print(f"Data saved to {RAW_DATA}")
        

if __name__ == "__main__":
    main()

  
