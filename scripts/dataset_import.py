from ucimlrepo import fetch_ucirepo 
import pandas as pd
from pathlib import Path
from pm.config import RAW_DATA

arx = RAW_DATA

if arx.exists():
    print("Data already exists. Skipping download.")

else:
    # fetch dataset 
    dataset = fetch_ucirepo(id=601)


    # data (as pandas dataframes) 
    x = dataset.data.original

    x.to_csv(arx, index=False)

    print(f"Data saved to {arx}")
    
    

  
