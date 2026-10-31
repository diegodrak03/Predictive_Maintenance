import numpy as np

from pm import config

def add_physics_features(df):
    """
    Adds the following 3 columns:   temp_diff_k             =   process temp - air temp
                                    power_w                 =   torque * rotational speed    
                                    overstrain_min_nm       =   toolwear * torque

    Args:
        df (pd.DataFrame): The DataFrame with the renamed columns.
    
    Returns:
        pd.DataFrame: The DataFrame with the new columns.                                    
    """
    df = df.copy()

    df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]

    omega = (df["rotational_speed_rpm"] * np.pi) / 30
    df["power_w"] = df["torque_nm"] * omega

    df["overstrain_min_nm"] = df["tool_wear_min"] * df["torque_nm"]

    return df
