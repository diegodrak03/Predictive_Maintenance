from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "dataset" / "raw_data" / "raw_data.csv"
PROCESSED_DIR = ROOT / "dataset" / "processed_data"
PROCESSED_DATA = PROCESSED_DIR / "processed_data.csv"
RESULTS_DIR = ROOT / "results"

SEED = 42

COLUMN_RENAMES = {
        "UID": "uid",
        "Product ID": "product_id",
        "Type": "type",
        "Air temperature": "air_temp_k",
        "Process temperature": "process_temp_k",
        "Rotational speed": "rotational_speed_rpm",
        "Torque": "torque_nm",
        "Tool wear": "tool_wear_min",
        "Machine failure": "machine_failure",
        "TWF": "tool_wear_failure",
        "HDF": "heat_dissipation_failure",
        "PWF": "power_failure",
        "OSF": "overstrain_failure",
        "RNF": "random_failure"
}

BASIC_NUMERICAL_COLUMNS = ["air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]

PHYSICAL_NUMERICAL_COLUMNS = ["temp_diff_k", "power_w", "overstrain_min_nm"]

CATEGORICAL_COLUMNS = ["type"]

LABEL = "machine_failure"

GROUPS = "tool_cycle"