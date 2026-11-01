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

METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

FAILURE_MODE_COLUMNS = ["heat_dissipation_failure", "power_failure", "overstrain_failure", "tool_wear_failure", "random_failure"]

FAILURE_MODE_NAMES = {
        "heat_dissipation_failure": "HDF",
        "power_failure": "PWF",
        "overstrain_failure": "OSF",
        "tool_wear_failure": "TWF",
        "random_failure": "RNF"
}

N_FOLDS = 5

SELECTED_NUMERICAL_COLUMNS = BASIC_NUMERICAL_COLUMNS + PHYSICAL_NUMERICAL_COLUMNS

SELECTED_THRESHOLD = 0.093

THRESHOLDS_TO_REPORT = [0.05, 0.093, 0.1, 0.2, 0.3, 0.4, 0.5]

HDF_TEMP_DIFF_K = 8.6

MIN_PRECISION = 0.5

# Inputs known before machining starts: no process temperature or torque, so no physics features
BEFORE_PROCESS_NUMERICAL_COLUMNS = ["air_temp_k", "rotational_speed_rpm", "tool_wear_min"]

# Cross-validation figures of the selected model and the baseline (decisions.md, sections 18 to 20)
CV_PR_AUC = (0.898, 0.042)
CV_PRECISION = 0.507
CV_RECALL = 0.883
CV_LOGISTIC_PR_AUC = (0.494, 0.045)

MIN_RECALL = 0.75
