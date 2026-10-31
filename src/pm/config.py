from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "dataset" / "raw_data" / "raw_data.csv"
PROCESSED_DIR = ROOT / "dataset" / "processed"
RESULTS_DIR = ROOT / "results"

SEED = 42