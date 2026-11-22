"""Where the project's files live. Everything under data/ is gitignored."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # monthly CRMLS exports
COMBINED_PATH = DATA_DIR / "intermediate" / "CRMLSSoldCombined.csv"  # all months stacked (01_exploration)
FILTERED_PATH = DATA_DIR / "intermediate" / "CRMLSSoldFiltered.csv"  # input to Dataset A
DATASET_A_PATH = DATA_DIR / "processed" / "CRMLSSoldDatasetA.csv"  # written by 02b_preprocessing_dataset_a
DISTRICTS_PATH = DATA_DIR / "external" / "school_districts_2425" / "DistrictAreas2425.shp"
MODELS_DIR = DATA_DIR / "models"

RESULTS_DIR = PROJECT_ROOT / "results"  # weekly deliverable tables
ARCHIVE_DIR = RESULTS_DIR / "archive"  # supporting tables: per-month backtests, side experiments
