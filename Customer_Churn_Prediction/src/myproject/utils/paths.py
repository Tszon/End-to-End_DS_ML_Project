# src/myproject/utils/paths.py
from pathlib import Path

# Resolve project root (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Common directories
CONFIG_PATH =  PROJECT_ROOT / "config.json"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "raw"
CLEANED_DATA = DATA_DIR / "cleaned"

PREPROCESSED_DATA = DATA_DIR / "ML_preprocessed"
FEATURE_DATA = DATA_DIR / "engineered_features"
REPORT_DATA = DATA_DIR / "reports"

MODELS_DIR = PROJECT_ROOT / "models"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
VIS_DIR = PROJECT_ROOT / "reports_app"

# Ensure directories exist
for d in [
    DATA_DIR, RAW_DATA, CLEANED_DATA,
    PREPROCESSED_DATA, FEATURE_DATA, REPORT_DATA,
    MODELS_DIR, 
    NOTEBOOKS_DIR, VIS_DIR
    ]:
    d.mkdir(parents=True, exist_ok=True)
