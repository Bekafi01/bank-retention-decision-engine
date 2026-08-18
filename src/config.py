from pathlib import Path
from typing import Optional
import os

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure essential directories exist
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, ARTIFACTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def find_raw_data_path() -> Optional[Path]:
    """Find the raw dataset CSV inside data/raw/."""
    candidates = list(DATA_RAW_DIR.glob("*.csv"))
    if not candidates:
        return None
    # Prefer Customer-Churn-Records.csv if present
    for c in candidates:
        if "customer" in c.name.lower() or "churn" in c.name.lower() or "bank" in c.name.lower():
            return c
    return candidates[0]


RAW_DATA_PATH = find_raw_data_path() or (DATA_RAW_DIR / "Customer-Churn-Records.csv")
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "bank_churn_engineered.parquet"

# Financial & Business Parameters (Banking & Wealth Management)
ANNUAL_NIM_RATE = 0.028  # 2.8% Net Interest Margin generated on customer deposits
VIP_OUTREACH_COST = 150.0  # Cost in €/$ for relationship manager outreach
VIP_SAVE_RATE = 0.65  # Expected retention success rate for VIP outreach
DIGITAL_INCENTIVE_COST = 25.0  # Cost in €/$ for digital campaign/fee-waiver
DIGITAL_SAVE_RATE = 0.30  # Expected retention success rate for digital outreach
HIGH_BALANCE_THRESHOLD = 50000.0  # Minimum balance for High-Net-Worth/VIP segment
DEFAULT_DECISION_THRESHOLD = 0.35  # Calibrated baseline threshold

# Machine Learning Parameters
RANDOM_STATE = 42
CV_SPLITS = 5
TEST_SIZE = 0.20
TARGET_COL = "Exited"
ID_COLS = ["RowNumber", "CustomerId", "Surname"]
