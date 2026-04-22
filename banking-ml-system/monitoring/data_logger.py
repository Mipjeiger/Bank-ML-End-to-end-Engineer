import pandas as pd
import os
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent
LOG_PATH = BASE_DIR / "data" / "reference" / "Customer-Churn-dataset.csv"

# Create function for logging data
def log_data(data: dict):
    df = pd.DataFrame([data])

    if not LOG_PATH.exists():
        df.to_csv(LOG_PATH, index=False)
    else:
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)