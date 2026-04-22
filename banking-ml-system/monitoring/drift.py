import pandas as pd
from pathlib import Path
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import TrainTestFeatureDrift

# Configuration
BASE_DIR = Path(__file__).parent.parent
REFERENCE_PATH = BASE_DIR / "data" / "reference" / "Loan-banking.csv"

def check_drift(data: dict):
    try:
        # Load reference dataset
        reference_df = pd.read_csv(REFERENCE_PATH)
        new_df = pd.DataFrame([data])

        reference_ds = Dataset(reference_df)
        new_ds = Dataset(new_df)

        # Check and result for feature drift
        check = TrainTestFeatureDrift()
        result = check.run(reference_ds, new_ds)

        drifted = result.value.get("drifted_features", [])

        return len(drifted) >0
    
    except Exception as e:
        print(f"Error during drift check: {e}")
        return False