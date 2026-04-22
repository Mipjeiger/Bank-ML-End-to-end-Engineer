from pathlib import Path
from feast import FileSource
import pandas as pd

# Create configuration path
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "eda_banking.parquet"

if DATA_DIR:
    print(f"Data file found at: {DATA_DIR}")
    df = pd.read_parquet(DATA_DIR)
    print(df.head())
else:
    print("Data file not found. Please check the path and ensure the file exists.")

# Define the file source for Feast
data_source = FileSource(
    path=str(DATA_DIR),
    timestamp_field="event_timestamp"
)