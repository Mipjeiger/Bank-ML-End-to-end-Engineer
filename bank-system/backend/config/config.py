from pathlib import Path
import pandas as pd

# Create configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / 'backend' / 'models'
REDIS_HOST = 'redis'
DATA_PATH = BASE_DIR / 'database' / 'eda_banking.parquet'