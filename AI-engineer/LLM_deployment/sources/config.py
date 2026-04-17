import os
from pathlib import Path
from dotenv import load_dotenv

# CONFIGURATION
ENV_PATH = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=ENV_PATH)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PDF_DIR    = BASE_DIR / "Database" / "PDF"
PDF_PATHS = list(PDF_DIR.glob("*.pdf"))
DATA_PATH = BASE_DIR / "Database" / "data" / "eda_banking.parquet"