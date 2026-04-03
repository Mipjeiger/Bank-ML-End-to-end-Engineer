import os
from pathlib import Path
from dotenv import load_dotenv

# CONFIGURATION
ENV_PATH = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=ENV_PATH)

HF_TOKEN = os.getenv('HF_TOKEN')
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "Database" / "data" / "eda_banking.parquet"
PDF_DIR    = BASE_DIR / "Database" / "PDF"
REPORT_MD = BASE_DIR / "ml-engineer" / "notebooks" / "Banking_llm_insights_report.md"
MODELS_DIR = BASE_DIR / "ml-engineer" / "models" / "banking_models" / "models"

PDF_PATHS = list(PDF_DIR.glob("*.pdf"))

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4
MAX_TOKENS = 512
TEMPERATURE = 0.3