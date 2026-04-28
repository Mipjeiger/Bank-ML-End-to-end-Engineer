from pathlib import Path
import os

# /app/config/config.py in Docker, bank-system/backend/config/config.py in terminal
_THIS_FILE = Path(__file__).resolve()          # .../config/config.py
_BACKEND_DIR = _THIS_FILE.parent.parent        # .../backend/
BASE_DIR = _BACKEND_DIR.parent                 # .../bank-system/

# Docker: models are at /app/models (WORKDIR=/app = backend/)
# Terminal: models are at bank-system/backend/models
MODEL_PATH = Path(os.getenv("MODELS_DIR", str(_BACKEND_DIR / "models")))
DATA_PATH = Path(os.getenv("DATA_PATH", str(BASE_DIR / "database" / "eda_banking.parquet")))

REDIS_HOST = os.getenv("REDIS_HOST", "redis")