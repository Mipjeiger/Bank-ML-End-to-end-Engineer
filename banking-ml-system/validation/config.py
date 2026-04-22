from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "validation" / "reports"
MODELS_DIR = BASE_DIR / "models"

# Deepchecks configuration
DEEPCHECKS_CONFIG = {
    "suite": "full_suite",  # or "vision_suite", "tabular_suite"
    "save_html": True,
    "save_json": False,
}

# Model tasks
TASKS = ["fraud", "marketing", "operational"]

# Target columns per task
TARGETS = {
    "fraud": "Fraud",
    "marketing": "Exited",
    "operational": "OperationalRiskScore"
}

# Feature columns
FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Complain", "Satisfaction Score",
    "Card Type", "Point Earned", "RiskScore", "BalancePerProduct",
    "AgeRisk", "HighValueCustomer", "LowCreditRisk",
    "ComplainFlag", "LowSatisfaction",
]

CATEGORICAL_COLS = ["Geography", "Gender", "Card Type"]

# Create directories if they don't exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)