import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import DATA_PATH

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "ml-engineer" / "models" / "banking_models" / "models"

# Model Registry
MODEL_REGISTRY = {
    "fraud": {
        "Decision Tree"         : "Decision_Tree_Fraud.pkl",
        "KNN"                   : "KNN_Fraud.pkl",
        "Logistic Regression"   : "Logistic_Regression_Fraud.pkl",
        "Random Forest"         : "Random_Forest_Fraud.pkl",
        "XGBoost"               : "XGBoost_Fraud.pkl",
    },
    "marketing": {
        "Decision Tree"         : "Decision_Tree_Marketing.pkl",
        "KNN"                   : "KNN_Marketing.pkl",
        "Logistic Regression"   : "Logistic_Regression_Marketing.pkl",
        "Random Forest"         : "Random_Forest_Marketing.pkl",
        "XGBoost"               : "XGBoost_Marketing.pkl",
    },
    "operational": {
        "Decision Tree"         : "Decision_Tree_Operational.pkl",
        "KNN"                   : "KNN_Operational.pkl",
        "Logistic Regression"   : "Logistic_Regression_Operational.pkl",
        "Random Forest"         : "Random_Forest_Operational.pkl",
        "XGBoost"               : "XGBoost_Operational.pkl",
    },
}

# ── Feature map per problem ───────────────────────────────────
FEATURE_MAP = {
    "fraud": [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember",
        "EstimatedSalary", "RiskScore", "OperationalRiskScore",
    ],
    "marketing": [
        "CreditScore", "Age", "Balance", "EstimatedSalary",
        "NumOfProducts", "IsActiveMember", "Point Earned",
        "Satisfaction Score", "HighValueCustomer", "MarketingScore",
    ],
    "operational": [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember",
        "RiskScore", "BalancePerProduct", "AgeRisk",
        "LowCreditRisk", "OperationalRiskScore",
    ],
}

# ── Target map per problem ────────────────────────────────────
TARGET_MAP = {
    "fraud"       : "Fraud",
    "marketing"   : "MarketingScore",
    "operational" : "OperationalRiskScore",
}

# Function to load a model
def load_model(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        print(f"🔄 Loading model from {pkl_path.name}...")
        return pickle.load(f)

def evaluate_model(problem: str) -> dict:
    """Load all models for a given problem, score them on the parquet dataset.
    and return ranked results with the best model recommendation."""

    from sklearn.metrics import(
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, mean_squared_error, r2_score
    )

    df = pd.read_parquet(DATA_PATH)
    models = MODEL_REGISTRY[problem]
    features = FEATURE_MAP[problem]
    target = TARGET_MAP[problem]

    # fillna missing values with dropna
    availabe_features = [f for f in features if f in df.columns]
    df_clean = df[availabe_features + [target]].dropna()

    X = df_clean[availabe_features]
    y = df_clean[target]

    results = []

    for model_name, pkl_file in models.items():
        pkl_path = MODELS_DIR / pkl_file
        if not pkl_path.exists():
            print(f"⚠️ Model file {pkl_file} not found, skipping {model_name}.")
            continue

        model = load_model(pkl_path)

        try:
            y_pred = model.predict(X)
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)[:, 1]

            # Detect classification vs regression
            is_classifier = len(np.unique(y)) <= 10

            if is_classifier:
                metrics = {
                    "accuracy": round(accuracy_score(y, y_pred), 4),
                    "f1_score": round(f1_score(y, y_pred, average="weighted", zero_division=0), 4),
                    "precision": round(precision_score(y, y_pred, average="weighted", zero_division=0), 4),
                    "recall": round(recall_score(y, y_pred, average="weighted", zero_division=0), 4),
                    "roc_auc": round(roc_auc_score(y, y_prob), 4) if y_prob is not None and len(np.unique(y)) == 2 else None,
                }
                score = metrics["f1_score"]  # Use F1-score for ranking classifiers
            else:
                mse = mean_squared_error(y, y_pred)
                metrics = {
                    "r2_score": round(r2_score(y, y_pred), 4),
                    "mse": round(mse, 4),
                    "rmse": round(np.sqrt(mse), 4),
                }
                score= metrics["r2_score"]  # Use R2-score for ranking regressors

            results.append({
                "model": model_name,
                "metrics": metrics,
                "score": score
            })

        except Exception as e:
            results.append({
                "model": model_name,
                "metrics": {},
                "score": -1,
                "error": str(e)
            })

    # Rank models by score
    results.sort(key=lambda x: x["score"], reverse=True)

    best = results[0] if results else None

    return {
        "problem": problem,
        "target_column": target,
        "features_used": availabe_features,
        "total_models": len(results),
        "ranked_models": results,
        "best_model": best["model"] if best else None,
        "best_score": best["score"] if best else None,
        "recommendation": _recommendation(problem, best["model"] if best else "N/A")
    }

def evaluate_all_problems():
    """Run evaluation across all 3 problems and return overall best per problem."""
    summary = {}
    for problem in MODEL_REGISTRY:
        summary[problem] = evaluate_model(problem)
    return summary

def _recommendation(problem: str, best_model: str) -> str:
    tips = {
        "fraud": (
            f"{best_model} performs best for fraud detection. "
            "For imbalanced fraud data, ensemble models (Random Forest, XGBoost) often excel due to their ability to capture complex patterns."
            "with SMOTE resampling data balancing. SMOTE resampling techniques can further enhance performance by addressing class imbalance."
            "Prioritize recall and ROC-AUC over accuracy to minimize missed fraud cases (false positives class)."
        ),
        "marketing": (
            f"{best_model} leads on marketing score prediction. "
            "Tree-based models handle non-linear interactions between customer features and marketing outcomes well. "
            "HighValueCustomer, Point Earned, and Satisfaction Score are key features to focus on for improving marketing strategies."
            "Use SHAP values to identify top marketing drivers per segment."
        ),
        "operational": (
            f"{best_model} is recommended for operational risk scoring. "
            "Gradient boosting models like XGBoost can capture complex relationships especially in generalized across AgeRisk. "
            "BalancePerProduct, and LowCreditRisk combinations are critical for operational risk. "
            "Ensure periodic revalidation for regulatory compliance."
        )
    }

    return tips.get(problem, f"{best_model} is the best model for {problem}, but no specific recommendation is available.")