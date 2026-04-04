import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import DATA_PATH, HF_TOKEN, MODEL_ID
from src.reasoning import reasoning_engine
from huggingface_hub import InferenceClient

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "ml-engineer" / "models" / "banking_models" / "models"
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

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

# Features map per problem
RAW_FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Complain", "Satisfaction Score",
    "Card Type", "Point Earned", "RiskScore", "BalancePerProduct",
    "AgeRisk", "HighValueCustomer", "LowCreditRisk",
    "ComplainFlag", "LowSatisfaction",
]

# ── Categorical columns that need encoding ────────────────────
CATEGORICAL_COLS = ["Geography", "Gender", "Card Type"]

# ── Known category values (from your training data) ──────────
CATEGORY_VALUES = {
    "Geography" : ["France", "Germany", "Spain"],
    "Gender"    : ["Female", "Male"],
    "Card Type" : ["DIAMOND", "GOLD", "PLATINUM", "SILVER"],
}

# ── Target map per problem ────────────────────────────────────
TARGET_MAP = {
    "fraud"       : "Fraud",
    "marketing"   : "MarketingScore",
    "operational" : "OperationalRiskScore",
}

# Preprocessing - encode categorical features to match training
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns exactly as they were during training.
    Drops original categorical cols and replaces with one-hot encoded columns for each category value."""
    df = df.copy()

    # Label encode Gender (Binary: Female=0, Male=1)
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Female":0, "Male": 1})

    # One-hot encode Geography
    if "Geography" in df.columns:
        for val in CATEGORY_VALUES["Geography"]:
            col_name = f"Geography_{val}"
            df[col_name] = (df["Geography"] == val).astype(int)
        df.drop(columns=["Geography"], inplace=True)

    # One-hot encode Card Type
    if "Card Type" in df.columns:
        for val in CATEGORY_VALUES["Card Type"]:
            col_name = f"CardType_{val}"
            df[col_name] = (df["Card Type"] == val).astype(int)
        df.drop(columns=["Card Type"], inplace=True)

    return df

# Get exact feature names from saved model
def get_model_features(model) -> list[str] | None:
    # Sklearn pipeline
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            features = get_model_features(step)
            if features is not None:
                return features
            
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
     
    if hasattr(model, "get_booster"):
        try:
            return model.get_booster().feature_names
        except Exception:
            pass

    if hasattr(model, "feature_names"):
        fn = model.feature_names
        if fn is not None:
            return list(fn)
        
    return None

# Function to load a model
def load_model(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        print(f"🔄 Loading model from {pkl_path.name}...")
        return pickle.load(f)

# Inspect endpoint helper
def inspect_model_features(problem: str) -> dict:
    results = {}
    for model_name, pkl_file in MODEL_REGISTRY[problem].items():
        pkl_path = MODELS_DIR / pkl_file
        if not pkl_path.exists():
            results[model_name] = {"error": "Model file not found"}
            continue

        model = load_model(pkl_path)
        features = get_model_features(model)
        n_feat = getattr(model, "n_features_in_", None)

        results[model_name] = {
            "n_features_expected": n_feat,
            "feature_names": features
        }
    
    return results

# Core evaluation function
def evaluate_models(problem: str,
                   df_summary: str = "",
                   retriever = None,
                   with_reasoning: bool = False
                   ) -> dict:
    """Load all models for a given problem, score them on the parquet dataset.
    and return ranked results with the best model recommendation."""

    from sklearn.metrics import(
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, mean_squared_error, r2_score
    )

    df = pd.read_parquet(DATA_PATH)
    target = TARGET_MAP[problem]

    # Keep only raw feature cols + target, drop Exited to avoid data leakage
    keep_cols = [c for c in RAW_FEATURES if c in df.columns] + [target]
    df = df[keep_cols].dropna()

    # Preprocess - encode categorical features to match training
    df_encoded = preprocess(df.drop(columns=[target]))
    y = df[target].reset_index(drop=True)

    results = []

    for model_name, pkl_file in MODEL_REGISTRY[problem].items():
        pkl_path = MODELS_DIR / pkl_file
        if not pkl_path.exists():
            results.append({
                "model": model_name, "metrics": {}, "score": -1,
                "error": f"File not found: {pkl_file}"
            })
            continue

        model = load_model(pkl_path)

        # Get exact features expected by the model
        model_features = get_model_features(model)

        if model_features:
            missing = [f for f in model_features if f not in df_encoded.columns]
            if missing:
                results.append({
                    "model": model_name,
                    "metrics": {},
                    "score": -1,
                    "feature_missing": missing,
                    "error": f"Missing features for {model_name}: {missing}"
                })
                continue
            X = df_encoded[model_features]
        else:
            X = df_encoded  # fallback to all features if we can't determine expected features

        try:
            y_pred = model.predict(X)
            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X)[:, 1]
                except Exception:
                    pass

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
                "model"         : model_name,
                "features_used" : list(X.columns),
                "n_features"    : len(X.columns),
                "metrics"       : metrics,
                "score"         : score,
            })

        except Exception as e:
            results.append({
                "model"         : model_name,
                "features_used" : list(X.columns),
                "metrics"       : {},
                "score"         : -1,
                "error"         : str(e),
            })

    # Rank models by score
    results.sort(key=lambda x: x["score"], reverse=True)
    best = next((r for r in results if r["score"] != -1), None)


    reasoning = reasoning_engine(
        problem       = problem,
        best_model    = best["model"],
        ranked_models = results,
        metrics       = best["metrics"],
        df_summary    = df_summary,
    ) if best else {}

    return {
        "problem"        : problem,
        "target_column"  : target,
        "total_models"   : len(results),
        "ranked_models"  : results,
        "best_model"     : best["model"] if best else None,
        "best_score"     : best["score"] if best else None,
        "recommendation" : _recommendation(problem, best["model"] if best else "N/A"),
        "reasoning"      : reasoning,   # populated below if requested
    }

def evaluate_all_problems(df_summary: str = "") -> dict:
    """Run evaluation across all 3 problems and return overall best per problem."""
    return {p: evaluate_models(problem=p, df_summary=df_summary) for p in MODEL_REGISTRY}

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