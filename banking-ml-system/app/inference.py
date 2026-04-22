import joblib
import pandas as pd
import random
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
METADATA_FILE = MODELS_DIR / "training_metadata.json"
FEATURES = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", 
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", 
            "Complain", "Satisfaction Score", "Card Type", "Point Earned", 
            "RiskScore", "BalancePerProduct", "AgeRisk", "HighValueCustomer", 
            "LowCreditRisk", "ComplainFlag", "LowSatisfaction"]

MODEL_NAMES = ["LogisticRegression", "RandomForestClassifier", "XGBClassifier", 
               "KNeighborsClassifier", "DecisionTreeClassifier"]
CATEGORY_COLS = ["Geography", "Gender", "Card Type"]

# Create class for inference models
class ModelInference:
    def __init__(self, task: str, auto_select_best: bool = True):
        """tasl: "fraud", "marketing", "operational
        auto_select_best: If True, uses highest F1 score model."""
        self.task = task
        self.scaler = StandardScaler()

        if auto_select_best:
            # Load metadata and select best model
            best_model = self._get_best_model()
            print(f"Selected best model: {self.model_name} for task: {self.task}")
            self.model_name = best_model
        else:
            self.model_name = None

        # Load the selected model
        self.model_path = MODELS_DIR / f"{self.model_name}_{task}.pkl"
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = joblib.load(self.model_path)

        # Load metadata for this model
        self.metadata = self._load_metadata()

    def _get_best_model(self) -> str:
        """Get model with highest F1 score for the task."""
        best_model = None
        best_f1 = -1

        # Check all metadata files for this task
        for model_name in MODEL_NAMES:
            metadata_file = MODELS_DIR / f"{model_name}_{self.task}_metadata.json"

            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    f1_score = metadata.get("f1_score", 0)

                    if f1_score > best_f1:
                        best_f1 = f1_score
                        best_model = model_name
        
        if best_model is None:
            raise FileNotFoundError(f"No metadata found for task: {self.task}")
        
        return best_model
    
    def _load_metadata(self) -> dict:
        """Load metadata for the selected model."""
        metadata_file = MODELS_DIR / f"{self.model_name}_{self.task}_metadata.json"

        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                return json.load(f)
        return {}
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data"""
        X = data[FEATURES].copy()

        for col in CATEGORY_COLS:
            if col in X.columns:
                X[col] = LabelEncoder().fit_transform(X[col])

        return X
    
    def predict(self, data: pd.DataFrame) -> dict:
        """Make prediction with best model"""
        df = pd.DataFrame([data], columns=FEATURES)
        X = self.preprocess(df)
        X_scaled = self.scaler.fit_transform(X)
        
        # Predict probabilities
        prediction = self.model.predict(X_scaled)[0]
        confidence = None

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_scaled)[0]
            confidence = float(max(proba))

        return {
            "task": self.task,
            "model_used": self.model_name,
            "prediction": int(prediction),
            "confidence": confidence,
            "model_f1_score": self.metadata.get("f1_score"),
            "model_accuracy": self.metadata.get("accuracy"),
            "model_precision": self.metadata.get("precision"),
            "model_recall": self.metadata.get("recall"),
        }
    
    def predict_ensemble(self, data: pd.DataFrame) -> dict:
        """Ensemble prediction - all models vote"""
        X = self.preprocess(data)
        X_scaled = self.scaler.fit_transform(X)

        predictions = {}
        votes = []

        for model_name in MODEL_NAMES:
            model_path = MODELS_DIR / f"{model_name}_{self.task}.pkl"
            metadata_path = MODELS_DIR / f"{model_name}_{self.task}_metadata.json"

            if model_path.exists():
                model = joblib.load(model_path)
                pred = model.predict(X_scaled)[0]

                # Get F1 score from metadata
                f1 = None
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        meta = json.load(f)
                        f1 = meta.get("f1_score")

                predictions[model_name] = {
                    "prediction": int(pred),
                    "f1_score": f1
                }
                votes.append(pred)
        
        # Majority vote
        ensembled_pred = 1 if sum(votes) > len(votes) / 2 else 0

        return {
            "task": self.task,
            "individual_predictions": predictions,
            "ensembled_prediction": ensembled_pred,
            "confidence": sum(votes) / len(votes),
            "num_models": len(predictions)
        }