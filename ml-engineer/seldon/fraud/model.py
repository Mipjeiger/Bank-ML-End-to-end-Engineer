import joblib
import numpy as np
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudModel:
    def __init__(self):
        self.models = {}
        self.load()

    def load(self):
        base_path = "/models"
        model_files = {
            "decision_tree": "Decision_Tree_Fraud.pkl",
            "logistic_regression": "Logistic_Regression_Fraud.pkl",
            "knn": "KNN_Fraud.pkl",
            "random_forest": "Random_Forest_Fraud.pkl",
            "xgboost": "XGBoost_Fraud.pkl",
        }

        for name, filename in model_files.items():
            full_path = os.path.join(base_path, filename)
            if os.path.exists(full_path):
                logger.info(f"Loading model: {name} from {full_path}")
                self.models[name] = joblib.load(full_path)
            else:
                logger.error(f"Model file not found: {full_path}")

    def predict(self, X, names=None):
        """Seldon Core calls this method.
            X is usually a numpy array or a list of lists.        
        """
        try:
            # Ensure X is in the correct format for prediction
            data = np.array(X)
            results = {}

            for name, model in self.models.items():
                # Get predictions and convert to list for JSON serialization
                results[name] = model.predict(data).tolist()

            return results
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {"error": str(e)}