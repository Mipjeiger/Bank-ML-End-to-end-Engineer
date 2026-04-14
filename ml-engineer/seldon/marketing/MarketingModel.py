import joblib
import numpy as np
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketingModel:
    def __init__(self):
        self.models = {}
        self.load()

    def load(self):
        base_path = "/models"
        model_files = {
            "decision_tree": "Decision_Tree_Marketing.pkl",
            "logistic_regression": "Logistic_Regression_Marketing.pkl",
            "knn": "KNN_Marketing.pkl",
            "random_forest": "Random_Forest_Marketing.pkl",
            "xgboost": "XGBoost_Marketing.pkl",
        }

        for name, filename in model_files.items():
            full_path = os.path.join(base_path, filename)
            if os.path.exists(full_path):
                logger.info(f"Loading model: {name} from {full_path}")
                self.models[name] = joblib.load(full_path)
            else:
                logger.error(f"Model file not found: {full_path}")

    def predict(self, X, names=None):
        """Seldon core calls this method.
            X is usually a numpy array or a list of lists."""
        try:
            data = np.array(X)
            results = {}
            
            for name, model in self.models.items():
                results[name] = model.predict(data).tolist()

            return results
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {"error": str(e)}