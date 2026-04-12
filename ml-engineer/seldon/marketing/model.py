import joblib
import numpy as np
import os

class MarketingModel:
    def __init__(self):
        self.models = {}

    def load(self):
        base_path = "/models"

        self.models = {
            "decision_tree": joblib.load(os.path.join(base_path, "Decision_Tree_Marketing.pkl")),
            "logistic_regression": joblib.load(os.path.join(base_path, "Logistic_Regression_Marketing.pkl")),
            "knn": joblib.load(os.path.join(base_path, "KNN_Marketing.pkl")),
            "random_forest": joblib.load(os.path.join(base_path, "Random_Forest_Marketing.pkl")),
            "xgboost": joblib.load(os.path.join(base_path, "XGBoost_Marketing.pkl")),
        }

    def predict(self, X, names=None):
        X = np.array(X)

        results = {}
        for name, model in self.models.items():
            results[name] = model.predict(X).tolist()

        return results