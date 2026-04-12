import joblib
import numpy as np
import os

class OperationalModel:
    def __init__(self):
        self.models = {}

    def load(self):
        base_path = "/models"

        self.models = {
            "decision_tree": joblib.load(os.path.join(base_path, "Decision_Tree_Operational.pkl")),
            "logistic_regression": joblib.load(os.path.join(base_path, "Logistic_Regression_Operational.pkl")),
            "knn": joblib.load(os.path.join(base_path, "KNN_Operational.pkl")),
            "random_forest": joblib.load(os.path.join(base_path, "Random_Forest_Operational.pkl")),
            "xgboost": joblib.load(os.path.join(base_path, "XGBoost_Operational.pkl")),
        }

    def predict(self, X, names=None):
        X = np.array(X)

        results = {}
        for name, model in self.models.items():
            results[name] = model.predict(X).tolist()

        return results