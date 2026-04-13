import joblib
import numpy as np
import os

class FraudModel:
    def __init__(self):
        self.models = {}
        self.load()

    def load(self):
        base_path = "/models"
        self.models = {
            "decision_tree": joblib.load(os.path.join(base_path, "Decision_Tree_Fraud.pkl")),
            "logistic_regression": joblib.load(os.path.join(base_path, "Logistic_Regression_Fraud.pkl")),
            "knn": joblib.load(os.path.join(base_path, "KNN_Fraud.pkl")),
            "random_forest": joblib.load(os.path.join(base_path, "Random_Forest_Fraud.pkl")),
            "xgboost": joblib.load(os.path.join(base_path, "XGBoost_Fraud.pkl")),
        }

    def predict(self, X, names=None):
        X = np.array(X)
        results = {}
        for name, model in self.models.items():
            results[name] = model.predict(X).tolist()

        return results