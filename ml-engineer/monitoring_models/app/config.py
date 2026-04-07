import os

BASE_MODEL_PATH = "/models"

MODEL_FILES = {
    "fraud": [
        "Decision_Tree_Fraud.pkl",
        "KNN_Fraud.pkl",
        "Logistic_Regression_Fraud.pkl",
        "Random_Forest_Fraud.pkl",
        "XGBoost_Fraud.pkl",
    ],
    "marketing": [
        "Decision_Tree_Marketing.pkl",
        "KNN_Marketing.pkl",
        "Logistic_Regression_Marketing.pkl",
        "Random_Forest_Marketing.pkl",
        "XGBoost_Marketing.pkl",
    ],
    "operational": [
        "Decision_Tree_Operational.pkl",
        "KNN_Operational.pkl",
        "Logistic_Regression_Operational.pkl",
        "Random_Forest_Operational.pkl",
        "XGBoost_Operational.pkl",
    ],
}