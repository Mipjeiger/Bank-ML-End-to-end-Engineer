import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

# Configuration
FEATURES = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", 
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", 
            "Complain", "Satisfaction Score", "Card Type", "Point Earned", 
            "RiskScore", "BalancePerProduct", "AgeRisk", "HighValueCustomer", 
            "LowCreditRisk", "ComplainFlag", "LowSatisfaction"]

MODEL_NAMES = ["LogisticRegression", "RandomForestClassifier", "XGBClassifier", 
               "KNeighborsClassifier", "DecisionTreeClassifier"]

# Create class for inference models
class ModelInference:
    def __init__(self, task: str, auto_select_best: bool = True):
        """tasl: "fraud", "marketing", "operational
        auto_select_best: If True, uses highest F1 score model"""