import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Configuration
DATA_PATH = Path(__file__).resolve().parents[1] / "feature_repo" / "data" / "eda_banking.parquet"

# Load data
df = pd.read_parquet(DATA_PATH)

# Create features and target
features = [ "CreditScore", "Balance", "NumOfProducts", "IsActiveMember"]

# Encode data
df_encoded = df.copy()
le = LabelEncoder()

for col in df_encoded.select_dtypes(include=["object"]).columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Define X and y
X = df_encoded[features]
y = df_encoded["Exited"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Hyperparameters for GridSearchCV
param_grids = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    },
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1],
        "colsample_bytree": [0.8, 1]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform"],
        "metric": ["euclidean", "manhattan"]
    },
    "Decision Tree": {
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 7],
        "min_samples_leaf": [1, 2, 4]
    }
}

# Train models with GridSearchCV
best_models = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(estimator=model, 
                               param_grid=param_grids[model_name], 
                               cv=5, 
                               n_jobs=-1, 
                               verbose=1)
    grid_search.fit(X, y)
    best_models[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")

    # Save the best model
    joblib.dump(grid_search.best_estimator_, f"../models/marketing_model.pkl")
    print(f"{model_name} model saved as ../models/marketing_model.pkl")