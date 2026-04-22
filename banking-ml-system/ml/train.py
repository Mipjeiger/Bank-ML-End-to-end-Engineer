import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from validation.deepchecks_runner import run_full_validation
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "eda_banking.parquet"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "validation" / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TARGETS = {
    "fraud": "Fraud",
    "marketing": "Exited",
    "operational": "OperationalRiskScore"
}
MODELS = {
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "XGBClassifier": XGBClassifier(n_estimators=100),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42)
}

# Create function for training the model
def train():
    # Load dataset
    df = pd.read_parquet(DATA_DIR)

    # Encode categorical variables
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Train for each target
    for task, target in TARGETS.items():
        print(f"🤖 TASK {task.upper()}")

        # Define X and y
        X = df.drop(columns=[target])
        y = df[target]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # SMOTE data
        X_train_resampled, y_train_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test)

        # Create param_grid for each model
        param_grid = {
            "LogisticRegression": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"],
                "max_iter": [100, 200, 300]
            },
            "RandomForestClassifier": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            },
            "XGBClassifier": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2]
            },
            "KNeighborsClassifier": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            },
            "DecisionTreeClassifier": {
                "max_depth": [10, 15, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }
        }

        # Train and validate each models
        for name, model in MODELS.items():
            print(f"Training {name}...")

            # GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(model, 
                                    param_grid[name],
                                    cv=5,
                                    scoring="f1_weighted",
                                    n_jobs=-1,
                                    verbose=1)
            grid_search.fit(X_train_scaled, y_train_resampled)
            best_model = grid_search.best_estimator_

            # Evaluate
            score_f1 = f1_score(y_test, best_model.predict(X_test_scaled), average="weighted")
            score_accuracy = accuracy_score(y_test, best_model.predict(X_test_scaled))
            score_precision = precision_score(y_test, best_model.predict(X_test_scaled), average="weighted")
            score_recall = recall_score(y_test, best_model.predict(X_test_scaled), average="weighted")
            print(f"Best F1 Score for {name}: {score_f1:.4f}")
            print(f"Best Accuracy for {name}: {score_accuracy:.4f}")
            print(f"Best Precision for {name}: {score_precision:.4f}")
            print(f"Best Recall for {name}: {score_recall:.4f}")

            # Prepare data for deepchecks (unscaled, original features names)
            train_df = pd.DataFrame(X_train_resampled, columns=X.columns)
            train_df["target"] = y_train_resampled

            test_df = pd.DataFrame(X_test, columns=X.columns)
            test_df["target"] = y_test

            # Run deepchecks validation
            print(f"Running Deepchecks validation for {name}...")
            report_path = run_full_validation(train_df, test_df, best_model)
            print(f"📊 Report saved to: {report_path}")

            # Save the model
            filename = f"{name}_{task}.pkl"
            model_path = MODELS_DIR / filename
            joblib.dump(best_model, model_path)
            print(f"✅ Model saved to: {model_path}\n")

# Usage
if __name__ == "__main__":
    train()