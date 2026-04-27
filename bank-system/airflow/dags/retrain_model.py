from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.task_group import TaskGroup
import pandas as pd
import joblib
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import redis

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "bank-system" / "data"
MODELS_DIR = BASE_DIR / "bank-system" / "backend" / "models"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

TARGETS = {
    "fraud": "Fraud",
    "marketing": "Exited",
    "operational": "OperationalRiskScore"
}

MODELS_LIST = {
    "LogisticRegression": None,
    "RandomForestClassifier": None,
    "XGBClassifier": None,
    "KNeighborsClassifier": None,
    "DecisionTreeClassifier": None,
}

# DAG definition
default_args = {
    'owner': 'ML-team',
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
    'start_date': datetime(2026, 4, 1),
}

dag = DAG(
    'model_retraining_pipeline',
    default_args=default_args,
    description='Pipeline models to retrain all ML models',
    schedule=None,
    catchup=False
)

# Reusable functions
def load_training_data():
    """Load preprocessed training data"""
    print("📂 Loading training data...")
    processed_file = DATA_DIR / "processed" / "eda_banking_processed.parquet"

    if not processed_file.exists():
        raise FileNotFoundError(f"Processed training data not found at {processed_file}. Please run the preprocessing pipeline first.")

    df = pd.read_parquet(processed_file)
    print(f"✅ Loaded {len(df)} rows.")

    return str(processed_file)

def retrain_model_for_task(task_name: str, **context):
    """Retrain model for a specific task (fraud, marketing, operational)"""
    print(f"🤖 Retraining model for task: {task_name}")

    # Load training data
    training_data_path = load_training_data()
    
    # Load data and assume target column
    df = pd.read_parquet(training_data_path)
    target_col = TARGETS[task_name]

    # Prepare features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Scale features
    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train_resampled)
    X_test_scaled = scalar.transform(X_test)

    # Save features
    features_path = MODELS_DIR / f"{task_name}_features.json"
    with open(features_path, "w") as f: 
        json.dump({"feature_names": X.columns.tolist()}, f)

    # Train and evaluate each models
    results = {}
    for model_name in MODELS_LIST.keys():
        print(f" 🤖 Retraining {model_name}...")

        # Load model
        model_file = MODELS_DIR / f"{model_name}_{task_name}.pkl"
        if not model_file.exists():
            print(f"⚠️ Model file {model_file} not found. Skipping {model_name} for {task_name}.")
            continue

        with open(model_file, "rb") as f:
            model = pickle.load(f)

        # Retrain model
        model.fit(X_train_scaled, y_train_resampled)

        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        f1 = f1_score(y_test, y_pred, average="weighted")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Save retrained model
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "task": task_name,
            "f1_score": float(f1),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "retrained_at": datetime.now().isoformat()
        }

        metadata_file = MODELS_DIR / f"{model_name}_{task_name}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

        results[model_name] = metadata
        print(f"✅ F1={f1:.4f}, Accuracy={accuracy:.4f}")
        
    # Store results in Redis
    redis_client.set(f"models:retrain:{task_name}:latest", json.dumps(results, default=str))

    return results

# Task definitions
load_data_task = PythonOperator(
    task_id='load_training_data',
    python_callable=load_training_data,
    dag=dag
)

# Create tasks for each target
with TaskGroup("retrain_models", dag=dag) as retrain_group:
    for task_name in TARGETS.keys():
        PythonOperator(
            task_id=f'retrain_{task_name}_model',
            python_callable=retrain_model_for_task,
            op_kwargs={"task_name": task_name},
            dag=dag
        )

# Notifty completion
def pipeline_complete():
    """Mark retraining as complete in Redis"""
    print("🎉 Model retraining pipeline complete!")
    redis_client.set("models:retrain:status", "completed")
    redis_client.set("models:retrain:completed_at", datetime.now().isoformat())

completed_task = PythonOperator(
    task_id='pipeline_complete',
    python_callable=pipeline_complete,
    dag=dag
)

# Dependencies
load_data_task >> retrain_group >> completed_task