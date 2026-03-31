import mlflow
import mlflow.sklearn
import joblib
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pathlib import Path
import json
import pandas as pd

# 1. CONFIGURATION
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "fraud_detection_results" / "models"
METRICS_DIR = BASE_DIR / "models" / "fraud_detection_results" / "metrics" / "complete_result_metrics.json"
# Mlflow configuration
EXPERIMENT_NAME = "Prediction_ML_Models_Banking_Project"
DATASET_VERSION = "fraud_v1"

mlflow.set_tracking_uri("http://localhost:5015")
mlflow.set_experiment(EXPERIMENT_NAME)

# 2. Load metrics
with open(METRICS_DIR, "r") as f:
    metrics = json.load(f)

# Helpers to parse model name
def parse_model_name(filename: str):
    """Example: 'Random_Forest_Fraud.pkl -> ("Random_Forest", "Fraud/Marketing/Operational")"""
    name = filename.replace(".pkl", "")
    parts = name.split("_")

    task = parts[-1] # Last part is the task
    algorithm = " ".join(parts[:-1])

    return algorithm, task

def get_metrics(metrics_data, task, algorithm):
    return metrics_data.get(algorithm, {}).get(task, {})

def create_dummy_input():
    """Needed to infer model signature for MLflow"""
    return pd.DataFrame([[0]*10], columns=[f"feature_{i}" for i in range(10)])

# 3. Main loop models
for model_file in MODEL_DIR.glob("*.pkl"):
    algorithm, task = parse_model_name(model_file.name)
    model_metrics = get_metrics(metrics, task, algorithm)
    print("Model directory:", MODEL_DIR)
    print("Files Found:", list(MODEL_DIR.glob("*.pkl")))

    if not model_metrics:
        print(f"Skipping {model_file.name} - no metrics found for {algorithm} on {task}")
        continue

    # Load model
    model = joblib.load(model_file)

    # Dummy input
    X_sample = create_dummy_input()
    # Infer signature for MLflow
    try:
        signature = infer_signature(X_sample, model.predict(X_sample))
    except Exception:
        signature = None

    # Log Params on MLflow
    with mlflow.start_run(run_name=f"{algorithm}_{task}") as run:

        run_id = run.info.run_id

        # log parameters
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("task", task)
        mlflow.log_param("dataset_version", DATASET_VERSION)
        mlflow.log_param("model_file", model_file.name)

        # Log models on MLflow
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)

        # Log metrics on MLflow
        if model_metrics:
            for k, v in model_metrics.items():
                mlflow.log_metric(k, float(v))

        # Log artifacts on MLflow
        mlflow.log_artifact(str(METRICS_DIR))

        # Tagging the run with metadata
        mlflow.set_tags({
            "project": "Banking_Project",
            "stage": "staging",
            "model_type": algorithm,
            "task": task,
        })
        
        # Register the model in MLflow Model Registry
        model_name = f"{algorithm}_{task}_model"

        try:
            mlflow.register_model(
                model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
                name=model_name
            )
            print(f"Registered model {model_name} in MLflow Model Registry")
        except Exception as e:
            print(f"Failed to register model {model_name}: {e}")

        print(f"Logged {algorithm} for {task} with metrics: {model_metrics}")