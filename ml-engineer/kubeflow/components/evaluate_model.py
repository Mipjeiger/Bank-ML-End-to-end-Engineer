from kfp import dsl
from typing import NamedTuple

# 1. Define the output structure as a Class
class EvaluationMetrics(NamedTuple):
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    report: str

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "xgboost"]
)
def evaluate_model(
    model_path: str, 
    test_data_path: str
) -> EvaluationMetrics: # Use the class name here
    import pandas as pd
    import pickle
    import os
    from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

    # Load test dataset
    df = pd.read_csv(test_data_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Predictions
    predictions = model.predict(X)
    
    # Calculate Metrics
    # We cast to float to ensure compatibility with Kubeflow's metadata
    acc = float(accuracy_score(y, predictions))
    f1 = float(f1_score(y, predictions, average='weighted'))
    prec = float(precision_score(y, predictions, average='weighted'))
    rec = float(recall_score(y, predictions, average='weighted'))
    report_str = str(classification_report(y, predictions))

    print(f"Accuracy: {acc:.4f}")

    # Return the class instance
    return EvaluationMetrics(
        accuracy=acc,
        f1_score=f1,
        precision=prec,
        recall=rec,
        report=report_str
    )