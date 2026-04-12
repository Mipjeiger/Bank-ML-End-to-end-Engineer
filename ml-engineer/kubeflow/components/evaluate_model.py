from kfp import dsl
from kfp.dsl import Output, Metrics, Model
from typing import NamedTuple


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "xgboost"]
)
def evaluate_model(
    model_path: Model, 
    test_data_path: str,
    metrics: Output[Metrics]
):
    import pandas as pd
    import pickle
    import os
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    # Load test dataset
    df = pd.read_csv(test_data_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Load model
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

    # kubeflow UI metrics
    metrics.log_metric("accuracy", float(acc))
    metrics.log_metric("f1_score", float(f1))
    metrics.log_metric("precision", float(prec))
    metrics.log_metric("recall", float(rec))