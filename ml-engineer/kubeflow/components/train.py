import pickle
from kfp import dsl
from typing import Dict, Any

@dsl.component(base_image="python:3.11")
def train_model(input_data: str, model_dir: str = '/app/ml-engineer/models/banking_models/models') -> str:
    import pandas as pd
    import pickle
    from pathlib import Path

    # Load dataset
    df = pd.read_csv(input_data)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    models_path = Path(model_dir)
    results = {}

    for model_file in models_path.glob("*.pkl"):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)

        # Predictions with model
        predictions = model.predict(X)
        accuracy = (predictions == y).sum() / len(y)
        results[model_file.stem] = accuracy
        print(f"Model: {model_file.stem}, Accuracy: {accuracy:.4f}")

    # Save results
    output_path = '/tmp/model_results.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    return output_path