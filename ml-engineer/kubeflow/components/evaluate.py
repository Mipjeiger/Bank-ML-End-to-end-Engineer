from kfp import dsl

@dsl.component(base_image="python:3.11")
def evaluate_model(model_dir: str, test_data: str) -> float:
    import pandas as pd
    import pickle
    from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

    # Load test dataset
    df = pd.read_csv(test_data)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Load model
    with open(model_dir, 'rb') as f:
        model = pickle.load(f)
    
    # Predictions with model
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    f1_score_value = f1_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    classification_report = classification_report(y, predictions)
    print(f"Models: {model_dir}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1_score_value:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Classification Report:")
    print(classification_report)

    # Save results
    output_path = '/tmp/evaluation_results.txt'
    with open(output_path, 'w') as f:
        f.write(f"Model: {model_dir}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1_score_value:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write("Classification Report:\n")
        f.write(classification_report)

    return accuracy, f1_score_value, precision, recall, classification_report