import os
import shutil
import json

BASE_SRC = "../../ml-engineer/models/banking_models/models"
BASE_DST = "../../ml-engineer/models"

MODEL_MAP = {
    "Fraud": "fraud",
    "Marketing": "marketing",
    "Operational": "operational"
}

ALGO_MAP = {
    "Decision_Tree": "decision_tree",
    "Random_Forest": "random_forest",
    "Logistic_Regression": "logistic_regression",
    "KNN": "knn",
    "XGBoost": "xgboost"
}

def create_model_settings(name, impl):
    return {
        "name": name,
        "implementation": impl
    }

for file in os.listdir(BASE_SRC):
    if not file.endswith(".pkl"):
        continue

    parts = file.replace(".pkl", "").split("_")

    algo = "_".join(parts[:-1])
    domain = parts[-1]

    algo_clean = ALGO_MAP.get(algo)
    domain_clean = MODEL_MAP.get(domain)

    if not algo_clean or not domain_clean:
        print(f"Skipping {file}")
        continue

    dst_dir = os.path.join(BASE_DST, domain_clean, algo_clean)
    os.makedirs(dst_dir, exist_ok=True)

    # copy model
    src_path = os.path.join(BASE_SRC, file)
    dst_model_path = os.path.join(dst_dir, "model.pkl")
    shutil.copy(src_path, dst_model_path)

    # create model-settings.json
    impl = "mlserver_sklearn.SKLearnModel"
    if "xgboost" in algo_clean:
        impl = "mlserver_xgboost.XGBoostModel"

    settings = create_model_settings(f"{domain_clean}-{algo_clean}", impl)

    with open(os.path.join(dst_dir, "model-settings.json"), "w") as f:
        json.dump(settings, f, indent=2)

    print(f"Processed {file}")