import joblib
import os
from app.config import BASE_MODEL_PATH, MODEL_FILES

models = {}

def load_models():
    for category, files in MODEL_FILES.items():
        models[category] = {}
        for file in files:
            path = os.path.join(BASE_MODEL_PATH, file)
            model_name = file.replace(".pkl", "")
            models[category][model_name] = joblib.load(path)

    return models