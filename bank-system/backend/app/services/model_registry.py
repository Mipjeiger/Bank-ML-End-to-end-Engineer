import pickle
import os
from pathlib import Path

# Create class configuration for ModelRegistry / ModelLoader
model_dir = Path(__file__).resolve().parent.parent.parent / 'models'

class ModelRegistry:
    def __init__(self, model_dir=model_dir):
        self.models = {}
        self.load_models(model_dir)

    def load_models(self, model_dir):
        for file in os.listdir(model_dir):
            if file.endswith('.pkl'):
                model_name = file[:-4]
                with open(os.path.join(model_dir, file), 'rb') as f:
                    self.models[model_name] = pickle.load(f)

    def get_model(self, model_name):
        return self.models.get(model_name)
    
registry = ModelRegistry()