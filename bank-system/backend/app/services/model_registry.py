import pickle
import os
import logging
from pathlib import Path

# Create class configuration for ModelRegistry / ModelLoader
logger = logging.getLogger(__name__)
model_dir = Path(__file__).resolve().parent.parent.parent / 'models'

class ModelRegistry:
    def __init__(self, model_dir=model_dir):
        self.models = {}
        self.model_dir = model_dir
        self.load_models(model_dir)

    def load_models(self, model_dir):
        """Load all models from the specified directory"""
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory '{model_dir}' does not exist. No models loaded.")
            return
        
        try:
            for file in os.listdir(model_dir):
                if file.endswith('.pkl'):
                    model_name = file[:-4]
                    with open(os.path.join(model_dir, file), 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                        logger.info(f"Loaded model '{model_name}' from '{file}'.")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")

    def get_model(self, model_name):
        """Get model by name, return None if not found"""
        return self.models.get(model_name)
    
    def list_models(self):
        """List all available models in the registry"""
        return list(self.models.keys())
    
registry = ModelRegistry()