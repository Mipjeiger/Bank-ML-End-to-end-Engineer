import pickle
import os
import logging
from pathlib import Path
from config.config import MODEL_PATH

# Create class configuration for ModelRegistry / ModelLoader
logger = logging.getLogger(__name__)
model_dir = Path(MODEL_PATH)

class ModelRegistry:
    def __init__(self, model_dir=model_dir):
        self.models = {}
        self.failed_models = {}
        self.model_dir = model_dir
        self.load_models(model_dir)
        self.print_status()

    def load_models(self, model_dir):
        """Load all models from the specified directory"""
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory '{model_dir}' does not exist. No models loaded.")
            return
        
        # Get all .pkl files
        pkl_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        logger.info(f"Found {len(pkl_files)} models in '{model_dir}'. Attempting to load...")

        for file in pkl_files:
            model_name = file[:-4] # Remove .pkl extension/name to get model name
            try:
                with open(os.path.join(model_dir, file), 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                    logger.info(f"Loaded model {model_name}")
            except Exception as e:
                self.failed_models[model_name] = str(e)
                logger.error(f"Models not found in {model_name}. Can't be loaded: {str(e)}")

    def get_model(self, model_name):
        """Get model by name, return None if not found"""
        if model_name not in self.models:
            logger.warning(f"Model '{model_name}' not found in registry. Available models: {list(self.models.keys())}")
            return None
        return self.models.get(model_name)
    
    def list_models(self):
        """List all available models in the registry"""
        return list(self.models.keys())
    
    def print_status(self):
        """Print load status"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Models loaded: {len(self.models)}/{len(self.models) + len(self.failed_models)}")
        logger.info(f"Successfully loaded models: {self.list_models()}")
        if self.failed_models:
            logger.error(f"Failed models: {len(self.failed_models)}")
            for model, error in self.failed_models.items():
                logger.error(f" - {model}: {error}")
        logger.info(f"{'='*50}\n")
    
registry = ModelRegistry()