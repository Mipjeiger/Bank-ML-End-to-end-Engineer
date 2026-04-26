import numpy as np
from app.services.model_registry import registry

"""Inference engine for the banking system"""
def predict(model_name, input_data):
    try:
        model = registry.get_model(model_name=model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found in registry.")
        
        features = np.array([list(input_data.values())])
        proba = model.predict_proba(features)[0][1]

        return float(proba[0])
    
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")