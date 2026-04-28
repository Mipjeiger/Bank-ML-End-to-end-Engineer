import numpy as np
import logging
from app.services.feature_engineering import feature_engineering
from app.services.model_registry import registry

logger = logging.getLogger(__name__)

def predict(model_name: str, input_data: dict):
    """Make prediction using specified model
    
    Args:
        model_name: Name of the model (without .pkl extension)
        input_data: Customer data as a dictionary
        
    Returns:
        float: Probability of churn between 0 and 1
    """
    try:
        if input_data is None:
            raise ValueError("input_data is None - ensure passing request.data, not the pull request object.")
        if not isinstance(input_data, dict):
            raise ValueError(f"input_data should be a dictionary, got {type(input_data).__name__}. Call .model_dump() on Pydantic models")

        model = registry.get_model(model_name=model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found in registry.")
        
        # Transform features
        features = feature_engineering.transform(input_data)

        if features is None or features.size == 0:
            raise ValueError("Feature transformation resulted in empty features. Check input data and feature engineering steps.")
        
        logger.debug(f"Feature shape: {features.shape}")

        # Get prediction
        proba = model.predict_proba(features)[0]
        return float(proba[1]) if len(proba) > 1 else float(proba[0]) # Return probability of positive class (churn)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise ValueError(f"Error during prediction: {str(e)}")
    
def batch_predict(model_name: str, data_list: list[dict]):
    """Batch prediction for multiple customers"""
    try:
        model = registry.get_model(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found in registry.")
        
        features = feature_engineering.transform_batch(data_list)
        probas = model.predict_proba(features)

        return [float(p[1]) if len(p) > 1 else float(p[0]) for p in probas]
    
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise ValueError(f"Error during batch prediction: {str(e)}")