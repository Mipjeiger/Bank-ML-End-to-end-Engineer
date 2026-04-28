import numpy as np
from fastapi import APIRouter, HTTPException
from app.schemas.customer import CustomerData, PredictionRequest
from app.services.inference import predict
from app.services.cache import get_cache, set_cache
from app.services.feature_engineering import feature_engineering
from app.monitoring.prometheus import REQUEST_COUNT, PREDICTION_COUNT

# Create router for API
router = APIRouter()

# Create a function enduring risk label based on probability
def _risk_label(probability: float) -> str:
    if probability >= 0.7:
        return "High Risk"
    elif probability >= 0.4:
        return "Medium Risk"
    return "Low Risk"

@router.post("/predict/{model_name}")
def predict_route(model_name: str, data: CustomerData):

    REQUEST_COUNT.inc() # Increment request count for monitoring
    cache_key = f"{model_name}_{data.customer_id}"
    cache_result = get_cache(cache_key)

    if cache_result:
        PREDICTION_COUNT.labels(model=model_name, status="cached").inc()
        return {"model": model_name,
                "cached": True,
                "customer_id": data.customer_id,
                "result": cache_result}
    
    # Try to perform feature engineering and prediction
    try:
        payload = data.dict()
        prediction = predict(model_name, payload)

        result = {
            "model": model_name,
            "customer_id": data.customer_id,
            "probability": float(prediction),
            "prediction": int(np.round(prediction)),
            "risk": _risk_label(prediction),
            "cached": False
        }
        set_cache(cache_key, result) # cache the result for future requests
        PREDICTION_COUNT.labels(model=model_name, status="success").inc()
        return result
    
    except Exception as e:
        PREDICTION_COUNT.labels(model=model_name, status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/batch-predict/{model_name}")
def batch_predict_route(model_name: str, batch_data: list[CustomerData]):
    """Batch prediction for multiple customers API"""

    REQUEST_COUNT.inc()
    results = []

    for data in batch_data:
        try:
            payload = data.dict()
            prediction = predict(model_name, payload)

            result = {
                "customer_id": data.customer_id,
                "probability": float(prediction),
                "prediction": int(np.round(prediction)),
                "risk": _risk_label(prediction),
                "cached": False
            }
            results.append(result)
        except Exception as e:
            results.append({
                "customer_id": data.customer_id,
                "error": str(e)
            })
    
    PREDICTION_COUNT.labels(model=model_name, status="batch").inc()
    return {"model": model_name, "results": results}

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.get("/debug/models")
def debug_models():
    """Debug endpoint - show all models and their status in the registry"""
    from app.services.model_registry import registry
    return {
        "loaded": registry.list_models(),
        "failed": registry.failed_models,
        "total": len(registry.models) + len(registry.failed_models)
    }