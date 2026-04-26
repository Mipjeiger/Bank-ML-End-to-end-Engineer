import numpy as np
from fastapi import FastAPI, HTTPException
from app.schemas.customer import CustomerData
from app.services.inference import predict
from app.services.cache import get_cache, set_cache
from app.services.feature_engineering import feature_engineering
from app.monitoring.prometheus import REQUEST_COUNT

# Create router for API
router = FastAPI(title="Banking System API",
                 description="API for bank system with model inference and caching, involving feature engineering and deployment monitoring.",
                 version="1.0.0")

@router.post("/predict/{model_name}")
def predict_route(model_name: str, data: CustomerData):
    # Increment request count for monitoring
    REQUEST_COUNT.inc()

    cache_key = f"{model_name}_{data.customer_id}"
    cache_result = get_cache(cache_key)
    if cache_result:
        return {"model": model_name,
                "cached": True,
                "customer_id": data.customer_id,
                "result": cache_result}
    
    # Try to perform feature engineering and prediction
    try:
        payload = data.dict()
        features = feature_engineering(payload)
        result = {
            "model": model_name,
            "probability": predict(model_name, features),
            "risk": np.where(predict(model_name, features) > 0.7, "High Risk", "Low Risk")
        }

        set_cache(cache_key, result)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))