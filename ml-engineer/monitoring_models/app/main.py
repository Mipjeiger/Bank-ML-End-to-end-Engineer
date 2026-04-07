from fastapi import FastAPI, HTTPException
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from pydantic import BaseModel, Field
from typing import List

from app.model_loader import load_models
from app.predictor import predict
from app.preprocessing import preprocess_input

# Execute the model loading function at startup
app = FastAPI(title="ML Model Monitoring API")

# Global model container
models = {}

# Load models at startup
@app.on_event("startup")
def startup_event():
    global models
    models = load_models()
    print(f"Loaded models: {list(models.keys())}")

# Define request schema
class PredictionRequest(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: bool
    IsActiveMember: bool
    EstimatedSalary: float
    Exited: bool
    Complain: bool
    SatisfactionScore: int
    CardType: str
    PointEarned: int
    RiskScore: int
    BalancePerProduct: float
    AgeRisk: bool
    HighValueCustomer: bool
    LowCreditRisk: bool
    ComplainFlag: bool
    LowSatisfaction: bool
    

# Endpoint for making predictions
@app.get("/")
def home():
    return {"status": "Model monitoring API is running"}

@app.post("/predict/{category}/{model_name}")
def run_prediction(category: str, model_name: str, request: PredictionRequest):
    try:
        # Validate category and model
        if category not in models:
            raise HTTPException(status_code=404, detail="Invalid category")
        
        if model_name not in models[category]:
            raise HTTPException(status_code=404, detail="Model not found in category")
        
        # Convert pydantic -> dict
        raw_data = request.dict()

        processed_data = preprocess_input(raw_data)
        result = predict(models, category, model_name, processed_data)

        return {"category": category,
                "prediction": result,
                "model": model_name,}

    except Exception as e:
        return {"error": str(e)}

# Prometheus metrics endpoint
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)