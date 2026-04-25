import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from app.schemas import CustomerData, PredictionRequest
from app.inference import ModelInference
from monitoring.drift import check_drift
from monitoring.data_logger import log_data
from validation.config import TARGETS

# Setup FastAPI
app = FastAPI(
    title="Banking ML API",
    description="API for banking Machine Learning Engineer system retrieving on deepchecks monitoring and model inference",
    version="1.0.0"
)

# Initialize framework to API endpoint
@app.get("/root")
def root():
    return {"message": "🚀 Banking ML API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy",
            "message": f"API is up and running. Current time run: {datetime.now().isoformat()}"}

@app.get("/get-available-task")
def get_available_tasks():
    return {"available_task": list(TARGETS.keys())}

# Endpoint for model inference
@app.post("/predict")
def prediction(request: PredictionRequest):
    try:
        # Insantiate inference class
        inference = ModelInference(task=request.task, auto_select_best=True)

        # Convert dict to DataFrame
        data_df = pd.DataFrame([request.data])

        # Make prediction
        result = inference.predict(data_df)

        # Log data
        log_data(request.task, request.data, result)

        # Drift check
        try:
            drift_result = check_drift(request.task, data_df)
            result["drift"] = drift_result
        except Exception as e:
            result["drift"] = f"Drift check failed: {str(e)}"

        return result
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict-ensemble")
def predict_ensemble(request: PredictionRequest):
    try:
        inference = ModelInference(task=request.task, auto_select_best=False)

        data_df = pd.DataFrame([request.data])
        result = inference.predict_ensemble(data_df)

        # Log ensemble prediction
        log_data(request.task, request.data, result)
    
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
