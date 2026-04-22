from datetime import datetime
from fastapi import FastAPI
from app.schemas import CustomerData
from app.inference import ModelInference
from monitoring.drift import check_drift
from monitoring.data_logger import log_data

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

# Endpoint for model inference
@app.post("/predict")
def prediction(data: CustomerData):
    data_dict = data.dict() # Define data dict
    log_data(data_dict) # Log incoming data for monitoring

    # Drift flag
    drift_flag = check_drift(data_dict)

    # Check for drift
    if check_drift(data_dict):
        return {"warning": "Data drift detected. Model performance may be affected."}
    
    # Prediction to model inference
    prediction = ModelInference.predict(data_dict)

    return {
        "prediction": prediction,
        "drift_flag": drift_flag
    }