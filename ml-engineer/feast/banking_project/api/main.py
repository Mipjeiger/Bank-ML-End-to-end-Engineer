from fastapi import FastAPI
from schemas import Request
from service import predict_fraud, predict_marketing, predict_operational

# Initialize FastAPI app
app = FastAPI(title="Banking Prediction API",
              description="API for predicting machine learning case",
              version="1.0")

# Define API endpoints
@app.get("/health")
def health_check():
    return {"API": "🚀 Banking API is ready",
            "status": "✅ Healthy"}

@app.post("/predict/fraud")
def fraud(req: Request):
    result = predict_fraud(req.customer_id)
    return {"fraud_prediction": result}

@app.post("/predict/marketing")
def marketing(req: Request):
    result = predict_marketing(req.customer_id)
    return {"marketing_prediction": result}

@app.post("/predict/operational")
def operational(req: Request):
    result = predict_operational(req.customer_id)
    return {"operational_prediction": result}