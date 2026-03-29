import joblib
import os
import gradio as gr
import glob
import logging
import json
import pickle
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# ---------------------------------
# LOGGING CONFIGURATION
# ---------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------
# CONFIGURATION
# ---------------------------------
CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent.parent

# Churn Model Configuration
MODEL_DIR_CHURN = BASE_DIR / "models" / "model"

# Fraud Detection Model Configuration
MODEL_DIR_FRAUD = BASE_DIR / "models" / "fraud_detection_results" / "models"
REGISTRY_FILE_FRAUD = BASE_DIR / "models" / "fraud_detection_results" / "model_registry.json"

# ────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS (Request/Response Schemas)
# ────────────────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """Base prediction request schema"""
    CreditScore: int = Field(..., description="Credit score of the customer")
    Geography: str = Field(..., description="Geographical location of the customer")
    Gender: str = Field(..., description="Gender (Male, Female)")
    Age: int = Field(..., description="Age (18-100)")
    Tenure: int = Field(..., description="Years as customer (0-10)")
    Balance: float = Field(..., description="Account balance")
    NumOfProducts: int = Field(..., description="Number of products (1-4)")
    HasCrCar: bool = Field(..., description="Has credit card")
    IsActiveMember: int = Field(..., description="Is active member (0 or 1)")
    EstimatedSalary: float = Field(..., description="Estimated salary")

    @validator('CreditScore')
    def validate_credit_score(cls, v):
        if not 300 <= v <= 850:
            raise ValueError('CreditScore must be between 300 and 850')
        return v
    
    @validator('Age')
    def validate_age(cls, v):
        if not 18 <= v <= 100:
            raise ValueError('Age must be between 18 and 100')
        return v
    
class PredictionResponse(BaseModel):
    """Standard prediction response schema"""
    prediction: int
    probability: Optional[float] = None
    model_used: str
    timestamp: str
    task: str

class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str
    timestamp: str
    models_loaded: int
    tasks_available: list[str]
    version: str = "1.0"

class ModelInfo(BaseModel):
    """Model information"""
    model_name: str
    task: str
    type: str
    metrics: Dict[str, float] = {}

# ────────────────────────────────────────────────────────────────────────────
# MODEL MANAGER CLASS
# ────────────────────────────────────────────────────────────────────────────

class ModelManager:
    """Manage loaded models and predictions"""

    def __init__(self):
        self.churn_model = None
        self.fraud_models: Dict = {}
        self.fraud_registry: Dict = {}
        self.churn_loaded = None
        self.fraud_loaded = None
        self.load_timestamp = None

    def load_churn_model(self) -> bool:
        """Load churn model"""
        try:
            if not MODEL_DIR_CHURN.exists():
                logger.warning(f"Churn model directory not found: {MODEL_DIR_CHURN}")
                return False
            
            # Get pickle files from churn model directory
            model_files = list(MODEL_DIR_CHURN.glob("*.pkl"))

            if not model_files:
                logger.warning(f"No model files found in churn model directory: {MODEL_DIR_CHURN}")
                return False

            # Load the main churn model
            model_file = model_files[0]
            with open(model_file, 'rb') as f:
                self.churn_model = pickle.load(f)

            logger.info(f"Churn model loaded: {model_file}")
            self.churn_loaded = True
            return True
        
        except Exception as e:
            logger.error(f"Error loading churn model: {e}")
            return False

    def load_fraud_models(self) -> bool:
        """Load fraud detection models and registry"""

        try:
            if not MODEL_DIR_FRAUD.exists():
                logger.warning(f"Fraud model directory not found: {MODEL_DIR_FRAUD}")
                return False
            
            # Load model registry
            if not REGISTRY_FILE_FRAUD.exists():
                with open(REGISTRY_FILE_FRAUD, 'r') as f:
                    self.fraud_registry = json.load(f)
                logger.info(f"Fraud model registry loaded: {REGISTRY_FILE_FRAUD}")

            # Get all pickle files
            model_files = sorted(MODEL_DIR_FRAUD.glob("*.pkl"))

            if not model_files:
                logger.warning(f"No model files found in fraud model directory: {MODEL_DIR_FRAUD}")
                return False
            
            logger.info(f"Loading {len(model_files)} fraud models from {MODEL_DIR_FRAUD}")

            # Load each fraud model
            for model_file in model_files:
                try:
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    
                    filename = model_file.stem
                    parts = filename.rsplit('_', 1)

                    if len(parts) == 2:
                        model_name = parts[0].replace('_', ' ')
                        task = parts[1]

                        if model_name not in self.fraud_models:
                            self.fraud_models[model_name] = {}

                        self.fraud_models[model_name][task] = model
                        logger.info(f"Loaded fraud model: {model_name} for task: {task}")

                except Exception as e:
                    logger.error(f"Error loading model {model_file}: {e}")
                    continue

            logger.info(f"Loaded {sum(len(t) for t in self.fraud_models.values())} fraud models successfully")
            self.fraud_loaded = True
            return True
        
        except Exception as e:
            logger.error(f"Error loading fraud models: {e}")
            return False
        
    def load_all_models(self) -> bool:
        """Load all models (churn and fraud)"""
        logger.info("Loading all models...")

        churn_ok = self.load_churn_model()
        fraud_ok = self.load_fraud_models()

        self.load_timestamp = datetime.now().isoformat()

        if churn_ok and fraud_ok:
            logger.info("All models loaded successfully")
            return True
        else:
            logger.warning("Some models failed to load")
            return False
        
    def predict_churn(self, data: pd.DataFrame) -> Tuple[int, str]:
        """Make churn prediction"""
        if self.churn_model is None:
            raise ValueError("Churn model not loaded")
        
        try:
            prediction = self.churn_model.predict(data)
            return int(prediction[0]), "Customer churn"
        except Exception as e:
            logger.error(f"Churn prediction error: {e}")
            raise

    def get_best_fraud_model(self, task: str):
        """Get best fraud model for a given task"""
        if task in self.fraud_registry and 'Model' in self.fraud_registry[task]:
            model_name = self.fraud_registry[task]['Model']
            if model_name in self.fraud_models and task in self.fraud_models[model_name]:
                return self.fraud_models[model_name][task], model_name
            
        # Fallback to first available model for the task
        for name, tasks in self.fraud_models.items():
            if task in  tasks:
                return tasks[task], name
            
        return None, None
    
    def predict_fraud(self, task: str, data: pd.DataFrame) -> Tuple[int, str]:
        """Make fraud prediction"""
        model, model_name = self.get_best_fraud_model(task=task)
        if model is None:
            raise ValueError(f"No model available for task: {task}")
        
        try:
            prediction = model.predict(data)
            return int(prediction[0]), model_name
        except Exception as e:
            logger.error(f"Fraud prediction error for task {task}: {e}")
            raise

    def get_model_info(self) -> Dict:
        """Get information about all loaded models"""
        return {
            "churn": {
                "loaded": self.churn_loaded,
                "model": self.churn_model.__class__.__name__ if self.churn_model else None
            },
            "fraud": {
                "loaded": self.fraud_loaded,
                "total_models": len(self.fraud_models),
                "task": sum(len(t) for t in self.fraud_models.values()),
                "models": {name: list(tasks.keys()) for name, tasks in self.fraud_models.items()}
            }
        }
    
# ────────────────────────────────────────────────────────────────────────────
# LIFESPAN EVENT HANDLER
# ────────────────────────────────────────────────────────────────────────────

model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shuwtdown events"""
    # Startup
    logger.info("Starting up application...")
    model_manager.load_all_models()
    logger.info("Application startup complete")
    yield
    # Shutdown
    logger.info("Shutting down application...")

# ────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION
# ────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fintech Intellience API",
    description="Production API for churn prediction, fraud detection, marketing, and operational risk",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ────────────────────────────────────────────────────────────────────────────
# ENDPOINTS - HEALTH & INFO
# ────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model availability"""
    return HealthResponse(
        status="healthy" if (model_manager.churn_loaded or model_manager.fraud_loaded) else "unhealthy",
        timestamp=datetime.now().isoformat(),
        churn_models_loaded=1 if model_manager.churn_loaded else 0,
        fraud_models_loaded=len(model_manager.fraud_models),
        tasks_available=["churn"] if model_manager.churn_loaded else [] + [task for model_dict in model_manager.fraud_models.values() for task in model_dict.keys()]
    )

@app.get("/models", tags=["Info"])
async def get_models_info():
    """Get information about loaded models"""
    return model_manager.get_model_info()

@app.get("/version", tags=["Info"])
async def get_version():
    """Get API version"""
    return {"version": "1.0.0",
            "status": "production",
            "churn_model_loaded": model_manager.churn_loaded,
            "fraud_models_loaded": len(model_manager.fraud_models)
            }

# ────────────────────────────────────────────────────────────────────────────
# ENDPOINTS - CHURN PREDICTION
# ────────────────────────────────────────────────────────────────────────────

@app.post("/predict/churn", response_model=PredictionResponse, tags=["Predictions"])
async def predict_churn(request: PredictionRequest):
    """Predict customer churn"""
    try:
        if not model_manager.churn_loaded:
            raise HTTPException(status_code=503, detail='Churn model not available')
        
        # Prediction toward to dataframe
        df = pd.DataFrame([request.dict()])
        prediction, model_name = model_manager.predict_churn(df)

        logger.info(f"Churn prediction: {prediction}")

        return PredictionResponse(
            prediction=prediction,
            model_used=model_name,
            model_type="fraud",
            timestamp=datetime.now().isoformat(),
            task="Operational Risk"
        )
    except Exception as e:
        logger.error(f"Error in churn prediction endpoint: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
# ────────────────────────────────────────────────────────────────────────────
# ROOT ENDPOINT
# ────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    """API documentation root and info"""
    return {
        "name": "Fintech Intelligence API",
        "version": "1.0.0",
        "status": "production",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "endpoints": {
            "churn": "/predict/churn",
            "fraud": ["/predict/fraud", "/predict/marketing", "/predict/operational"],
            "info": ["/models", "/version", "/health"]
        },
        "models_info": model_manager.get_model_info()
    }

# ────────────────────────────────────────────────────────────────────────────/
# ERROR HANDLERS
# ────────────────────────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail,
                 "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "An unexpected error occurred",
                 "timestamp": datetime.now().isoformat()}
    )

# Usage
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        workers=int(os.getenv("API_WORKERS", 4)),
        reload=True
    )