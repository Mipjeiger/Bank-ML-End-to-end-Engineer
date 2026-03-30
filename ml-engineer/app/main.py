import joblib
import os
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
from pydantic import BaseModel, Field, field_validator
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
    RowNumber: int = Field(..., description="Row number in the dataset")
    CustomerId: int = Field(..., description="Unique customer ID")
    Surname: int = Field(..., description="Customer surname (not used in prediction)")
    CreditScore: int = Field(..., description="Credit score of the customer")
    Geography: str = Field(..., description="Geographical location of the customer")
    Gender: str = Field(..., description="Gender (Male, Female)")
    Age: int = Field(..., description="Age (18-100)")
    Tenure: int = Field(..., description="Years as customer (0-10)")
    Balance: float = Field(..., description="Account balance")
    NumOfProducts: int = Field(..., description="Number of products (1-4)")
    HasCrCard: bool = Field(..., description="Has credit card")
    IsActiveMember: int = Field(..., description="Is active member (0 or 1)")
    EstimatedSalary: float = Field(..., description="Estimated salary")
    Complain: int = Field(..., description="Customer complain (0 or 1)")
    SatisfactionScore: int = Field(..., description="Customer satisfaction score (1-5)")
    CardType: str = Field(..., description="Type of card (Blue, Silver, Gold, Platinum)")
    PointEarned: int = Field(..., description="Loyalty points earned")

    @field_validator('CreditScore')
    @classmethod
    def validate_credit_score(cls, v: int) -> int:
        if not 300 <= v <= 850:
            raise ValueError('CreditScore must be between 300 and 850')
        return v

    @field_validator('Age')
    @classmethod
    def validate_age(cls, v: int) -> int:
        if not 18 <= v <= 100:
            raise ValueError('Age must be between 18 and 100')
        return v
    
    @field_validator('IsActiveMember')
    @classmethod
    def validate_is_active_member(cls, v: int) -> int:
        if v not in [0, 1]:
            raise ValueError('IsActiveMember must be either 0 or 1')
        return v

    @field_validator('Complain')
    @classmethod
    def validate_complain(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError('Complain must be 0 or 1')
        return v

    @field_validator('SatisfactionScore')
    @classmethod
    def validate_satisfaction_score(cls, v: int) -> int:
        if not 1 <= v <= 5:
            raise ValueError('SatisfactionScore must be between 1 and 5')
        return v

class FraudPredictionRequest(BaseModel):
    """Base fraud prediction request schema"""
    CreditScore: int = Field(..., description="Credit score of the customer")
    Geography: str = Field(..., description="Geographical location of the customer")
    Gender: str = Field(..., description="Gender (Male, Female)")
    Age: int = Field(..., description="Age (18-100)")
    Tenure: int = Field(..., description="Years as customer (0-10)")
    Balance: float = Field(..., description="Account balance")
    NumOfProducts: int = Field(..., description="Number of products (1-4)")
    HasCrCard: bool = Field(..., description="Has credit card")
    IsActiveMember: int = Field(..., description="Is active member (0 or 1)")
    EstimatedSalary: float = Field(..., description="Estimated salary")
    Exited: bool = Field(..., description="Customer exited (0 or 1)")
    Complain: int = Field(..., description="Customer complain (0 or 1)")
    SatisfactionScore: int = Field(..., description="Customer satisfaction score (1-5)")
    CardType: str = Field(..., description="Type of card (Blue, Silver, Gold, Platinum)")
    PointEarned: int = Field(..., description="Loyalty points earned")
    RiskScore: int = Field(..., description="Operational risk score (1-100)")
    BalancePerProduct: float = Field(..., description="Balance per product")
    AgeRisk: int = Field(..., description="Age risk score (1-100)")
    HighValueCustomer: int = Field(..., description="High value customer (0 or 1)")
    LowCreditRisk: int = Field(..., description="Low credit risk (0 or 1)")
    ComplainFlag: int = Field(..., description="Complain flag (0 or 1)")
    LowSatisfaction: int = Field(..., description="Low satisfaction flag (0 or 1)")

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
    churn_models_loaded: int
    fraud_models_loaded: int
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
        
    def predict_churn(self, data: np.ndarray) -> Tuple[int, float, str]:
        """Make churn prediction with probability"""
        if self.churn_model is None:
            raise ValueError("Churn model not loaded")
        
        try:
            prediction = self.churn_model.predict(data)
            probability = self.churn_model.predict_proba(data)
            churn_probability = float(probability[0][1])
            return int(prediction[0]), churn_probability, "Customer churn"
        except Exception as e:
            logger.error(f"Churn prediction error: {e}")
            raise

    def get_best_fraud_model(self, task: str, model_name: str = None):
        """Get fraud model based on user choice or registry"""
        # 1. if user specifies model -> use it
        if model_name:
            if model_name in self.fraud_models:
                if task in self.fraud_models[model_name]:
                    return self.fraud_models[model_name][task], model_name
                else:
                    raise ValueError(f"Model '{model_name}' does not support task '{task}")
            else:
                raise ValueError(f"Model '{model_name}' not found in loaded fraud models")
            
        # 2. Use registry (auto best models)
        if task in self.fraud_registry and 'Model' in self.fraud_registry[task]:
            best_model_name = self.fraud_registry[task]['Model']
            if best_model_name in self.fraud_models and task in self.fraud_models[best_model_name]:
                return self.fraud_models[best_model_name][task], best_model_name

        # 3. Fallback to first available model for the task
        for name, tasks in self.fraud_models.items():
            if task in  tasks:
                return tasks[task], name
            
        return None, None
    
    def predict_fraud(self, task: str, data: np.ndarray, model_name: str = None):
        """Make fraud prediction"""
        model, model_name = self.get_best_fraud_model(task, model_name)
        if model is None:
            raise ValueError(f"No model available for task: {task}")
        
        prediction = model.predict(data)
        probability = model.predict_proba(data)

        return int(prediction[0]), float(probability[0][1]), model_name

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
    
# Class API calls to loads startup and shutdown events
model_manager = ModelManager()

# ────────────────────────────────────────────────────────────────────────────
# LIFESPAN EVENT HANDLER
# ────────────────────────────────────────────────────────────────────────────

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
# ENDPOINTS - POST PREDICTION
# ────────────────────────────────────────────────────────────────────────────

# Encoding maps
"""Create encoding features to prevent data leakage and ensure consistency fits."""
GEOGRAPHY_MAP = {"France": 0, "Germany": 1, "Spain": 2}
GENDER_MAP = {"Male": 1, "Female": 0}
CARD_TYPE_MAP = {"DIAMOND": 0, "GOLD": 1, "PLATINUM": 2, "SILVER": 3}

# Churn Prediction Endpoint
@app.post("/predict/churn", response_model=PredictionResponse, tags=["Predictions"])
async def predict_churn(request: PredictionRequest):
    """Predict customer churn"""
    try:
        if not model_manager.churn_loaded:
            raise HTTPException(status_code=503, detail='Churn model not available')


        # Validate categorical variables
        if request.Geography not in GEOGRAPHY_MAP:
            raise HTTPException(status_code=400, detail=f"Invalid Geography value: {request.Geography}. Allowed values: {list(GEOGRAPHY_MAP.keys())}")
        
        if request.Gender not in GENDER_MAP:
            raise HTTPException(status_code=400, detail=f"Invalid Gender value: {request.Gender}. Allowed values: {list(GENDER_MAP.keys())}")

        if request.CardType not in CARD_TYPE_MAP:
            raise HTTPException(status_code=400, detail=f"Invalid Card Type value: {request.CardType}. Allowed values: {list(CARD_TYPE_MAP.keys())}")

        # Build numpy array in the EXACT same column order as training
        features = np.array([[
            request.RowNumber,
            request.CustomerId,
            request.Surname,
            request.CreditScore,
            GEOGRAPHY_MAP[request.Geography],
            GENDER_MAP[request.Gender],
            request.Age,
            request.Tenure,
            request.Balance,
            request.NumOfProducts,
            int(request.HasCrCard),
            int(request.IsActiveMember),
            request.EstimatedSalary,
            request.Complain,
            request.SatisfactionScore,
            CARD_TYPE_MAP[request.CardType],
            request.PointEarned
        ]])

        prediction, probability, model_name = model_manager.predict_churn(features)
        logger.info(f"Churn prediction: {prediction}, probability: {probability}")

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_used=model_name,
            timestamp=datetime.now().isoformat(),
            task="Customer churn"
        )
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error in churn prediction endpoint: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Create function for preprocessing fraud input data
def preprocess_fraud_input(request: FraudPredictionRequest) -> np.ndarray:
        # Define feature_order
        FEATURE_ORDER = [
            "CreditScore", "Geography", "Gender", "Age", "Tenure",
            "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
            "EstimatedSalary", "Exited", "Complain", "SatisfactionScore",
            "CardType", "PointEarned",
            "RiskScore", "BalancePerProduct", "AgeRisk",
            "HighValueCustomer", "LowCreditRisk", "ComplainFlag", "LowSatisfaction"
            ]
        """Preprocess fraud prediction input to match model features and order"""
        # Validate categorical variables
        if request.Geography not in GEOGRAPHY_MAP:
            raise HTTPException(status_code=400, detail=f"Invalid Geography value: {request.Geography}. Allowed values: {list(GEOGRAPHY_MAP.keys())}")
        
        if request.Gender not in GENDER_MAP:
            raise HTTPException(status_code=400, detail=f"Invalid Gender value: {request.Gender}. Allowed values: {list(GENDER_MAP.keys())}")
        
        if request.CardType not in CARD_TYPE_MAP:
            raise HTTPException(status_code=400, detail=f"Invalid Card Type value: {request.CardType}. Allowed values: {list(CARD_TYPE_MAP.keys())}")
        
        # Build features dict
        data = {
        "CreditScore": request.CreditScore,
        "Geography": GEOGRAPHY_MAP[request.Geography],
        "Gender": GENDER_MAP[request.Gender],
        "Age": request.Age,
        "Tenure": request.Tenure,
        "Balance": request.Balance,
        "NumOfProducts": request.NumOfProducts,
        "HasCrCard": int(request.HasCrCard),
        "IsActiveMember": int(request.IsActiveMember),
        "EstimatedSalary": request.EstimatedSalary,
        "Exited": int(request.Exited),
        "Complain": request.Complain,
        "SatisfactionScore": request.SatisfactionScore,
        "CardType": CARD_TYPE_MAP[request.CardType],
        "PointEarned": request.PointEarned,
        "RiskScore": request.RiskScore,
        "BalancePerProduct": request.BalancePerProduct,
        "AgeRisk": request.AgeRisk,
        "HighValueCustomer": request.HighValueCustomer,
        "LowCreditRisk": request.LowCreditRisk,
        "ComplainFlag": request.ComplainFlag,
        "LowSatisfaction": request.LowSatisfaction
        }

        features = np.array([[data[col] for col in FEATURE_ORDER]])
        return features

# Fraud Prediction Endpoint
@app.post("/predict/fraud/{task}", response_model=PredictionResponse, tags=["Predictions"])
async def predict_fraud(task: str, request: FraudPredictionRequest, model_name: str = None):
    """Predict fraud for a given task"""
    try:
        if not model_manager.fraud_loaded:
            raise HTTPException(status_code=503, detail='Fraud models not available')
        
        # Convert input data to numpy array (assuming data is a dict of features)
        features = preprocess_fraud_input(request)
        prediction, probability, model_name = model_manager.predict_fraud(task=task, 
                                                                          data=features, 
                                                                          model_name=model_name)
        logger.info(f"Fraud prediction for task {task}: {prediction}, probability: {probability}")

        return {
            "prediction": prediction,
            "probability": probability,
            "model_used": model_name,
            "timestamp": datetime.now().isoformat(),
            "task": task
        }
    except Exception as e:
        raise
    except Exception as e:
        logger.error(f"Error in fraud detection endpoint: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
# Market Risk Prediction Endpoint
@app.post("/predict/marketing", response_model=PredictionResponse, tags=["Predictions"])
async def predict_marketing)



# Operation Risk Prediction Endpoint
@app.post("/predict/operational-risk")




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
        reload=False
    )