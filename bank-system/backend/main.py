from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.routes import router
from app.monitoring.prometheus import setup_metrics
from config.config import MODEL_PATH

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown

app = FastAPI(title="Banking System API",
              description="API for bank system with model inference and caching, involving feature engineering and deployment monitoring.",
              version="1.0.0",
              lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Set up Prometheus metrics
setup_metrics(app)

# Include the router and metrics monitoring
app.include_router(router)

# Endpoint for health check
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Welcome to the Banking System API!"}

@app.get("/models-registry")
async def list_models():
    from app.services.model_registry import registry
    return {"available_models": registry.list_models()}