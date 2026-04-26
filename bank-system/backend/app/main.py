from fastapi import FastAPI
from app.api.routes import router
from app.monitoring.prometheus import setup_metrics

app = FastAPI(title="Banking System API",
              description="API for bank system with model inference and caching, involving feature engineering and deployment monitoring.",
              version="1.0.0")

# Include the router and metrics monitoring
app.include_router(router)
setup_metrics(app)