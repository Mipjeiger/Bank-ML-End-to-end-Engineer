from prometheus_client import Counter, generate_latest, CollectorRegistry, Gauge
from fastapi import Response
from typing import Optional

# Shared registry for Prometheus metrics
REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter("request_total", "Total number of requests", registry=REGISTRY)
PREDICTION_COUNT = Counter("prediction_total", "Total number of predictions made", ["model", "status"], registry=REGISTRY)
DATA_DRIFT_SCORE = Gauge("data_drift_score", "Data drift score based on statistical tests", registry=REGISTRY)

"""Prometheus metrics for monitoring the API."""
def setup_metrics(app):
    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(REGISTRY), media_type="text/plain")