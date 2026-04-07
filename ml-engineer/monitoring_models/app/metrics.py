from prometheus_client import Counter, Histogram, Gauge

# Requests counter
REQUEST_COUNT = Counter(
    "model_requests_total",
    "Total number of prediction requests",
    ["model_name", "category"]
)

# Latency
REQUEST_LATENCY = Histogram(
    "model_latency_seconds",
    "Latency of prediction",
    ["model_name", "category"]
)

# Errors
ERROR_COUNT = Counter(
    "model_errors_total",
    "Total errors",
    ["model_name"]
)

# Prediction values
PREDICTION_VALUE = Gauge(
    "model_prediction_value",
    "Prediction output",
    ["model_name"]
)