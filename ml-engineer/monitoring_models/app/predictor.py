import time
from app.metrics import REQUEST_COUNT, REQUEST_LATENCY, ERROR_COUNT

def predict(models, category, model_name, data):
    try:
        model = models[category][model_name]

        start = time.time()
        result = model.predict([data])
        latency = time.time() - start

        REQUEST_COUNT.labels(
            model_name=model_name,
            category=category
        ).inc()

        REQUEST_LATENCY.labels(
            model_name=model_name,
            category=category
        ).observe(latency)

        return result.tolist()
    
    except Exception as e:
        ERROR_COUNT.labels(
            model_name=model_name
        ).inc()
        raise e