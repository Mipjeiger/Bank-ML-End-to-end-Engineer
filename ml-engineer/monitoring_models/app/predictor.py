import time
from app.metrics import REQUEST_COUNT, REQUEST_LATENCY, ERROR_COUNT

def predict(models, catetgory, model_name, data):
    try:
        model = models[catetgory][model_name]

        start = time.time()
        result = model.predict([data])
        latency = time.time() - start

        REQUEST_COUNT.labels(model_name=model_name, catetgory=catetgory).inc()
        REQUEST_LATENCY.labels(model_name=model_name, catetgory=catetgory).observe(latency)

        return result.tolist()
    
    except Exception as e:
        ERROR_COUNT.labels(model_name=model_name, catetgory=catetgory).inc()
        raise e