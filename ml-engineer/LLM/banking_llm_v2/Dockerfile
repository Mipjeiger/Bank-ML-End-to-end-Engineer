FROM python:3.11-slim

WORKDIR /app

COPY app.py .
COPY prompt_engine.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]