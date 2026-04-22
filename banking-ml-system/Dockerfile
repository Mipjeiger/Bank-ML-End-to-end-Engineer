FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "deepchecks[monitoring]"

CMD sh -c "deepchecks-monitoring start --host 0.0.0.0 --port 8081 & uvicorn app.main:app --host 0.0.0.0 --port 8080"