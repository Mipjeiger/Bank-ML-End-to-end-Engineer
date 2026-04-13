FROM python:3.11-slim

WORKDIR /app

COPY ml-engineer/seldon/fraud/model.py /app/
COPY ml-engineer/models/banking_models/models /models/
COPY ml-engineer/seldon/fraud/requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV MODEL_NAME=fraud-model

CMD ["seldon-core-microservice", "model:FraudModel"]