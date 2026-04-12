FROM python:3.11-slim

WORKDIR /app

COPY model.py /app/
COPY models/ /models/

RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV MODEL_NAME=marketing-model

CMD ["seldon-core-microservice", "model:MarketingModel"]