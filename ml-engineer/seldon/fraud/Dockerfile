FROM python:3.11-slim

WORKDIR /app

# Build from Bank-Project/ root:
# docker build -f ml-engineer/seldon/fraud/Dockerfile -t mipjeiger/fraud-model:latest .
COPY ml-engineer/seldon/fraud/model.py /app/
COPY ml-engineer/seldon/fraud/requirements.txt /app/
COPY ml-engineer/models/banking_models/models/ /models/

RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set PYTHONPATH to /app so model.py can be found    
ENV PYTHONPATH=/app 

# Explicity define the model Name and Service Type
ENV MODEL_NAME=FraudModel
ENV SERVICE_TYPE=MODEL

CMD ["seldon-core-microservice", "model", "FraudModel"]