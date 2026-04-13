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

CMD ["seldon-core-microservice", "model", "--service-type", "MODEL"]