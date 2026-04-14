FROM python:3.11-slim

WORKDIR /app

COPY ml-engineer/seldon/operational/OperationalModel.py /app/OperationalModel.py
COPY ml-engineer/seldon/operational/requirements.txt /app/
COPY ml-engineer/models/banking_models/models/ /models/

RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/app
ENV MODEL_NAME=OperationalModel
ENV SERVICE_TYPE=MODEL
    
CMD ["seldon-core-microservice", "OperationalModel"]