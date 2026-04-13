FROM python:3.11-slim
WORKDIR /app
COPY ml-engineer/seldon/marketing/model.py /app/
COPY ml-engineer/seldon/marketing/requirements.txt /app/
COPY ml-engineer/models/banking_models/models/ /models/
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip && apt-get clean && rm -rf /var/lib/apt/lists/*
CMD ["seldon-core-microservice", "model", "--service-type", "MODEL"]