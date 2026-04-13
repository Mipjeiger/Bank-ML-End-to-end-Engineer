FROM python:3.11-slim

WORKDIR /app

COPY ml-engineer/LLM/banking_llm/app.py /app/
COPY ml-engineer/LLM/banking_llm/prompt_engine.py /app/
COPY ml-engineer/LLM/banking_llm/requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV MODEL_NAME=banking-llm

CMD ["seldon-core-microservice", "app:BankingLLM"]