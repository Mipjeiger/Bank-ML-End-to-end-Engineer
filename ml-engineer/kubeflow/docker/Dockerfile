FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy with full paths from project root
COPY ml-engineer/kubeflow/docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full kubeflow directory
COPY ml-engineer/kubeflow .

# Set PYTHONPATH for imports
ENV PYTHONPATH="/app:${PYTHONPATH}"

CMD ["python", "pipelines/main_pipeline.py"]