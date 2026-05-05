FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install \
    "numpy<2" \
    "huggingface_hub==0.26.5" \
    "llama-cpp-python==0.3.1" \
    "psutil" \
    "google-cloud-storage"

WORKDIR /app
COPY models/shared.py /app/shared.py
COPY models /app/models
ENV PYTHONPATH=/app:/app/models

ENTRYPOINT ["python"]
