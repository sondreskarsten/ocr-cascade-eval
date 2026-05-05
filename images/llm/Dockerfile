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
RUN pip install xgrammar transformers || true

WORKDIR /app
COPY shared.py /app/shared.py
COPY runners /app/runners
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m"]
