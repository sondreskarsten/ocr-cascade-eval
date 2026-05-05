FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install \
    "numpy<2" \
    "arelle-release==2.34.10" \
    "psutil" \
    "google-cloud-storage" \
    "lxml"

RUN pip install pyesef || true

WORKDIR /app
COPY models/shared.py /app/shared.py
COPY models /app/models
ENV PYTHONPATH=/app:/app/models

ENTRYPOINT ["python"]
