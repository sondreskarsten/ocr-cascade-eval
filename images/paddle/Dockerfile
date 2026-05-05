FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
ENV FLAGS_use_mkldnn=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install \
    "numpy<2" \
    "paddlepaddle==3.2.0" \
    "paddleocr==3.5.0" \
    "Pillow" \
    "psutil" \
    "google-cloud-storage"

WORKDIR /app
COPY models/shared.py /app/shared.py
COPY models /app/models
ENV PYTHONPATH=/app:/app/models

ENTRYPOINT ["python"]
