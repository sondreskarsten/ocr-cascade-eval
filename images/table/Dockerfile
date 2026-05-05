FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential default-jre-headless ghostscript \
    poppler-utils tesseract-ocr tesseract-ocr-nor tesseract-ocr-eng \
    ocrmypdf libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install \
    "numpy<2" \
    "camelot-py[cv]==0.11.0" \
    "ghostscript==0.7" \
    "tabula-py==2.10.0" \
    "pdfplumber==0.11.4" \
    "pytesseract==0.3.13" \
    "Pillow" \
    "psutil" \
    "google-cloud-storage"
RUN pip install "splink" "dedupe" "valentine-codes" || true

WORKDIR /app
COPY shared.py /app/shared.py
COPY runners /app/runners
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m"]
