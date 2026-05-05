FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
ENV HF_HOME=/tmp/hf_cache TRANSFORMERS_NO_ADVISORY_WARNINGS=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl tesseract-ocr tesseract-ocr-nor poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install \
    "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
RUN pip install \
    "numpy<2" \
    "transformers==4.46.3" \
    "sentence-transformers==3.3.1" \
    "accelerate==1.1.1" \
    "peft==0.13.2" \
    "huggingface_hub==0.26.5" \
    "FlagEmbedding==1.3.2" \
    "datasets==3.1.0" \
    "setfit==1.1.0" \
    "gliner==0.2.16" \
    "gliclass==0.1.10" \
    "Pillow==11.0.0" \
    "pytesseract==0.3.13" \
    "pdfplumber==0.11.4" \
    "pandas==2.2.3" \
    "psutil==6.1.0" \
    "google-cloud-storage==2.18.2" \
    "sentencepiece==0.2.0" \
    "protobuf==5.28.3"

WORKDIR /app
COPY models/shared.py /app/shared.py
COPY models /app/models
ENV PYTHONPATH=/app:/app/models

ENTRYPOINT ["python"]
