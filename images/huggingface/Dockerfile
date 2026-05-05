FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
ENV HF_HOME=/tmp/hf_cache TRANSFORMERS_NO_ADVISORY_WARNINGS=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl tesseract-ocr tesseract-ocr-nor poppler-utils \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
RUN pip install \
    "numpy<2" \
    "transformers>=4.46,<4.50" \
    "sentence-transformers>=3.3" \
    "accelerate>=1.1" \
    "peft>=0.13" \
    "huggingface_hub>=0.26" \
    "datasets>=3.0" \
    "setfit>=1.1" \
    "gliner>=0.2" \
    "gliclass>=0.1" \
    "Pillow" "pytesseract" "pdfplumber" "pandas" "psutil" \
    "google-cloud-storage" "sentencepiece" "protobuf<6" \
    "rapidfuzz" "pandera" "outlines" "instructor" "marvin" "dspy-ai" \
    "einops" "timm" \
    "easyocr" \
    "python-doctr[torch]" \
    "rank_bm25"

RUN pip install \
    "nougat-ocr" \
    "marker-pdf" \
    "docling" \
    "colpali-engine" \
    "spacy" \
    "stanza" \
    "guidance" \
    "lmql" \
    "jsonformer" \
    "rank_bm25" || true
RUN python -m spacy download nb_core_news_lg || true

RUN pip install "surya-ocr" || true
RUN pip install "FlagEmbedding" || true

WORKDIR /app
COPY shared.py /app/shared.py
COPY runners /app/runners
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m"]
