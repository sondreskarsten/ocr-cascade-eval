FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN pip install --upgrade pip && pip install \
    google-cloud-storage google-auth requests
WORKDIR /app
COPY scripts/supervisor.py /app/supervisor.py
COPY jobs.json /app/jobs.json
ENTRYPOINT ["python","/app/supervisor.py"]
