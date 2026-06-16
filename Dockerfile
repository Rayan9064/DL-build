FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
    LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/app \
    TORCH_HOME=/root/.cache/torch

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

EXPOSE 8080 9090

CMD ["python", "-m", "egocentric_backend.server", "--host", "0.0.0.0", "--port", "9090", "--preload"]
