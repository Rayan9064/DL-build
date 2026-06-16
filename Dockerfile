FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
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
    && pip install -r requirements.txt \
    && python -c "from pathlib import Path; path = Path('/usr/local/lib/python3.11/site-packages/label_studio/tasks/api.py'); text = path.read_text(); old = \"        serializer = self.get_serializer_class()(\\n            self.task, many=False, context=context, expand=['annotations.completed_by']\\n        )\\n\"; new = \"        serializer = self.get_serializer_class()(self.task, many=False, context=context)\\n\"; assert old in text, 'Expected Label Studio task API snippet not found'; path.write_text(text.replace(old, new)); print('Patched Label Studio task API serializer expansion')"

COPY . .

EXPOSE 8080 9090

CMD ["python", "-m", "egocentric_backend.server", "--host", "0.0.0.0", "--port", "9090", "--preload"]
