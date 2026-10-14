FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN groupadd --system tracklab \
    && useradd --system --gid tracklab --create-home tracklab

COPY pyproject.toml LICENSE ./
COPY src/ ./src/

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir .

COPY configs/ ./configs/

RUN mkdir -p \
        /app/data/uploads \
        /app/outputs/jobs \
        /app/models \
        /tmp/Ultralytics \
        /tmp/matplotlib \
    && chmod 1777 /tmp/Ultralytics /tmp/matplotlib \
    && chown -R tracklab:tracklab /app /home/tracklab

USER tracklab

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "tracklab.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
