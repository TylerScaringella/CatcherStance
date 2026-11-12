FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

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

COPY requirements.txt ./
RUN pip install -r requirements.txt \
    && python -m playwright install --with-deps chromium

COPY src ./src
COPY models ./models
COPY data ./data
COPY research/specs ./research/specs

RUN useradd --create-home --uid 1000 catcher \
    && mkdir -p /app/data/auth /app/data/cache /app/data/runs /app/data/tmp \
    && chown -R catcher:catcher /app

ENV PYTHONPATH=/app/src \
    CATCHER_STANCE_HOST=0.0.0.0 \
    CATCHER_STANCE_PORT=8000

USER catcher

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", "--timeout", "0", "backend.app:create_app()"]
