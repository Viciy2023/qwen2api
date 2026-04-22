# syntax=docker/dockerfile:1.7

ARG SOURCE_REPO=https://github.com/YuJunZhiXue/qwen2API.git
ARG SOURCE_REF=main
ARG FALLBACK_IMAGE=yujunzhixue/qwen2api:latest

# 备用方案：保留官方镜像 stage，便于后续切回镜像模式。
FROM ${FALLBACK_IMAGE} AS official_runtime

FROM debian:bookworm-slim AS source_fetch
ARG SOURCE_REPO
ARG SOURCE_REF
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 --branch ${SOURCE_REF} ${SOURCE_REPO} qwen2api

FROM node:20-bookworm-slim AS frontend_builder
WORKDIR /app
COPY --from=source_fetch /src/qwen2api/frontend/package*.json ./
RUN npm ci
COPY --from=source_fetch /src/qwen2api/frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm AS source_runtime
WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace \
    PORT=7860 \
    WORKERS=1 \
    ENGINE_MODE=hybrid \
    BROWSER_POOL_SIZE=1 \
    MAX_INFLIGHT=1 \
    ACCOUNT_MIN_INTERVAL_MS=1200 \
    REQUEST_JITTER_MIN_MS=120 \
    REQUEST_JITTER_MAX_MS=360 \
    STREAM_KEEPALIVE_INTERVAL=5 \
    ACCOUNTS_FILE=/data/accounts.json \
    USERS_FILE=/data/users.json \
    CAPTURES_FILE=/data/captures.json \
    CONFIG_FILE=/data/config.json \
    API_KEYS_FILE=/data/api_keys.json

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdbus-glib-1-2 \
    libdrm2 \
    libgbm1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpangocairo-1.0-0 \
    libpulse0 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    libxshmfence1 \
    fonts-liberation \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

COPY --from=source_fetch /src/qwen2api/backend/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && pip install --no-cache-dir h2
RUN python -m camoufox fetch

COPY --from=source_fetch /src/qwen2api/backend ./backend
COPY --from=source_fetch /src/qwen2api/start.py ./start.py
COPY --from=frontend_builder /app/dist ./frontend/dist
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh && mkdir -p /data /workspace/logs /workspace/frontend

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-7860}/healthz" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
