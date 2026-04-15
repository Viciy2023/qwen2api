FROM yujunzhixue/qwen2api:latest

ENV PORT=7860 \
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
    CONFIG_FILE=/data/config.json

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh && mkdir -p /data

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
