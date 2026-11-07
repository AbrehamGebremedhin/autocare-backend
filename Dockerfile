FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.18 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    HF_HOME=/home/autocare/.cache/huggingface

RUN useradd --create-home --uid 10001 autocare \
    && mkdir /app \
    && chown autocare:autocare /app

# A fresh named volume mounted here copies this owner, so the non-root app can write uploads.
RUN mkdir -p /data/media && chown autocare:autocare /data/media

WORKDIR /app
USER autocare

COPY --chown=autocare:autocare pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=autocare:autocare . .
RUN uv sync --frozen --no-dev

# Bake the models into HF_HOME. Settings validation needs placeholder values at build time only.
RUN JWT_SECRET_KEY=build-time-placeholder-not-a-secret-000 \
    DEEPSEEK_API_KEY=placeholder \
    DATABASE_URL=postgresql://placeholder@localhost/placeholder \
    uv run --no-sync python -m app.services.hearing

EXPOSE 8000

# One worker only: the in-flight session lock lives in process memory.
# --proxy-headers trusts X-Forwarded-For only from the addresses in $FORWARDED_ALLOW_IPS (uvicorn's default: 127.0.0.1).
CMD ["uv", "run", "--no-sync", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
