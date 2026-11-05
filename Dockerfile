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

WORKDIR /app
USER autocare

COPY --chown=autocare:autocare pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=autocare:autocare . .
RUN uv sync --frozen --no-dev

# Bake the models into HF_HOME. Settings validation needs placeholder values at build time only.
RUN JWT_SECRET_KEY=build-time-placeholder-not-a-secret-000 \
    SUPABASE_URL=http://placeholder SUPABASE_KEY=placeholder \
    DEEPSEEK_API_KEY=placeholder \
    uv run --no-sync python -m app.services.hearing

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
