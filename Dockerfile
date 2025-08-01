# Use official Python image with specific version
FROM python:3.12.4-slim

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Install system dependencies and security updates
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/archives/*

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies with security flags
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org -r requirements.txt \
    && pip check

# Copy app code with proper ownership
COPY --chown=appuser:appuser . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/temp \
    && chown -R appuser:appuser /app \
    && chmod -R 755 /app \
    && chmod -R 777 /app/logs /app/temp

# Switch to non-root user
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Set environment variables for host services
ENV REDIS_HOST=host.docker.internal
ENV MILVUS_HOST=host.docker.internal
ENV OLLAMA_HOST=host.docker.internal
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI app with production settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--access-log"]
