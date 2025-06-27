# Use official Python image
FROM python:3.12.4-slim

# Set work directory
WORKDIR /app

# Install system dependencies (minimal, no recommends)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && find /usr/local/lib/python3.12/site-packages -type d -name '__pycache__' -exec rm -rf {} + \
    && find /usr/local/lib/python3.12/site-packages -type d -name 'tests' -exec rm -rf {} + \
    && find /usr/local/lib/python3.12/site-packages -type d -name 'test' -exec rm -rf {} + \
    && find /usr/local/lib/python3.12/site-packages -type d -name 'docs' -exec rm -rf {} +

# Copy the rest of the code
COPY . .
# Remove unnecessary files and folders to reduce image size
RUN rm -rf tests __pycache__ docs .git

# Copy .env file for local docker run (docker-compose already injects envs)
COPY .env .

# Expose FastAPI port
EXPOSE 8000

# Default command (can be overridden by docker-compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
