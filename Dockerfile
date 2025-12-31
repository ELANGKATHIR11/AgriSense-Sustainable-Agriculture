# AgriSense Backend - Production Dockerfile
# Multi-stage build for optimized image size

# Stage 1: Builder - Install dependencies
FROM python:3.12-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY agrisense_app/backend/requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime - Minimal production image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AGRISENSE_ENV=production
ENV PYTHONPATH=/app/agrisense_app/backend:/app/agrisense_app:/app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r agrisense && useradd -r -g agrisense agrisense

# Set working directory
WORKDIR /app
# Copy requirements and install in the runtime image to ensure executables (uvicorn) are present
COPY agrisense_app/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY agrisense_app /app/agrisense_app
# Do not bake large model files into the image. Models are fetched at container start
# or mounted from a volume. The runtime will populate `/app/agrisense_app/backend/ml_models` when needed.
RUN mkdir -p /app/agrisense_app/backend/ml_models

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R agrisense:agrisense /app

# Switch to non-root user
USER agrisense

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Expose port
EXPOSE 8004

# Add a small runtime fetcher that will download models if the models folder is empty

COPY scripts/fetch_models.py /app/scripts/fetch_models.py

# Start application via a tiny entrypoint that ensures models exist (if configured)
CMD ["/bin/sh", "-lc", "python /app/scripts/fetch_models.py --dest /app/agrisense_app/backend/ml_models || true; uvicorn agrisense_app.backend.main:app --host 0.0.0.0 --port 8004 --workers 4"]
