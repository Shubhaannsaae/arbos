# Multi-stage production Docker build
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config.yaml .

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs

# Change ownership to non-root user
RUN chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "src.inference.model_server:app", "--host", "0.0.0.0", "--port", "8000"]

# Production stage
FROM base as production

# Copy trained models (if available)
COPY models/ ./models/

# Copy additional production configs
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# Use production entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter==1.0.0 \
    ipykernel==6.25.1 \
    notebook==7.0.2

# Expose Jupyter port
EXPOSE 8888

# Development command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
