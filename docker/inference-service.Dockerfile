# docker/inference-service.Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA if needed (uncomment for GPU)
# RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Copy application
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/models/ ./data/models/

# Create non-root user
RUN useradd -m -u 1000 autoshield && chown -R autoshield:autoshield /app
USER autoshield

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the service
CMD ["uvicorn", "src.autoshield.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]