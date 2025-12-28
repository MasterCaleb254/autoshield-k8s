# docker/inference-service.Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.115.6 \
    "uvicorn[standard]==0.32.1" \
    numpy==2.2.1
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1

# Install PyTorch with CUDA if needed (uncomment for GPU)
# RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Copy application
COPY src/ ./src/
COPY src/scripts/ ./scripts/
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
CMD ["uvicorn", "autoshield.api.server:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]