# Dockerfile.feature-extractor
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    kubectl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY autoshield/ ./autoshield/
COPY feature_extractor.py .

# Create non-root user
RUN useradd -m -u 1000 autoshield && chown -R autoshield:autoshield /app
USER autoshield

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "feature_extractor.py"]