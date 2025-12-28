# docker/dashboard.Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY monitoring/ ./monitoring/

# Create non-root user
RUN useradd -m -u 1000 autoshield && chown -R autoshield:autoshield /app
USER autoshield

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Expose ports
EXPOSE 8081
EXPOSE 9090

# Run dashboard
CMD ["uvicorn", "src.autoshield.dashboard.api:dashboard_app", "--host", "0.0.0.0", "--port", "8081"]