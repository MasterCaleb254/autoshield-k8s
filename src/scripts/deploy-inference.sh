#!/bin/bash
# scripts/deploy-inference.sh

# Build inference service image
docker build -f docker/inference-service.Dockerfile -t autoshield/inference:latest .

# Load into k3d cluster
k3d image import autoshield/inference:latest -c autoshield-cluster

# Deploy inference service
kubectl apply -f deployment/inference-service.yaml

# Wait for deployment
kubectl wait --for=condition=available deployment/inference-service \
  -n autoshield-system --timeout=300s

# Port forward for testing
kubectl port-forward svc/inference-service 8000:8000 -n autoshield-system &

# Test the service
echo "Testing inference service..."
curl http://localhost:8000/health
curl http://localhost:8000/

echo "Inference service deployed successfully!"
echo "API available at: http://localhost:8000"