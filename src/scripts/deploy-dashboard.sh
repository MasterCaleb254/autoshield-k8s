#!/bin/bash
# scripts/deploy-dashboard.sh

# Build dashboard image
docker build -f docker/dashboard.Dockerfile -t autoshield/dashboard:latest .

# Load into k3d cluster
k3d image import autoshield/dashboard:latest -c autoshield-cluster

# Deploy dashboard
kubectl apply -f deployment/dashboard.yaml

# Deploy monitoring stack (optional)
kubectl apply -f monitoring/prometheus/prometheus.yaml

# Wait for deployment
kubectl wait --for=condition=available deployment/dashboard \
  -n autoshield-system --timeout=300s

# Port forward for local access
kubectl port-forward svc/dashboard 8081:8081 -n autoshield-system &
kubectl port-forward svc/dashboard 9090:9090 -n autoshield-system &

echo "Dashboard deployed successfully!"
echo "Access at: http://localhost:8081"
echo "Metrics at: http://localhost:9090/metrics"