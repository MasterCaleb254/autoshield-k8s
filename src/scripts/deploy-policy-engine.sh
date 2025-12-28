#!/bin/bash
# scripts/deploy-policy-engine.sh

# Build orchestrator image
docker build -f docker/orchestrator.Dockerfile -t autoshield/orchestrator:latest .

# Load into k3d cluster
k3d image import autoshield/orchestrator:latest -c autoshield-cluster

# Create config maps for policies
kubectl create configmap autoshield-policies \
  --namespace autoshield-system \
  --from-file=config/policies/default.yaml \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy orchestrator
kubectl apply -f deployment/orchestrator.yaml

# Wait for deployment
kubectl wait --for=condition=available deployment/orchestrator \
  -n autoshield-system --timeout=300s

# Check logs
kubectl logs -l component=orchestrator -n autoshield-system --tail=10

echo "Policy engine deployed successfully!"