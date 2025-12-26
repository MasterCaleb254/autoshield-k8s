#!/bin/bash
# build-and-deploy.sh

# Build feature extractor image
docker build -f Dockerfile.feature-extractor -t autoshield/feature-extractor:latest .

# Load into k3d cluster
k3d image import autoshield/feature-extractor:latest -c autoshield-cluster

# Create namespace
kubectl create namespace autoshield-system

# Create kubeconfig configmap
kubectl create configmap kube-config \
  --namespace autoshield-system \
  --from-file=$HOME/.kube/config

# Deploy feature extractor
kubectl apply -f deployment/feature-extractor.yaml

# Wait for deployment
kubectl wait --for=condition=available deployment/feature-extractor \
  -n autoshield-system --timeout=300s

# Check logs
kubectl logs -l component=feature-extractor -n autoshield-system --tail=10