#!/bin/bash
# setup-cluster.sh

# Create cluster with 3 nodes
k3d cluster create autoshield-cluster \
  --api-port 6550 \
  --servers 1 \
  --agents 2 \
  --port "8080:80@loadbalancer" \
  --k3s-arg "--disable=traefik@server:0" \
  --k3s-arg "--disable=metrics-server@server:0"

# Verify cluster
kubectl cluster-info
kubectl get nodes -o wide