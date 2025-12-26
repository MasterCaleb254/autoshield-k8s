#!/bin/bash
# install-cilium.sh

# Add Cilium Helm repo
helm repo add cilium https://helm.cilium.io/
helm repo update

# Install Cilium
helm install cilium cilium/cilium \
  --version 1.15.0 \
  --namespace kube-system \
  --create-namespace \
  -f cilium-values.yaml

# Wait for Cilium to be ready
kubectl wait --for=condition=ready pod -n kube-system -l k8s-app=cilium --timeout=300s
kubectl wait --for=condition=ready pod -n kube-system -l k8s-app=hubble-relay --timeout=300s
kubectl wait --for=condition=ready pod -n kube-system -l k8s-app=hubble-ui --timeout=300s

# Verify installation
cilium status