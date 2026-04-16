#!/usr/bin/env bash
# apply-order.sh — applies resources in the correct dependency order
# Run from: Bank-Project/ml-engineer/seldon_v2/
set -euo pipefail

echo "==> [1/6] Applying CRD..."
kubectl apply -f mlops.seldon.io_servers.yaml
kubectl wait --for=condition=Established crd/servers.mlops.seldon.io --timeout=60s

echo "==> [2/6] Applying namespace..."
kubectl apply -f infra/k8s/namespace.yaml

echo "==> [3/6] Applying secret (MinIO credentials)..."
kubectl apply -f infra/k8s/secret.yaml

echo "==> [4/6] Applying ingress..."
kubectl apply -f infra/k8s/ingress.yaml

echo "==> [5/6] Applying servers (sklearn + xgboost)..."
kubectl apply -f servers/server.yaml

echo "==> [6/6] Applying models & pipelines (all domains)..."
kubectl apply -f servers/fraud/
kubectl apply -f servers/marketing/
kubectl apply -f servers/operational/

echo ""
echo "✅ All resources applied. Check status with:"
echo "   kubectl get servers,models,pipelines -n seldon-system"