#!/usr/bin/env bash
# apply-order.sh — run from Bank-Project/ml-engineer/seldon_v2/
set -euo pipefail

echo "==> [1/7] Applying CRDs (Server, Model, Pipeline)..."
kubectl apply -f mlops.seldon.io_servers.yaml
kubectl apply -f mlops.seldon.io_models.yaml
kubectl apply -f mlops.seldon.io_pipelines.yaml
kubectl wait --for=condition=Established crd/servers.mlops.seldon.io --timeout=60s
kubectl wait --for=condition=Established crd/models.mlops.seldon.io --timeout=60s
kubectl wait --for=condition=Established crd/pipelines.mlops.seldon.io --timeout=60s

echo "==> [2/7] Applying namespace..."
kubectl apply -f infra/k8s/namespace.yaml

echo "==> [3/7] Applying secret (MinIO credentials)..."
kubectl apply -f infra/k8s/secret.yaml

echo "==> [4/7] Applying ingress..."
kubectl apply -f infra/k8s/ingress.yaml

echo "==> [5/7] Applying servers (sklearn + xgboost)..."
kubectl apply -f servers/server.yaml

echo "==> [6/7] Applying models (all domains)..."
kubectl apply -f servers/fraud/fraud-model.yaml
kubectl apply -f servers/fraud/fraud-xgb.yaml
kubectl apply -f servers/marketing/marketing-model.yaml
kubectl apply -f servers/marketing/marketing-xgb.yaml
kubectl apply -f servers/operational/operational-model.yaml
kubectl apply -f servers/operational/operational-xgb.yaml

echo "==> [7/7] Applying pipelines (all domains)..."
kubectl apply -f servers/fraud/fraud-pipeline.yaml
kubectl apply -f servers/marketing/marketing-pipeline.yaml
kubectl apply -f servers/operational/operational-pipeline.yaml

echo ""
echo "✅ All resources applied. Check status with:"
echo "   kubectl get servers,models,pipelines -n seldon-system"