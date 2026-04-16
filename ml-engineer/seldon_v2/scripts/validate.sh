#!/usr/bin/env bash
# validate.sh — run from Bank-Project/ml-engineer/seldon_v2/
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; ERRORS=$((ERRORS+1)); }

ERRORS=0

echo "========================================"
echo " Seldon v2 — Pre-Apply Validation"
echo "========================================"

# 1. kubectl reachability
echo ""
echo "── Cluster connectivity ──"
if kubectl cluster-info &>/dev/null; then
  ok "kubectl can reach the cluster"
else
  fail "kubectl cannot reach the cluster — check kubeconfig"
fi

# 2. Required namespaces
echo ""
echo "── Namespaces ──"
for NS in seldon-system seldon-system-v2; do
  if kubectl get namespace "$NS" &>/dev/null; then
    ok "Namespace '$NS' exists"
  else
    warn "Namespace '$NS' missing — will be created on apply"
  fi
done

# 3. CRD presence
echo ""
echo "── CRDs ──"
for CRD in servers.mlops.seldon.io models.mlops.seldon.io pipelines.mlops.seldon.io; do
  if kubectl get crd "$CRD" &>/dev/null; then
    ok "CRD '$CRD' registered"
  else
    fail "CRD '$CRD' NOT found — apply the CRD yaml files first"
  fi
done

# 4. Secret existence
echo ""
echo "── Secrets ──"
if kubectl get secret minio-credentials -n seldon-system &>/dev/null; then
  ok "Secret 'minio-credentials' exists in seldon-system"
else
  fail "Secret 'minio-credentials' missing — apply infra/k8s/secret.yaml first"
fi

# 5. Dry-run servers
echo ""
echo "── Dry-run: servers ──"
if kubectl apply --dry-run=client -f servers/server.yaml &>/dev/null; then
  ok "servers/server.yaml — dry-run passed"
else
  fail "servers/server.yaml — dry-run FAILED"
fi

# 6. Dry-run models + pipelines
echo ""
echo "── Dry-run: models & pipelines ──"
for DOMAIN in fraud marketing operational; do
  for FILE in servers/$DOMAIN/*.yaml; do
    if kubectl apply --dry-run=client -f "$FILE" &>/dev/null; then
      ok "$FILE — dry-run passed"
    else
      fail "$FILE — dry-run FAILED"
    fi
  done
done

# 7. Kustomize build — use kubectl kustomize as fallback, never skip
echo ""
echo "── Kustomize build ──"
if command -v kustomize &>/dev/null; then
  KUSTOMIZE_CMD="kustomize build"
else
  KUSTOMIZE_CMD="kubectl kustomize"
fi
if $KUSTOMIZE_CMD servers/ &>/dev/null; then
  ok "kustomize build servers/ succeeded (via: $KUSTOMIZE_CMD)"
else
  fail "kustomize build servers/ FAILED (via: $KUSTOMIZE_CMD)"
fi

# 8. Image check — manifest inspect only (no download), 10s timeout
echo ""
echo "── Image references ──"
IMAGES=(
  "seldonio/mlserver:1.5.0"
  "mipjeiger/fraud-combiner:latest"
  "mipjeiger/marketing-combiner:latest"
  "mipjeiger/ops-combiner:latest"
)
if ! command -v docker &>/dev/null; then
  warn "docker not found — skipping image checks"
else
  for IMG in "${IMAGES[@]}"; do
    if timeout 10s docker manifest inspect "$IMG" &>/dev/null 2>&1; then
      ok "Image exists in registry: $IMG"
    else
      warn "Image not reachable (private, missing, or no internet): $IMG"
    fi
  done
fi

echo ""
echo "========================================"
if [ "$ERRORS" -eq 0 ]; then
  echo -e "${GREEN}All checks passed — safe to apply.${NC}"
else
  echo -e "${RED}$ERRORS check(s) failed — fix before applying.${NC}"
  exit 1
fi