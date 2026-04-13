🚀 1. BUILD (you already did this correctly)
docker build -t mipjeiger/fraud-model -f ml-engineer/seldon/fraud/Dockerfile .
docker build -t mipjeiger/marketing-model -f ml-engineer/seldon/marketing/Dockerfile .
docker build -t mipjeiger/operational-model -f ml-engineer/seldon/operational/Dockerfile .
docker build -t mipjeiger/banking-llm -f llm/banking_llm/Dockerfile .

🏷️ 2. TAG (optional but recommended)
docker tag mipjeiger/fraud-model mipjeiger/fraud-model:latest
docker tag mipjeiger/marketing-model mipjeiger/marketing-model:latest
docker tag mipjeiger/operational-model mipjeiger/operational-model:latest
docker tag mipjeiger/banking-llm mipjeiger/banking-llm:latest

🔐 3. LOGIN (only once)
docker login

📤 4. PUSH (correct version)
docker push mipjeiger/fraud-model:latest
docker push mipjeiger/marketing-model:latest
docker push mipjeiger/operational-model:latest
docker push mipjeiger/banking-llm:latest

🚀 6. DEPLOY TO KUBERNETES (CORRECT PATHS)
Run from Bank-Project root:
kubectl apply -f ml-engineer/seldon/fraud/seldon.yaml
kubectl apply -f ml-engineer/seldon/marketing/seldon.yaml
kubectl apply -f ml-engineer/seldon/operational/seldon.yaml
kubectl apply -f llm/banking_llm/seldon.yaml

🔍 7. VERIFY (must do)
kubectl get pods
If something fails:
kubectl describe pod <pod-name>
kubectl logs <pod-name>

🌐 8. TEST (very important)
Port forward:
kubectl port-forward svc/fraud-model 8000:8000
curl -X POST http://localhost:8000/api/v1.0/predictions \
  -H "Content-Type: application/json" \
  -d '{"data":{"ndarray":[[1,2,3,4,5]]}}'

  🧠 FINAL FLOW
  BUILD → TAG → PUSH → UPDATE YAML → APPLY → TEST