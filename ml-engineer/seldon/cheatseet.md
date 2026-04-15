🚀 🔥 END-TO-END CHEATSHEET (STEP BY STEP)

🧩 0. Check cluster ready
kubectl cluster-info
kubectl get nodes

📦 1. Helm setup (Seldon Core)
✅ Check repo
helm repo list
You already have:
seldon
seldonio (duplicate but OK)
✅ Update repo
helm repo update
✅ Install Seldon Core
helm install seldon-core seldon/seldon-core-operator \
  --namespace seldon-system \
  --create-namespace \
  --set usageMetrics.enabled=false \
  --set istio.enabled=false \
  --set image.tag=1.17.0

--> output helm should be 
NAME: seldon-core
LAST DEPLOYED: Wed Apr 15 19:30:04 2026
NAMESPACE: seldon-system
STATUS: deployed
REVISION: 1
DESCRIPTION: Install complete
TEST SUITE: None

✅ Verify installation
helm list -n seldon-system
kubectl get pods -n seldon-system
You should see:
seldon-controller-manager → Running ✅

🧠 2. (IMPORTANT) Build & Push Docker Images
⚠️ This is where your current error is (ImagePullBackOff)
✅ Build images
docker build -t <dockerhub-username>/fraud-model:latest -f ml-engineer/seldon/fraud/Dockerfile .
docker build -t <dockerhub-username>/marketing-model:latest -f ml-engineer/seldon/marketing/Dockerfile .
docker build -t <dockerhub-username>/operational-model:latest -f ml-engineer/seldon/operational/Dockerfile .
docker build -t <dockerhub-username>/banking-llm:latest -f ml-engineer/LLM/banking_llm/Dockerfile .
✅ Login DockerHub
docker login
✅ Push images
docker push <dockerhub-username>/fraud-model:latest
docker push <dockerhub-username>/marketing-model:latest
docker push <dockerhub-username>/operational-model:latest
docker push <dockerhub-username>/banking-llm:latest

🚀 3. Deploy Seldon Models
kubectl apply -f ml-engineer/seldon/fraud/seldon.yaml
kubectl apply -f ml-engineer/seldon/marketing/seldon.yaml
kubectl apply -f ml-engineer/seldon/operational/seldon.yaml
kubectl apply -f ml-engineer/LLM/banking_llm/seldon.yaml
✅ Verify deployments
kubectl get seldondeployments
✅ Check pods
kubectl get pods -A
Expected:
STATUS → Running ✅ (NOT ImagePullBackOff)

🔍 4. Debug (if something fails)
🔥 Most important commands
kubectl describe pod <pod-name>
kubectl logs <pod-name>
kubectl get events --sort-by=.metadata.creationTimestamp
❗ If you see:
ImagePullBackOff ❌
👉 Fix:
wrong image name
not pushed
private repo (need secret)

🌐 5. Kubernetes Dashboard
✅ Get token
kubectl -n kubernetes-dashboard create token admin-user
✅ Run proxy
kubectl proxy
✅ Open UI
http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/
🔥 What to check in dashboard
Pods → status
Logs
Events
Seldon workloads

🧹 6. Clean & redeploy (VERY COMMON LOOP)
Delete all models
kubectl delete seldondeployment --all -n default
Or delete everything in default
kubectl delete all --all -n default
Redeploy
kubectl apply -f ml-engineer/

🔁 7. Full Dev Loop (REAL WORKFLOW)
# 1. Build
docker build -t <user>/fraud-model:latest .

# 2. Push
docker push <user>/fraud-model:latest

# 3. Deploy
kubectl apply -f seldon.yaml

# 4. Check
kubectl get pods

# 5. Debug
kubectl describe pod <pod>
kubectl logs <pod>
⚡ 🔥 POWER ALIASES (SAVE TIME)
alias k='kubectl'
alias kgp='kubectl get pods -A'
alias kgs='kubectl get seldondeployments'
alias kdp='kubectl describe pod'
alias klog='kubectl logs -f'
alias kdel='kubectl delete'
alias h='helm'
alias hl='helm list -A'

🧠 🔥 KEY INSIGHT (IMPORTANT FOR YOU)
Right now your bottleneck is NOT:
Helm ❌
Dashboard ❌
Seldon ❌

👉 It is:
Docker image not accessible ❌

✅ SUCCESS CHECKLIST
Before saying “done”, make sure:
 helm list -n seldon-system → deployed
 kubectl get pods -n seldon-system → Running
 docker push success
 kubectl get pods → Running (NOT ImagePullBackOff)
 Dashboard shows green pods