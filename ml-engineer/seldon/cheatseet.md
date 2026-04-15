🚀 🔹 KUBECTL CHEATSHEET (daily use)
📦 Cluster & context
kubectl cluster-info
kubectl config get-contexts
kubectl config use-context docker-desktop
📋 Get resources
kubectl get pods
kubectl get pods -A
kubectl get pods -o wide

kubectl get svc
kubectl get deployments
kubectl get all -n default

kubectl get seldondeployments -n default
🔍 Debugging (VERY IMPORTANT 🔥)
kubectl describe pod <pod-name>
kubectl logs <pod-name>
kubectl logs -f <pod-name>   # streaming

kubectl get events --sort-by=.metadata.creationTimestamp
👉 For your case (ImagePullBackOff), ALWAYS:
kubectl describe pod <pod>
🧹 Delete / cleanup
kubectl delete pod <pod-name>
kubectl delete pods --all -n default

kubectl delete deployment <name>
kubectl delete deployment --all -n default

kubectl delete seldondeployment --all -n default
🚀 Apply / deploy
kubectl apply -f deployment.yaml
kubectl apply -f .
kubectl delete -f deployment.yaml
🔁 Restart / scale
kubectl rollout restart deployment <name>

kubectl scale deployment <name> --replicas=0
kubectl scale deployment <name> --replicas=1
🌐 Access services
kubectl port-forward svc/<service-name> 8080:80

kubectl proxy
🔐 Dashboard token
kubectl -n kubernetes-dashboard create token admin-user
📂 Exec into container
kubectl exec -it <pod-name> -- /bin/sh
🚀 🔹 HELM CHEATSHEET
Helm = package manager for Kubernetes
📦 Release management
helm list -A
helm list -n seldon-system
➕ Install chart
helm install seldon-core seldon-core-operator \
  --repo https://storage.googleapis.com/seldon-charts \
  -n seldon-system \
  --create-namespace
🔄 Upgrade
helm upgrade seldon-core seldon-core-operator \
  -n seldon-system
❌ Uninstall
helm uninstall seldon-core -n seldon-system
🔍 Debug Helm
helm status seldon-core -n seldon-system
helm get values seldon-core -n seldon-system
helm get manifest seldon-core -n seldon-system
📥 Add repo
helm repo add seldon https://storage.googleapis.com/seldon-charts
helm repo update
🔎 Search charts
helm search repo seldon
🔥 🔹 POWER USER ALIASES (HIGHLY RECOMMENDED)
Add to your .zshrc / .bashrc:
alias k='kubectl'
alias kgp='kubectl get pods'
alias kga='kubectl get all'
alias kdp='kubectl describe pod'
alias klog='kubectl logs -f'
alias kdel='kubectl delete'

alias h='helm'
alias hl='helm list -A'

🧠 🔹 YOUR CURRENT DEBUG FLOW (IMPORTANT)
Given your situation:
ImagePullBackOff ❌

👉 Use this exact flow:
kubectl get pods
kubectl describe pod <pod>
kubectl logs <pod>
Then fix:
Docker image name
Tag
Push to registry

🚀 🔹 BONUS (Seldon specific)
kubectl get seldondeployments
kubectl describe seldondeployment <name>

🔥 TL;DR (most used commands)
If you remember nothing else, remember this:
kubectl get pods -A
kubectl describe pod <pod>
kubectl logs -f <pod>

kubectl delete seldondeployment --all -n default

helm list -A
helm uninstall <release> -n <namespace>
