# AlexandrAI - Deployment & Orchestration Guide

## 1. Local Development via Docker Compose

Run the entire distributed microservice stack locally using Docker Compose:

```bash
# Build and bring up all containers
docker-compose up --build -d

# View status of running containers
docker-compose ps

# Tail logs across microservices with request tracing
docker-compose logs -f api-gateway llm-worker
```

---

## 2. Kubernetes Deployment via kubectl

Apply pure Kubernetes manifests directly into the `alexandrai` namespace:

```bash
# Create namespace
kubectl create namespace alexandrai

# Apply ConfigMaps and Secrets
kubectl apply -f infra/k8s/configmap.yaml -n alexandrai
kubectl apply -f infra/k8s/secrets.yaml -n alexandrai
kubectl apply -f infra/k8s/pvc.yaml -n alexandrai

# Apply Deployments and Services
kubectl apply -f infra/k8s/deployments.yaml -n alexandrai
kubectl apply -f infra/k8s/services.yaml -n alexandrai
kubectl apply -f infra/k8s/ingress.yaml -n alexandrai
kubectl apply -f infra/k8s/hpa.yaml -n alexandrai
```

---

## 3. Helm Chart Deployment

Deploy parameterized Helm charts across environments (`dev`, `staging`, `prod`):

```bash
# Production installation
helm upgrade --install alexandrai infra/helm/alexandrai \
  --namespace alexandrai \
  --create-namespace \
  --values gitops/environments/prod/values.yaml

# Staging installation
helm upgrade --install alexandrai-staging infra/helm/alexandrai \
  --namespace alexandrai-staging \
  --create-namespace \
  --values gitops/environments/staging/values.yaml
```

---

## 4. GitOps Deployment via ArgoCD

Apply the declarative ArgoCD Application manifest:

```bash
kubectl apply -f gitops/argocd/application.yaml -n argocd
```
ArgoCD will continuously reconcile the cluster state against `infra/helm/alexandrai` and target environment configuration files.
