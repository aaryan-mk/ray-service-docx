# Docker Image Deployment Agent Guide

## What This Is

This guide lets you deploy GPU-accelerated code or heavy applications to a Kubernetes Ray cluster. Code is baked into Docker images, built automatically by Azure Pipelines (CI/CD), pushed to Azure Container Registry (ACR), and deployed to Kubernetes via Flux (a GitOps tool that watches repos and auto-applies changes).

There are two deployment patterns:
1. **GPU Worker** — Add Ray actors to the existing GPU worker image for GPU inference
2. **Standalone Service** — Create a new Docker-based HTTP service (API server, etc.)

Both patterns use the same infrastructure: Azure DevOps repos → Azure Pipelines → ACR → Flux → Kubernetes.

## When to Use This vs ConfigMap

| Condition | ConfigMap (agent-configmap.md) | Docker (this guide) |
|-----------|-------------------------------|---------------------|
| CPU only, small deps | Yes | Overkill |
| GPU/CUDA needed | No | Yes |
| Dependencies >500MB | No | Yes |
| Code >1MB | No | Yes |
| System packages (apt) | No | Yes |

---

## Infrastructure Overview

### Repositories

All code repos live in Azure DevOps. K8s manifests live in a separate repo. Flux watches the manifests repo and auto-applies changes.

| Component | Repo location | Purpose |
|-----------|--------------|---------|
| Your service code | Azure DevOps `data-ai` project | Docker image source |
| K8s manifests | `https://dev.azure.com/predictintel/Cybersmart-Next/_git/draft-deployment` | Defines how services run in K8s |

### How Deployment Works

```
Your code repo (Azure DevOps)
        │
        │ push to main + dev
        ▼
Azure Pipeline (auto-triggered)
        │
        │ builds Docker image
        ▼
Azure Container Registry (ACR): cybersmartstg.azurecr.io/data-ai/<your-service>:<tag>
        │
        │ Either: Flux auto-updates tag (for HelmRelease services)
        │ Or:     You manually restart the pod (for GPU workers)
        ▼
Kubernetes cluster applies new image
```

### Cluster Layout

```
┌─── ray namespace ──────────────────────────────────────────┐
│                                                             │
│  HEAD node (rayproject/ray:2.52.0-py310)                   │
│  ├── Coordinator (no tasks), Ray Dashboard (:8265)         │
│  ├── Ray Serve HTTP (:8000) — for ConfigMap apps           │
│  └── Ray Client (:10001) — for external services to call   │
│                                                             │
│  CPU worker (rayproject/ray:2.52.0-py310)                  │
│  └── Runs ConfigMap-deployed Ray Serve apps                │
│                                                             │
│  GPU worker (your-gpu-image:tag)                           │
│  ├── initContainer: installs heavy ML deps at startup      │
│  └── Ray actors for GPU inference                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─── your-namespace ─────────────────────────────────────────┐
│                                                             │
│  Your API service (your-api-image:tag)                     │
│  ├── HTTP server (FastAPI, Flask, etc.)                    │
│  ├── Connects to Ray cluster via Ray Client (:10001)       │
│  └── Connects to databases, storage, etc.                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Pattern 1: GPU Worker Deployment

Use this when you need to run Ray actors on GPU. Your code runs inside the existing GPU worker pod.

### How It Works

The GPU worker uses a **split dependency** approach:
- **Docker image**: Contains your code + light dependencies (keeps image small for CI/CD build agents)
- **initContainer**: Installs heavy ML packages (PyTorch, ONNX, etc.) at pod startup into a shared volume
- **Main container**: Your code runs here, finds heavy deps via `PYTHONPATH=/shared-packages:/app`

### Step 1: Create Your Service Repo

Create a new repo in Azure DevOps (`data-ai` project) with this structure:

```
your-gpu-service/
├── Dockerfile
├── azure-pipelines.yaml
├── your_tasks.py            # Ray actors and remote functions
├── certs/
│   └── proxy-hub-ca-1.crt   # Enterprise proxy CA cert (get from team)
└── README.md
```

### Step 2: Write Your Dockerfile

Base image must be `rayproject/ray:2.52.0-py310-cu121` (Ray + Python 3.10 + CUDA 12.1). Only install **light** deps — heavy ML deps go in the initContainer.

```dockerfile
FROM rayproject/ray:2.52.0-py310-cu121

USER root
COPY certs/proxy-hub-ca-1.crt /usr/local/share/ca-certificates/proxy-hub-ca.crt
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgl1-mesa-glx libglib2.0-0 \
    && update-ca-certificates && rm -rf /var/lib/apt/lists/*
USER ray

WORKDIR /app

# ONLY light deps — heavy deps (torch, onnx, etc.) via initContainer
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    scikit-learn==1.3.2 \
    Pillow>=9.0.0 \
    # ...your light deps...

COPY your_tasks.py .
# COPY any other files...

ENV PYTHONPATH="/app" OMP_NUM_THREADS="1"
EXPOSE 8000
```

**Why light deps only**: The full image with heavy ML packages would be ~11GB, exceeding Azure DevOps build agent disk (~10GB).

### Step 3: Write Your Ray Actors

Your Python file MUST start with the `/shared-packages` path trick so it can find heavy deps installed by the initContainer:

```python
# your_tasks.py
import sys, os

# REQUIRED: Add shared-packages to path for heavy deps from initContainer
SHARED_PACKAGES_DIR = "/shared-packages"
if os.path.exists(SHARED_PACKAGES_DIR) and SHARED_PACKAGES_DIR not in sys.path:
    sys.path.insert(0, SHARED_PACKAGES_DIR)

import ray
import numpy as np

# Now you can import heavy deps (torch, onnxruntime, etc.)

@ray.remote(num_gpus=0.5)
class MyGPUActor:
    def __init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load models, set up state...

    def process(self, data):
        # Your GPU inference logic
        return {"result": "processed"}

@ray.remote(num_gpus=0.5)
class AnotherActor:
    def __init__(self):
        import onnxruntime as ort
        self.session = ort.InferenceSession("/tmp/models/model.onnx",
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def infer(self, input_data):
        return self.session.run(None, {"input": input_data})
```

### Step 4: Set Up Azure Pipeline

Create `azure-pipelines.yaml`:

```yaml
trigger:
  - main
  - dev

variables:
  AzureSubscription: 'S2T Sela-data-ai'
  KeyVaultName: 'predicintel-devops-kv'

resources:
  repositories:
    - repository: template
      type: git
      name: Cybersmart-Next/template

extends:
  template: project-pipelines.yml@template
  parameters:
    registryHybrid: false
    DockerFiles:
      - registryUrl: 'cybersmartstg.azurecr.io/data-ai/your-gpu-service'
        file: '$(Build.SourcesDirectory)/Dockerfile'
        displayName: 'Build Main Docker Image'
        context: '$(Build.SourcesDirectory)/'
        jobName: job1
    DevRegistries:
      - registryUrl: 'cybersmartstg.azurecr.io/data-ai/your-gpu-service'
    UATRegistries:
      - registryUrl: 'cybersmartstg.azurecr.io/data-ai/your-gpu-service'
```

### Step 5: Add GPU Worker Group to ray-service.yaml

Clone the draft-deployment repo and edit `dev/ray-service.yaml`. Add a new GPU worker group under `workerGroupSpecs:` (or modify the existing one if you're replacing it):

```yaml
      # Your GPU worker group
      - replicas: 1
        minReplicas: 1
        maxReplicas: 3
        groupName: your-gpu-worker-group
        rayStartParams:
          block: 'true'
          num-gpus: "1"
        template:
          metadata:
            labels:
              component: your-gpu-worker
          spec:
            priorityClassName: gpu-workload-high
            imagePullSecrets:
              - name: regcred
            tolerations:
              - key: "sku"
                operator: "Equal"
                value: "gpu"
                effect: "NoSchedule"
              - key: "nvidia.com/gpu"
                operator: "Exists"
                effect: "NoSchedule"
            affinity:
              nodeAffinity:
                requiredDuringSchedulingIgnoredDuringExecution:
                  nodeSelectorTerms:
                    - matchExpressions:
                        - key: gpu/vram
                          operator: In
                          values:
                            - 16gb
            initContainers:
              - name: install-heavy-deps
                image: rayproject/ray:2.52.0-py310-cu121
                command: ["/bin/bash", "-c"]
                args:
                  - |
                    set -e
                    # Install your heavy ML dependencies
                    pip install --target=/shared-packages --no-cache-dir \
                      torch torchvision --index-url https://download.pytorch.org/whl/cu121
                    pip install --target=/shared-packages --no-cache-dir \
                      onnxruntime-gpu nvidia-cudnn-cu12 "numpy<2"
                    # Add more packages as needed...

                    # Fix numpy (OpenCV/other packages may install numpy 2.x)
                    pip install --target=/shared-packages --no-cache-dir --upgrade "numpy<2"

                    # Remove packages that conflict with Ray
                    rm -rf /shared-packages/protobuf* /shared-packages/google/protobuf* 2>/dev/null || true
                    rm -rf /shared-packages/click* 2>/dev/null || true

                    # Download models if needed
                    mkdir -p /tmp/models
                    if [ ! -f /tmp/models/your_model.onnx ]; then
                      wget -q -O /tmp/models/your_model.onnx https://your-model-url
                    fi

                    echo "=== Done ==="
                volumeMounts:
                  - mountPath: /shared-packages
                    name: python-packages
                  - mountPath: /tmp/models
                    name: model-cache
                resources:
                  limits: { cpu: "2", memory: "4Gi" }
                  requests: { cpu: "1", memory: "2Gi" }
            containers:
              - name: ray-worker
                image: cybersmartstg.azurecr.io/data-ai/your-gpu-service:dev
                imagePullPolicy: Always
                resources:
                  limits: { cpu: "6", memory: "12Gi", nvidia.com/gpu: 1 }
                  requests: { cpu: "6", memory: "4Gi", nvidia.com/gpu: 1 }
                volumeMounts:
                  - mountPath: /tmp/ray
                    name: log-volume
                  - mountPath: /tmp/models
                    name: model-cache
                  - mountPath: /shared-packages
                    name: python-packages
                env:
                  - name: NVIDIA_VISIBLE_DEVICES
                    value: "all"
                  - name: PYTHONPATH
                    value: "/shared-packages:/app"
                  - name: LD_LIBRARY_PATH
                    value: "/shared-packages/nvidia/cudnn/lib:/shared-packages/nvidia/cuda_runtime/lib"
            volumes:
              - name: log-volume
                emptyDir: {}
              - name: model-cache
                persistentVolumeClaim:
                  claimName: your-models-pvc    # Create a PVC or use emptyDir
              - name: python-packages
                emptyDir:
                  sizeLimit: 10Gi
```

### Step 6: Push and Deploy

```bash
# 1. Push your service code
cd your-gpu-service
git add -A && git commit -m "Initial GPU service" && git push origin main && git push origin dev

# 2. Push K8s manifests
cd draft-deployment
git add dev/ray-service.yaml && git commit -m "Add your-gpu-service worker" && git push origin main

# 3. Wait for pipeline (~5 min), then restart GPU pod
kubectl delete pod -n ray -l component=your-gpu-worker

# 4. Verify
kubectl get pods -n ray -l component=your-gpu-worker -w
kubectl logs -n ray -l component=your-gpu-worker -c install-heavy-deps --tail=20
kubectl logs -n ray -l component=your-gpu-worker -c ray-worker --tail=20
```

### Updating GPU Worker Code

```bash
cd your-gpu-service
# edit files...
git add -A && git commit -m "Update" && git push origin main && git push origin dev
# Wait for pipeline (~5 min)
kubectl delete pod -n ray -l component=your-gpu-worker
```

---

## Pattern 2: Standalone API Service

Use this when you need an HTTP API server that connects to the Ray cluster (or runs independently). Your service runs as a separate pod in its own namespace.

### Step 1: Create Your Service Repo

Create a new repo in Azure DevOps (`data-ai` project):

```
your-api-service/
├── Dockerfile
├── azure-pipelines.yaml
├── requirements.txt
├── private-requirements.txt  # Optional, failures ignored
├── main.py                   # Entry point
├── certs/
│   └── proxy-hub-ca-1.crt
└── README.md
```

### Step 2: Write Your Dockerfile

```dockerfile
FROM python:3.10-slim

COPY certs/proxy-hub-ca-1.crt /usr/local/share/ca-certificates/proxy-hub-ca.crt
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && update-ca-certificates && rm -rf /var/lib/apt/lists/*

ARG VERSION="" GIT_REF="" BUILD_TIME=""

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY private-requirements.txt .
RUN pip install --no-cache-dir -r private-requirements.txt || true
COPY . .

ENV ENVIRONMENT="production" APP_PORT=8000
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

All deps go in the Docker image (no initContainer needed for CPU-only services).

### Step 3: Write requirements.txt

```
# Your framework
fastapi
uvicorn
pydantic
pydantic-settings

# Ray Client — MUST match cluster version exactly
ray[client]==2.52.0

# Your other deps
httpx
asyncpg
numpy<2
# ...
```

**Critical**: `ray[client]==2.52.0` must match the Ray cluster version. Mismatch = handshake failures.

### Step 4: Set Up Azure Pipeline

```yaml
trigger:
  - main
  - dev

variables:
  AzureSubscription: 'S2T Sela-data-ai'
  KeyVaultName: 'predicintel-devops-kv'

resources:
  repositories:
    - repository: template
      type: git
      name: Cybersmart-Next/template

extends:
  template: project-pipelines.yml@template
  parameters:
    registryHybrid: false
    DockerFiles:
      - registryUrl: 'cybersmartstg.azurecr.io/data-ai/your-api-service'
        file: '$(Build.SourcesDirectory)/Dockerfile'
        displayName: 'Build Main Docker Image'
        context: '$(Build.SourcesDirectory)/'
        jobName: job1
    DevRegistries:
      - registryUrl: 'cybersmartstg.azurecr.io/data-ai/your-api-service'
    UATRegistries:
      - registryUrl: 'cybersmartstg.azurecr.io/data-ai/your-api-service'
```

### Step 5: Create HelmRelease in draft-deployment

Create `draft-deployment/dev/your-api-service.yaml`:

```yaml
apiVersion: helm.toolkit.fluxcd.io/v2
kind: HelmRelease
metadata:
  name: your-api-service
  namespace: default              # HelmRelease resource lives here
spec:
  targetNamespace: your-namespace # Pod deploys to this namespace
  interval: 1m
  chart:
    spec:
      chart: microservice
      reconcileStrategy: ChartVersion
      sourceRef:
        kind: HelmRepository
        name: s2t-acr-helm-repo
  values:
    image:
      repository: cybersmartstg.azurecr.io/data-ai/your-api-service # {"$imagepolicy": "flux-system:your-api-service-image-policy:name"}
      tag: latest # {"$imagepolicy": "flux-system:your-api-service-image-policy:tag"}
      pullPolicy: Always
    imagePullSecrets:
      - name: regcred-acr
    imageAutoUpdate:
      enabled: true
      fluxPullSecret: regcred-acr
      fluxApiVersion: image.toolkit.fluxcd.io/v1
    env:
      ENVIRONMENT: development
      APP_PORT: "8000"
      # Ray Cluster
      RAY_ADDRESS: "ray://rayservice-head-svc.ray.svc.cluster.local:10001"
      RAY_NAMESPACE: "default"
      # Add your env vars here
      # DATABASE_URL: "postgresql://..."
      # API_KEY: "..."
    replicaCount: 1
    containerPort: 8000
    resources:
      limits:
        cpu: "1"
        memory: 2Gi
      requests:
        cpu: 250m
        memory: 512Mi
    livenessProbe:
      httpGet:
        path: /health        # Your health endpoint
        port: 8000
      initialDelaySeconds: 15
      periodSeconds: 30
    readinessProbe:
      httpGet:
        path: /ready          # Your readiness endpoint
        port: 8000
      initialDelaySeconds: 10
      periodSeconds: 10
```

**Important**: The `# {"$imagepolicy": ...}` comments are Flux markers — Flux auto-updates the `tag` when a new image is pushed to ACR. **Never remove these comments.**

### Step 6: Push and Deploy

```bash
# 1. Push your service code
cd your-api-service
git add -A && git commit -m "Initial API service" && git push origin main && git push origin dev

# 2. Push K8s manifest
cd draft-deployment
git add dev/your-api-service.yaml && git commit -m "Add your-api-service HelmRelease" && git push origin main

# 3. AUTOMATIC: Pipeline builds image → Flux detects new tag → deploys pod

# 4. Verify (~5 min)
kubectl get pods -n your-namespace
kubectl logs -n your-namespace -l app.kubernetes.io/name=your-api-service-microservice --tail=20

# Test
kubectl port-forward svc/your-namespace-your-api-service-microservice 8080:80 -n your-namespace &
curl http://localhost:8080/health
```

### Updating API Service Code

```bash
cd your-api-service
# edit files...
git add -A && git commit -m "Update" && git push origin main && git push origin dev
# Fully automatic: Pipeline → ACR → Flux updates tag → pod restarts
```

---

## Key Concepts

### initContainer Pattern (GPU Workers)

GPU workers use an initContainer because heavy ML packages (~4-5GB) don't fit in the Docker image during CI/CD build. The initContainer runs at pod startup and installs packages into a shared volume.

```
Pod startup:
1. initContainer: install-heavy-deps (~5 min)
   └── pip install --target=/shared-packages torch onnxruntime-gpu ...
2. initContainer: wait-gcs-ready (auto-generated by KubeRay)
   └── Waits for Ray head to be ready
3. Main container starts
   └── PYTHONPATH=/shared-packages:/app → finds packages from step 1
```

**Known package conflicts with Ray:**
- `protobuf` — PaddlePaddle/ONNX install v6.x, Ray needs <5.0 → `rm -rf protobuf*`
- `click` — PaddleOCR installs it, conflicts with Ray CLI → `rm -rf click*`
- `numpy 2.x` — OpenCV installs it, ONNX Runtime needs <2 → reinstall `numpy<2` last

**Critical env var**: `LD_LIBRARY_PATH` must include `/shared-packages/nvidia/cudnn/lib` or ONNX Runtime silently falls back to CPU.

### Flux ImagePolicy (API Services)

For HelmRelease services, Flux automatically detects new images in ACR and updates the `tag:` field in your YAML. This is why pushing code → building image → deploying is fully automatic with no manual pod restart.

The `# {"$imagepolicy": ...}` comment on the `tag:` and `repository:` lines tells Flux which ImagePolicy to use. Never remove these.

### Version Constraints

All Ray components must use the same version. Currently `2.52.0`:

| Component | Where to set |
|-----------|-------------|
| HEAD node image | `ray-service.yaml` headGroupSpec: `rayproject/ray:2.52.0-py310` |
| CPU worker image | `ray-service.yaml` workerGroupSpecs: `rayproject/ray:2.52.0-py310` |
| GPU worker base image | Your Dockerfile: `rayproject/ray:2.52.0-py310-cu121` |
| GPU initContainer image | `ray-service.yaml` initContainers: `rayproject/ray:2.52.0-py310-cu121` |
| API service Ray client | Your requirements.txt: `ray[client]==2.52.0` |

**Change one → change all.** Mismatch = handshake failures.

### Push Rules

| Repo type | Push to | Deployment |
|-----------|---------|------------|
| Service code (Azure DevOps `data-ai`) | `main` AND `dev` | Pipeline builds image |
| K8s manifests (`draft-deployment`) | `main` only | Flux auto-applies |

For GPU workers: after pipeline finishes, manually restart pod with `kubectl delete pod`.
For API services: Flux handles everything automatically.

---

## Verification

### Ray Cluster

```bash
kubectl get pods -n ray
kubectl logs <pod> -n ray -c install-heavy-deps --tail=20   # initContainer
kubectl logs <pod> -n ray -c ray-worker --tail=20            # main container

# GPU working?
kubectl exec <pod> -n ray -- python3 -c "
import sys; sys.path.insert(0, '/shared-packages')
import torch; print('CUDA:', torch.cuda.is_available())
"
```

### API Services

```bash
kubectl get pods -n your-namespace
kubectl logs -n your-namespace -l app.kubernetes.io/name=your-api-service-microservice --tail=20
kubectl port-forward svc/... 8080:80 -n your-namespace &
curl http://localhost:8080/health
```

### Flux

```bash
kubectl get kustomization -n flux-system draft-deployments -o jsonpath='{.status.lastAppliedRevision}'
flux reconcile kustomization draft-deployments -n flux-system  # force
```

---

## Troubleshooting

| Problem | Check | Fix |
|---------|-------|-----|
| GPU pod stuck in Init | `kubectl logs <pod> -n ray -c install-heavy-deps` | Pip timeout, disk space, network |
| ONNX on CPU not GPU | Check `LD_LIBRARY_PATH` env var | Must include `/shared-packages/nvidia/cudnn/lib` |
| API can't reach Ray | `kubectl get svc -n ray \| grep head` | HEAD svc must exist with port 10001 |
| Pipeline not triggering | Azure DevOps → Pipelines | Push must be to `main` or `dev` |
| Flux not updating | `flux get kustomizations -n flux-system` | `flux reconcile kustomization draft-deployments -n flux-system` |
| New pod has old image | Pipeline still building | Wait for pipeline, then restart pod |
| Ray version mismatch | Compare all 5 locations | Must all match |
| Package conflicts with Ray | Ray CLI crashes or import errors | `rm -rf /shared-packages/<conflicting-package>*` in initContainer |
