# Docker Image Deployment Agent Guide

## What This Is

This guide lets you deploy GPU-accelerated code or heavy applications to a Kubernetes cluster. Code is baked into Docker images, built automatically by Azure Pipelines (CI/CD), pushed to Azure Container Registry (ACR), and deployed to Kubernetes via Flux (a GitOps tool that watches repos and auto-applies changes to the cluster).

There are three separate git repositories that work together:
- **face-ocr-worker** — GPU Ray actors (ML inference code)
- **face-ocr-api** — FastAPI HTTP server (handles requests, delegates to GPU actors)
- **draft-deployment** — Kubernetes YAML manifests (defines how everything runs)

## When to Use This vs ConfigMap

| Condition | ConfigMap (agent-configmap.md) | Docker (this guide) |
|-----------|-------------------------------|---------------------|
| CPU only, small deps | Yes | Overkill |
| GPU/CUDA needed | No | Yes |
| Dependencies >500MB | No | Yes |
| Code >1MB | No | Yes |
| System packages (apt) | No | Yes |

---

## Repositories

### Clone URLs and Push Rules

```bash
# GPU worker (Ray actors)
git clone https://dev.azure.com/predictintel/data-ai/_git/face-ocr-worker

# API server (FastAPI)
git clone https://dev.azure.com/predictintel/data-ai/_git/face-ocr-api

# K8s manifests
git clone https://dev.azure.com/predictintel/Cybersmart-Next/_git/draft-deployment
```

| Repo | Push to | What happens |
|------|---------|-------------|
| `face-ocr-worker` | `main` AND `dev` | Pipeline builds Docker image → you manually restart GPU pod |
| `face-ocr-api` | `main` AND `dev` | Pipeline builds Docker image → Flux auto-deploys (no manual restart) |
| `draft-deployment` | `main` only | Flux auto-applies K8s YAML to cluster |

### Architecture

```
face-ocr-worker repo                face-ocr-api repo
   (ray_tasks.py)                      (FastAPI)
        │                                  │
        │ push main+dev                    │ push main+dev
        ▼                                  ▼
   Azure Pipeline                     Azure Pipeline
        │                                  │
        ▼                                  ▼
   ACR image:                         ACR image:
   face-ocr-worker:dev                face-ocr-api:<semver>
        │                                  │
        │ manual pod restart               │ Flux auto-updates tag
        ▼                                  ▼
┌─── ray namespace ──────────────┐  ┌── cyber namespace ──────────┐
│                                │  │                              │
│  HEAD node (coordinator)       │  │  face-ocr-api pod            │
│  ├─ Ray Serve (port 8000)     │  │  ├─ FastAPI (port 8000)      │
│  ├─ Dashboard (port 8265)     │◄─┤  ├─ Ray Client → HEAD:10001 │
│  └─ Ray Client (port 10001)   │  │  ├─ PostgreSQL               │
│                                │  │  ├─ ChromaDB                 │
│  GPU worker                    │  │  └─ DOSASHOP (blob storage)  │
│  ├─ initContainer: heavy deps  │  │                              │
│  ├─ face-ocr-worker:dev image  │  └──────────────────────────────┘
│  ├─ FaceRecognitionActor (GPU) │
│  └─ OCRActor (GPU)             │
│                                │
│  CPU worker (ConfigMap apps)   │
│  └─ Not relevant here          │
└────────────────────────────────┘

draft-deployment repo ──push main──▶ Flux auto-applies to cluster
```

---

## Skill: Update GPU Worker Code

When you need to change ML inference logic (face detection, embeddings, OCR, or add new actors).

### What to Edit

The `face-ocr-worker` repo. Main file is `ray_tasks.py`. Structure:

```
face-ocr-worker/
├── Dockerfile              # Light deps only (see Appendix A)
├── azure-pipelines.yaml    # Pipeline config (see Appendix C)
├── requirements.txt        # Reference list
├── ray_tasks.py            # All Ray actors + remote functions
├── certs/
│   └── proxy-hub-ca-1.crt  # Enterprise proxy CA cert
└── README.md
```

**Key rule**: `ray_tasks.py` (and any new Python file you add) MUST start with the `/shared-packages` path trick so it can find heavy deps installed by the initContainer:

```python
import sys, os
SHARED_PACKAGES_DIR = "/shared-packages"
if os.path.exists(SHARED_PACKAGES_DIR) and SHARED_PACKAGES_DIR not in sys.path:
    sys.path.insert(0, SHARED_PACKAGES_DIR)
```

### Push Workflow

```bash
cd face-ocr-worker

# 1. Edit code
# 2. Commit
git add -A && git commit -m "Description of change"

# 3. Push to BOTH branches
git push origin main && git push origin dev

# 4. WAIT for pipeline to finish (~5 min)
#    Check: Azure DevOps → data-ai project → Pipelines

# 5. AFTER pipeline finishes, restart GPU pod (pulls new image)
kubectl delete pod -n ray -l component=gpu-worker

# 6. Wait ~5 min (initContainer installs packages, then main container starts)
kubectl get pods -n ray -l component=gpu-worker -w

# 7. Verify
kubectl logs -n ray -l component=gpu-worker -c ray-worker --tail=20
```

**CRITICAL**: Wait for the pipeline to push the new image BEFORE restarting the pod. `imagePullPolicy: Always` means it pulls latest on restart — if the pipeline isn't done, it pulls the old image.

### Adding a New Actor

Add to `ray_tasks.py`:

```python
@ray.remote(num_gpus=0.5)  # or num_cpus=1 for CPU-only
class MyNewActor:
    def __init__(self):
        import torch  # available from /shared-packages
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def process(self, data):
        return {"result": "done"}
```

Push face-ocr-worker → wait for pipeline → restart pod → then add API endpoint in face-ocr-api to call it.

### Adding Light Dependencies

Small pure-Python packages: add to the `pip install` block in `Dockerfile`, push.

### Adding Heavy Dependencies

Large ML packages (>100MB, compiled, CUDA): add to the initContainer, NOT the Dockerfile. See [Skill: Modify initContainer](#skill-modify-initcontainer).

---

## Skill: Update API Server Code

When you need to change HTTP endpoints, job management, database logic, or how the API talks to Ray actors.

### What to Edit

The `face-ocr-api` repo. Structure:

```
face-ocr-api/
├── Dockerfile                # python:3.10-slim (see Appendix A)
├── azure-pipelines.yaml      # Pipeline config (see Appendix C)
├── requirements.txt          # All pip deps (see Appendix A)
├── private-requirements.txt  # Optional internal packages
├── main.py                   # Entry: uvicorn main:app
├── core/
│   ├── config.py             # Env var settings (pydantic-settings)
│   ├── server.py             # FastAPI app + router registration
│   ├── log_config.py         # Logging
│   └── ray_client.py         # Ray Client connection management
├── app/
│   ├── api/v1/
│   │   ├── ping.py           # GET /api/v1/ping, /health, /ready
│   │   ├── faces.py          # POST /api/v1/faces/upload, GET /status, /results
│   │   └── ocr.py            # POST /api/v1/ocr/upload, GET /status, /results
│   ├── services/
│   │   └── ray_service.py    # Creates/manages Ray actors on cluster
│   └── models/               # Pydantic schemas
├── certs/
│   └── proxy-hub-ca-1.crt
└── README.md
```

**Key**: `ray[client]==2.52.0` in requirements.txt MUST match the cluster Ray version exactly.

### Push Workflow

```bash
cd face-ocr-api

# 1. Edit code
# 2. Commit
git add -A && git commit -m "Description of change"

# 3. Push to BOTH branches
git push origin main && git push origin dev

# 4. FULLY AUTOMATIC: Pipeline builds → ACR → Flux ImagePolicy → HelmRelease updates → pod restarts

# 5. Verify (~5 min)
kubectl get pods -n cyber -l app.kubernetes.io/name=face-ocr-api-microservice
kubectl logs -n cyber -l app.kubernetes.io/name=face-ocr-api-microservice --tail=20

# 6. Test
kubectl port-forward svc/cyber-face-ocr-api-microservice 8080:80 -n cyber &
curl http://localhost:8080/api/v1/ping
```

### Adding a New Endpoint

1. Create `app/api/v1/my_feature.py`:
   ```python
   from fastapi import APIRouter
   router = APIRouter(prefix="/api/v1/my-feature", tags=["my-feature"])

   @router.post("/process")
   async def process(data: dict):
       return {"result": "ok"}
   ```
2. Register in `core/server.py`:
   ```python
   from app.api.v1 import my_feature
   app.include_router(my_feature.router)
   ```
3. Push (Flux auto-deploys).

### Adding Dependencies

Add to `requirements.txt`, push. Docker build installs them.

---

## Skill: Update Kubernetes Config

When you need to change env vars, resource limits, initContainer packages, or deployment configuration.

### What to Edit

The `draft-deployment` repo. Two files matter:

```
draft-deployment/dev/
├── ray-service.yaml     # RayService: HEAD + CPU worker + GPU worker + initContainer
└── face-ocr-api.yaml    # HelmRelease: face-ocr-api pod config
```

### Push Workflow

```bash
cd draft-deployment
git add dev/ && git commit -m "Description" && git push origin main
# Flux applies in ~1-2 min
kubectl get kustomization -n flux-system draft-deployments -o jsonpath='{.status.lastAppliedRevision}'
```

### ray-service.yaml — Navigating Sections

| Search for this | You found | What you can do |
|----------------|-----------|----------------|
| `kind: PersistentVolumeClaim` | PVC for ML models | Don't touch |
| `serveConfigV2: \|` | Ray Serve app config | Edit for ConfigMap apps (see agent-configmap.md) |
| `name: ray-head` | HEAD node container | Edit for ConfigMap volumes only |
| `groupName: default-worker-group` | CPU worker | Edit for ConfigMap volumes only |
| `groupName: gpu-worker-group` | GPU worker group | Edit (your Docker deployment target) |
| `name: install-heavy-deps` | initContainer script | Edit to add/change heavy deps |
| `image: cybersmartstg.azurecr.io/data-ai/face-ocr-worker` | GPU worker image ref | Rarely edit |
| `PYTHONPATH` / `LD_LIBRARY_PATH` | GPU worker env vars | Edit to add env vars |

### face-ocr-api.yaml — Navigating Sections

| Search for this | You found | What you can do |
|----------------|-----------|----------------|
| `repository:` (with `$imagepolicy` comment) | Docker image URL | **Never edit** — Flux auto-manages this |
| `tag:` (with `$imagepolicy` comment) | Image version | **Never edit** — Flux auto-updates this |
| `env:` | Environment variables | Add/change vars here |
| `resources:` | CPU/memory limits | Change if pod needs more |
| `livenessProbe:` / `readinessProbe:` | Health checks | Change if endpoints change |

**Never remove `# {"$imagepolicy": ...}` comments** — Flux needs them.

---

## Skill: Modify initContainer

When you need to add heavy ML packages to the GPU worker.

### How It Works

The GPU worker pod runs an initContainer before the main container. It installs ~4-5GB of packages into `/shared-packages` (an emptyDir shared between initContainer and main container). The main container finds them via `PYTHONPATH=/shared-packages:/app`.

### Current Install Order

| Step | Package | Install command |
|------|---------|----------------|
| 1 | PyTorch + CUDA 12.1 | `pip install --target=/shared-packages torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| 2 | PaddlePaddle + PaddleOCR | `pip install --target=/shared-packages paddlepaddle-gpu "paddleocr>=2.7.0,<3.0.0"` |
| 3 | EasyOCR | `pip install --target=/shared-packages easyocr>=1.7.0` |
| 4 | ONNX Runtime + cuDNN 9 | `pip install --target=/shared-packages onnxruntime-gpu nvidia-cudnn-cu12 "numpy<2"` |
| 5 | OpenCV | `pip install --target=/shared-packages opencv-python-headless==4.8.1.78` |
| 6 | numpy fix | `pip install --target=/shared-packages --upgrade "numpy<2"` |
| Cleanup | Remove conflicts | `rm -rf /shared-packages/protobuf* click*` |

Then it downloads ML models to PVC (skips if already present):
- SCRFD (17MB): `https://huggingface.co/immich-app/buffalo_l/resolve/main/detection/model.onnx` → `/tmp/models/scrfd_10g.onnx`
- ArcFace (167MB): `https://huggingface.co/immich-app/buffalo_l/resolve/main/recognition/model.onnx` → `/tmp/models/w600k_r50.onnx`

### Why Each Fix Exists

| Fix | Why |
|-----|-----|
| Pin PaddleOCR <3.0.0 | v3.x has breaking API changes |
| Install `nvidia-cudnn-cu12` | ONNX Runtime 1.19+ needs cuDNN 9, not 8 |
| Reinstall numpy<2 after OpenCV | OpenCV pulls numpy 2.x, ONNX crashes with it |
| Remove protobuf | PaddlePaddle installs protobuf 6.x, Ray needs <5.0 |
| Remove click | PaddleOCR's click conflicts with Ray CLI |

### Adding a New Package

1. Open `draft-deployment/dev/ray-service.yaml`
2. Find `name: install-heavy-deps` → find the `args:` block (long bash script)
3. Add your pip install BEFORE the numpy fix and cleanup:
   ```bash
   echo "[NEW] Installing my-package..."
   pip install --target=/shared-packages --no-cache-dir my-package>=1.0.0
   ```
4. Push `draft-deployment` to `main`
5. Restart GPU pod: `kubectl delete pod -n ray -l component=gpu-worker`

### Volumes

| Volume | Type | Size | Mount | Who uses it |
|--------|------|------|-------|-------------|
| `python-packages` | emptyDir | 10Gi | `/shared-packages` | initContainer writes, main container reads |
| `model-cache` | PVC (azurefile, RWX) | 1Gi | `/tmp/models` | initContainer writes (once), main container reads |
| `log-volume` | emptyDir | — | `/tmp/ray` | Main container |

### Environment Variables on GPU Worker

| Variable | Value | Critical? |
|----------|-------|-----------|
| `PYTHONPATH` | `/shared-packages:/app` | Yes — finds initContainer packages |
| `LD_LIBRARY_PATH` | `/shared-packages/nvidia/cudnn/lib:/shared-packages/nvidia/cuda_runtime/lib` | Yes — without this ONNX falls back to CPU |
| `NVIDIA_VISIBLE_DEVICES` | `all` | Yes — GPU access |
| `RAY_memory_monitor_refresh_ms` | `0` | No — disables memory monitor |

---

## Skill: Change face-ocr-api Environment Variables

1. Open `draft-deployment/dev/face-ocr-api.yaml`
2. Find the `env:` section under `values:`
3. Add or modify:
   ```yaml
       env:
         MY_NEW_VAR: "my-value"
   ```
4. Push draft-deployment to `main`
5. Flux applies → pod restarts with new env var

**Current env vars** (in `face-ocr-api.yaml`):

| Variable | Value | Purpose |
|----------|-------|---------|
| `DATABASE_URL` | `postgresql://postgres:oRRtW4ga@postgres.cyber.svc.cluster.local:5432/face-ocr-db` | PostgreSQL |
| `RAY_ADDRESS` | `ray://rayservice-head-svc.ray.svc.cluster.local:10001` | Ray cluster |
| `RAY_NAMESPACE` | `face-ocr` | Ray namespace |
| `CHROMA_HOST` | `chroma-chromadb.default.svc.cluster.local` | ChromaDB |
| `CHROMA_PORT` | `8000` | ChromaDB port |
| `CHROMA_TOKEN` | `eYN1zg4hT86APjTPC8rfMpIgOfSQ33dM` | ChromaDB auth |
| `CHROMA_COLLECTION` | `face_identities` | ChromaDB collection |
| `DOSASHOP_BASE_URL` | `https://api.dosashop1.com/azblob` | Blob storage |
| `DOSASHOP_KEY` | `c7371a6b564f4e62a8b06e162a49eebd` | Blob storage key |
| `DOSASHOP_DEFAULT_CONTAINER` | `vidint` | Default container |
| `DOSASHOP_UPLOAD_CONTAINER` | `vidint` | Upload container |
| `DOSASHOP_SAS_TTL_SECONDS` | `1800` | SAS token TTL |
| `MAX_UPLOAD_SIZE_MB` | `500` | Upload limit |
| `JOB_TTL_SECONDS` | `3600` | Job cleanup |
| `REDIS_URL` | `redis://redis-master.cyber.svc.cluster.local:6379/0` | Redis |

---

## Version Constraints

All five must use the same Ray version. Currently `2.52.0`:

| Where | File | Value |
|-------|------|-------|
| HEAD node image | `ray-service.yaml` headGroupSpec | `rayproject/ray:2.52.0-py310` |
| CPU worker image | `ray-service.yaml` workerGroupSpecs[0] | `rayproject/ray:2.52.0-py310` |
| GPU worker image | `face-ocr-worker/Dockerfile` FROM | `rayproject/ray:2.52.0-py310-cu121` |
| initContainer image | `ray-service.yaml` initContainers[0] | `rayproject/ray:2.52.0-py310-cu121` |
| API Ray client | `face-ocr-api/requirements.txt` | `ray[client]==2.52.0` |

**Change one → change all five.** Mismatch = handshake failures.

---

## Verification

### Ray Cluster

```bash
kubectl get pods -n ray
kubectl get pods -n ray -l component=gpu-worker -o wide
kubectl logs <gpu-pod> -n ray -c install-heavy-deps --tail=20
kubectl logs <gpu-pod> -n ray -c ray-worker --tail=20

# GPU working?
kubectl exec <gpu-pod> -n ray -- python3 -c "
import sys; sys.path.insert(0, '/shared-packages')
import torch; print('CUDA:', torch.cuda.is_available())
import onnxruntime; print('Providers:', onnxruntime.get_available_providers())
"

# Models?
kubectl exec <gpu-pod> -n ray -- ls -lh /tmp/models/

# Resources?
kubectl exec <gpu-pod> -n ray -- python3 -c "import ray; ray.init(address='auto'); print(ray.cluster_resources())"
```

### face-ocr-api

```bash
kubectl get pods -n cyber -l app.kubernetes.io/name=face-ocr-api-microservice
kubectl logs -n cyber -l app.kubernetes.io/name=face-ocr-api-microservice --tail=20
kubectl port-forward svc/cyber-face-ocr-api-microservice 8080:80 -n cyber &
curl http://localhost:8080/api/v1/ping
curl http://localhost:8080/api/v1/ready
```

### Flux

```bash
kubectl get kustomization -n flux-system draft-deployments -o jsonpath='{.status.lastAppliedRevision}'
kubectl get imagepolicy -n flux-system cyber-face-ocr-api-helm-image-policy
flux reconcile kustomization draft-deployments -n flux-system  # force
```

---

## Troubleshooting

| Problem | Check | Fix |
|---------|-------|-----|
| GPU pod stuck in Init | `kubectl logs <pod> -n ray -c install-heavy-deps` | Pip timeout / disk space / network |
| ONNX on CPU not GPU | `LD_LIBRARY_PATH` env var on GPU worker | Must include `/shared-packages/nvidia/cudnn/lib` |
| API can't reach Ray | `kubectl get svc -n ray \| grep head` | HEAD svc must exist, `ray-client-server-port: '10001'` in rayStartParams |
| Pipeline not triggering | Check Azure DevOps → Pipelines | Push must be to `main` or `dev` |
| Flux not updating | `flux get kustomizations -n flux-system` | Force: `flux reconcile kustomization draft-deployments -n flux-system` |
| New pod has old image | Pipeline still building | Wait for pipeline, then restart pod |
| Ray version mismatch | Compare all 5 locations | Must all be `2.52.0` |

---

## Appendix A: Dockerfiles and Requirements

### face-ocr-worker/Dockerfile

```dockerfile
FROM rayproject/ray:2.52.0-py310-cu121

USER root
COPY certs/proxy-hub-ca-1.crt /usr/local/share/ca-certificates/proxy-hub-ca.crt
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgl1-mesa-glx libglib2.0-0 \
    && update-ca-certificates && rm -rf /var/lib/apt/lists/*
USER ray
WORKDIR /app

# Light deps ONLY — heavy deps via initContainer
RUN pip install --no-cache-dir \
    numpy>=1.24.0 scikit-learn==1.3.2 Pillow>=9.0.0 \
    huggingface-hub==0.19.4 rapidfuzz>=3.0.0 imagehash>=4.3.1 \
    aiohttp==3.9.1 fastapi==0.104.1 uvicorn==0.24.0 \
    python-multipart==0.0.6 pydantic==2.5.0

COPY ray_tasks.py .
ENV PYTHONPATH="/app" MODEL_CACHE_DIR="/tmp/models" OMP_NUM_THREADS="1"
EXPOSE 8000
```

### face-ocr-api/Dockerfile

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

ENV APP_NAME="face-ocr-api" ENVIRONMENT="production" APP_PORT=8000
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### face-ocr-api/requirements.txt

```
pydantic
pydantic-settings
fastapi
uvicorn
python-multipart
ray[client]==2.52.0
httpx
asyncpg
redis[hiredis]>=5.0.0
chromadb>=0.6.0,<0.7.0
prometheus-fastapi-instrumentator
numpy<2
```

---

## Appendix B: face-ocr-api.yaml (Full HelmRelease)

This is `draft-deployment/dev/face-ocr-api.yaml`. The `# {"$imagepolicy": ...}` comments are Flux markers — never remove them:

```yaml
apiVersion: helm.toolkit.fluxcd.io/v2
kind: HelmRelease
metadata:
  name: face-ocr-api
  namespace: default
spec:
  targetNamespace: cyber
  interval: 1m
  chart:
    spec:
      chart: microservice
      reconcileStrategy: ChartVersion
      sourceRef:
        kind: HelmRepository
        name: s2t-acr-helm-repo
  values:
    vpa:
      enabled: true
      updateMode: "Auto"
    image:
      repository: cybersmartstg.azurecr.io/data-ai/face-ocr-api # {"$imagepolicy": "flux-system:cyber-face-ocr-api-helm-image-policy:name"}
      tag: 1.0.0-alpha.1770916300 # {"$imagepolicy": "flux-system:cyber-face-ocr-api-helm-image-policy:tag"}
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
      DATABASE_URL: postgresql://postgres:oRRtW4ga@postgres.cyber.svc.cluster.local:5432/face-ocr-db
      RAY_ADDRESS: "ray://rayservice-head-svc.ray.svc.cluster.local:10001"
      RAY_NAMESPACE: face-ocr
      CHROMA_HOST: chroma-chromadb.default.svc.cluster.local
      CHROMA_PORT: "8000"
      CHROMA_TOKEN: "eYN1zg4hT86APjTPC8rfMpIgOfSQ33dM"
      CHROMA_COLLECTION: face_identities
      DOSASHOP_BASE_URL: https://api.dosashop1.com/azblob
      DOSASHOP_KEY: "c7371a6b564f4e62a8b06e162a49eebd"
      DOSASHOP_DEFAULT_CONTAINER: vidint
      DOSASHOP_UPLOAD_CONTAINER: vidint
      DOSASHOP_SAS_TTL_SECONDS: "1800"
      MAX_UPLOAD_SIZE_MB: "500"
      JOB_TTL_SECONDS: "3600"
      REDIS_URL: "redis://redis-master.cyber.svc.cluster.local:6379/0"
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
        path: /api/v1/ping
        port: 8000
      initialDelaySeconds: 15
      periodSeconds: 30
    readinessProbe:
      httpGet:
        path: /api/v1/ready
        port: 8000
      initialDelaySeconds: 10
      periodSeconds: 10
```

---

## Appendix C: Pipeline Template

Both repos use this identical pattern (replace `<REPO-NAME>`):

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
      - registryUrl: 'cybersmartstg.azurecr.io/data-ai/<REPO-NAME>'
        file: '$(Build.SourcesDirectory)/Dockerfile'
        displayName: 'Build Main Docker Image'
        context: '$(Build.SourcesDirectory)/'
        jobName: job1
    DevRegistries:
      - registryUrl: 'cybersmartstg.azurecr.io/data-ai/<REPO-NAME>'
    UATRegistries:
      - registryUrl: 'cybersmartstg.azurecr.io/data-ai/<REPO-NAME>'
```

`<REPO-NAME>` = `face-ocr-worker` or `face-ocr-api`.

---

## Appendix D: GPU Worker Section in ray-service.yaml

This is the GPU worker group in `draft-deployment/dev/ray-service.yaml`. It comes after the CPU worker group (`groupName: default-worker-group`):

```yaml
      # GPU worker group
      - replicas: 1
        minReplicas: 1
        maxReplicas: 3
        groupName: gpu-worker-group
        rayStartParams:
          block: 'true'
          num-cpus: "8"
          num-gpus: "1"
        template:
          metadata:
            labels:
              app: rayservice-sample
              component: gpu-worker
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
              - key: "nvidia.com/gpu"
                operator: "Exists"
                effect: "PreferNoSchedule"
              - key: "kubernetes.azure.com/scalesetpriority"
                operator: "Equal"
                value: "spot"
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
                    echo "=== Installing heavy ML dependencies ==="

                    # [1/5] PyTorch
                    pip install --target=/shared-packages --no-cache-dir \
                      torch torchvision --index-url https://download.pytorch.org/whl/cu121

                    # [2/5] PaddlePaddle + PaddleOCR (pin <3.0.0)
                    pip install --target=/shared-packages --no-cache-dir \
                      paddlepaddle-gpu>=2.5.0 "paddleocr>=2.7.0,<3.0.0"

                    # [3/5] EasyOCR
                    pip install --target=/shared-packages --no-cache-dir easyocr>=1.7.0

                    # [4/5] ONNX Runtime GPU + cuDNN 9
                    pip install --target=/shared-packages --no-cache-dir \
                      onnxruntime-gpu nvidia-cudnn-cu12 "numpy<2"

                    # [5/5] OpenCV
                    pip install --target=/shared-packages --no-cache-dir \
                      opencv-python-headless==4.8.1.78

                    # [6/6] Fix numpy (OpenCV installs 2.x, ONNX needs <2)
                    pip install --target=/shared-packages --no-cache-dir --upgrade "numpy<2"

                    # Remove Ray-conflicting packages
                    rm -rf /shared-packages/protobuf* /shared-packages/google/protobuf* /shared-packages/google-*.dist-info 2>/dev/null || true
                    rm -rf /shared-packages/click* /shared-packages/click-*.dist-info 2>/dev/null || true

                    # Verify
                    export PYTHONPATH=/shared-packages
                    python -c "import torch; print('PyTorch:', torch.__version__)"
                    python -c "import paddle; print('PaddlePaddle:', paddle.__version__)"
                    python -c "import cv2; print('OpenCV:', cv2.__version__)" || echo "cv2 skipped"
                    python -c "import onnxruntime; print('ONNX Runtime:', onnxruntime.__version__)"

                    # Download models to PVC (skip if already present)
                    mkdir -p /tmp/models
                    if [ -f /tmp/models/scrfd_10g.onnx ] && [ -f /tmp/models/w600k_r50.onnx ]; then
                      echo "Models already in PVC"
                    else
                      wget -q -O /tmp/models/scrfd_10g.onnx https://huggingface.co/immich-app/buffalo_l/resolve/main/detection/model.onnx
                      wget -q -O /tmp/models/w600k_r50.onnx https://huggingface.co/immich-app/buffalo_l/resolve/main/recognition/model.onnx
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
                image: cybersmartstg.azurecr.io/data-ai/face-ocr-worker:dev
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
                  - name: RAY_memory_monitor_refresh_ms
                    value: "0"
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
                  claimName: face-ocr-models
              - name: python-packages
                emptyDir:
                  sizeLimit: 10Gi
```
