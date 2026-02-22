# Enhance-NLP v2 — Ray Serve Deployment Guide

## Overview

Deploy the Enhance-NLP v2 pipeline (100-language NLP) as a Ray Serve application on the existing Kubernetes Ray cluster. Uses the Docker image + initContainer pattern for heavy ML dependencies.

### Architecture

```
Pod: nlp-worker
│
├── initContainer: install-heavy-deps (~3-5 min)
│   └── pip install torch (CPU) + transformers + sentencepiece → /shared-packages
│
├── Main container: ray-worker
│   ├── Docker image: enhance-nlp-v2 (light deps + pipeline code)
│   ├── PYTHONPATH=/shared-packages:/app
│   └── Joins Ray cluster → Ray Serve routes /enhance-nlp here
│
└── Volumes:
    ├── /shared-packages  (emptyDir)  ← heavy deps from initContainer
    └── /tmp/hf-cache     (PVC)       ← HuggingFace model cache
```

### Request Flow

```
Client → Ray Serve HTTP (:8000/enhance-nlp)
    → EnhanceNLP replica on nlp-worker pod
    → pycld2 (LID) → OPUS-MT (translation) → RoBERTa (sentiment)
      → REBEL (relations) → BART-CNN (summary) → BART-MNLI (classify)
      → LOO + cross-attention (attribution)
    → 15-field ModelResponse JSON
```

### Resource Requirements

| Resource | Value |
|----------|-------|
| CPU | 4 cores (2 for Ray actor, 2 headroom) |
| Memory | 10 Gi limit, 6 Gi request |
| Model storage (PVC) | 10 Gi |
| initContainer deps | ~2 Gi (torch CPU + transformers) |
| Processing time per request | 15-30s (CPU) |

---

## Prerequisites

- Access to Azure DevOps `data-ai` project
- Access to `draft-deployment` repo (`Cybersmart-Next` project)
- `kubectl` access to the K8s cluster
- Existing Ray cluster running (rayservice in `ray` namespace)

---

## Step 1: Code Repo (Azure DevOps)

The code is already pushed to:

```
https://dev.azure.com/predictintel/data-ai/_git/Enhance-NLP-v2
```

Branches: `main` and `dev`

### Repo Structure

```
Enhance-NLP-v2/
├── Dockerfile              ← rayproject/ray:2.52.0-py310 + light deps + code
├── azure-pipelines.yaml    ← CI/CD pipeline definition
├── requirements.txt        ← Light deps only (pycld2, scipy, scikit-learn)
├── serve_app.py            ← Ray Serve entry point (@serve.deployment)
├── pipeline.py             ← Core NLP engine (LID, translation, sentiment, relations, attribution)
├── processor.py            ← LocalNLPProcessor — wraps pipeline, returns 15-field dict
├── opus_models.py          ← OPUS-MT model map (22 dedicated + multilingual fallback)
├── ray-service-additions.yaml  ← YAML snippets for ray-service.yaml
└── certs/
    └── proxy-hub-ca-1.crt  ← Enterprise proxy CA cert
```

### Key Files

**serve_app.py** — Ray Serve entry point:
- Class `EnhanceNLP` decorated with `@serve.deployment`
- `__call__` method accepts POST with `{"text": "..."}`, returns 15-field JSON
- Starts with `sys.path.insert(0, "/shared-packages")` for initContainer deps
- Bind variable: `app = EnhanceNLP.bind()`

**Dockerfile** — Light image:
- Base: `rayproject/ray:2.52.0-py310`
- Installs: build-essential (for pycld2 C compilation), light pip deps
- Does NOT include torch or transformers (those go in initContainer)
- Sets `PYTHONPATH="/shared-packages:/app"`

**requirements.txt** — Light deps only:
```
pycld2==0.42
scipy>=1.12.0
numpy<2
vaderSentiment==3.3.2
scikit-learn>=1.4.0
```

---

## Step 2: Set Up Azure Pipeline

1. Go to https://dev.azure.com/predictintel/data-ai/_build
2. **New Pipeline** → **Azure Repos Git** → select **Enhance-NLP-v2**
3. **Existing Azure Pipelines YAML file** → path: `/azure-pipelines.yaml`
4. Click **Run**

The pipeline:
- Triggers on push to `main` or `dev`
- Uses shared template from `Cybersmart-Next/template`
- Builds Docker image from our Dockerfile
- Pushes to ACR: `cybersmartstg.azurecr.io/data-ai/enhance-nlp-v2:dev`
- Takes ~5 minutes

### Verify Pipeline Succeeded

Check Azure DevOps → Pipelines → the build should be green. The image is now in ACR.

---

## Step 3: Update ray-service.yaml

Clone the deployment repo:

```bash
git clone https://dev.azure.com/predictintel/Cybersmart-Next/_git/draft-deployment
cd draft-deployment
```

Edit `dev/ray-service.yaml`. Three changes needed:

### Change 1: Add PVC for HuggingFace Model Cache

Insert as a new YAML document BEFORE the `RayService` block (after the existing `face-ocr-models` PVC):

```yaml
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: enhance-nlp-models
  namespace: ray
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: azurefile
  resources:
    requests:
      storage: 10Gi
```

### Change 2: Add App to serveConfigV2

Under `serveConfigV2: |` → `applications:`, add after the existing apps (same indentation as `fruit_app`, `math_app`):

```yaml
      - name: enhance_nlp_app
        import_path: serve_app.app
        route_prefix: /enhance-nlp
        runtime_env:
          working_dir: /app
          env_vars:
            HF_HOME: /tmp/hf-cache
            TRANSFORMERS_CACHE: /tmp/hf-cache
        deployments:
          - name: EnhanceNLP
            num_replicas: 1
            ray_actor_options:
              num_cpus: 2
              memory: 8589934592
```

### Change 3: Add NLP Worker Group

Under `workerGroupSpecs:`, after `default-worker-group` and BEFORE `gpu-worker-group`, add:

```yaml
      # NLP Worker Group (CPU, heavy ML deps via initContainer)
      - replicas: 1
        minReplicas: 1
        maxReplicas: 2
        groupName: nlp-worker-group
        rayStartParams:
          block: 'true'
          num-cpus: "4"
        template:
          metadata:
            labels:
              app: rayservice-sample
              component: nlp-worker
          spec:
            imagePullSecrets:
              - name: regcred
            tolerations:
              - key: "kubernetes.azure.com/scalesetpriority"
                operator: "Equal"
                value: "spot"
                effect: "NoSchedule"
            initContainers:
              - name: install-heavy-deps
                image: rayproject/ray:2.52.0-py310
                command: ["/bin/bash", "-c"]
                args:
                  - |
                    set -e
                    echo "=== Installing heavy ML dependencies for Enhance-NLP v2 ==="

                    echo "[1/3] Installing PyTorch (CPU)..."
                    pip install --target=/shared-packages --no-cache-dir \
                      torch --index-url https://download.pytorch.org/whl/cpu

                    echo "[2/3] Installing Transformers + SentencePiece..."
                    pip install --target=/shared-packages --no-cache-dir \
                      transformers sentencepiece

                    echo "[3/3] Fixing numpy version..."
                    pip install --target=/shared-packages --no-cache-dir --upgrade "numpy<2"

                    echo "Removing conflicting packages..."
                    rm -rf /shared-packages/protobuf* /shared-packages/google/protobuf* 2>/dev/null || true
                    rm -rf /shared-packages/click* 2>/dev/null || true

                    echo "=== Verifying installations ==="
                    export PYTHONPATH=/shared-packages
                    python -c "import torch; print('PyTorch:', torch.__version__)"
                    python -c "import transformers; print('Transformers:', transformers.__version__)"
                    python -c "import sentencepiece; print('SentencePiece:', sentencepiece.__version__)"

                    echo "=== Heavy dependencies installed successfully ==="
                volumeMounts:
                  - mountPath: /shared-packages
                    name: python-packages
                resources:
                  limits:
                    cpu: "2"
                    memory: "4Gi"
                  requests:
                    cpu: "1"
                    memory: "2Gi"
            containers:
              - name: ray-worker
                image: cybersmartstg.azurecr.io/data-ai/enhance-nlp-v2:dev
                imagePullPolicy: Always
                resources:
                  limits:
                    cpu: "4"
                    memory: "10Gi"
                  requests:
                    cpu: "2"
                    memory: "6Gi"
                volumeMounts:
                  - mountPath: /tmp/ray
                    name: log-volume
                  - mountPath: /shared-packages
                    name: python-packages
                  - mountPath: /tmp/hf-cache
                    name: model-cache
                env:
                  - name: RAY_memory_monitor_refresh_ms
                    value: "0"
                  - name: PYTHONPATH
                    value: "/shared-packages:/app"
                  - name: HF_HOME
                    value: "/tmp/hf-cache"
                  - name: TRANSFORMERS_CACHE
                    value: "/tmp/hf-cache"
                  - name: OMP_NUM_THREADS
                    value: "1"
            volumes:
              - name: log-volume
                emptyDir: {}
              - name: python-packages
                emptyDir:
                  sizeLimit: 5Gi
              - name: model-cache
                persistentVolumeClaim:
                  claimName: enhance-nlp-models
```

### Push

```bash
git add dev/ray-service.yaml
git commit -m "Deploy enhance-nlp-v2 on Ray Serve"
git push origin main
```

Flux auto-applies within ~2 minutes.

---

## Step 4: Verify Deployment

### Check Flux Applied

```bash
kubectl get kustomization -n flux-system draft-deployments \
  -o jsonpath='{.status.lastAppliedRevision}'
```

### Check PVC Created

```bash
kubectl get pvc -n ray enhance-nlp-models
```

### Watch Pod Start

```bash
kubectl get pods -n ray -l component=nlp-worker -w
```

### Check initContainer Logs

```bash
kubectl logs -n ray -l component=nlp-worker -c install-heavy-deps --tail=30
```

### Check Main Container Logs

```bash
kubectl logs -n ray -l component=nlp-worker -c ray-worker --tail=30
```

### Check Ray Serve App Status

```bash
kubectl port-forward svc/rayservice-head-svc 8265:8265 -n ray &
curl -s http://localhost:8265/api/serve/applications/ | python3 -m json.tool
```

App should show status: `RUNNING`.

### Test the Endpoint

```bash
kubectl port-forward svc/rayservice-serve-svc 8000:8000 -n ray &

# Tswana (positive sentiment)
curl -X POST http://localhost:8000/enhance-nlp \
  -H "Content-Type: application/json" \
  -d '{"text": "Ke a leboga Modimo wa me ka dilo tsotlhe tse di molemo"}'

# Hindi
curl -X POST http://localhost:8000/enhance-nlp \
  -H "Content-Type: application/json" \
  -d '{"text": "भारतीय क्रिकेट टीम ने शानदार जीत हासिल की"}'

# English
curl -X POST http://localhost:8000/enhance-nlp \
  -H "Content-Type: application/json" \
  -d '{"text": "The new renewable energy policy has been a disaster for local communities."}'
```

### Expected Response Format

```json
{
  "elapsed": 18.5,
  "lang": "tn",
  "summary": "...",
  "sentiment": 0.628,
  "sentimentPhrases": [{"phrase": "leboga", "score": 0.45}],
  "themes": [{"theme": "modimo", "score": 0.8, "sentiment": 0.0}],
  "queryDefinedTopics": [...],
  "conceptDefinedTopics": [...],
  "namedEntities": [{"name": "Modimo", "type": "Entity", "sentiment": 0.0}],
  "namedEntityRelationships": [...],
  "classifications": [{"topic": "Environment", "score": 0.65, "sentiment": 0.0}],
  "categories": [...],
  "translatedText": "I thank my God for all the good things",
  "sarcasm_score": null,
  "sarcastic_phrases": []
}
```

---

## Connection Chain

Every field must link correctly:

```
Python:   app = EnhanceNLP.bind()              ← bind variable
              ↓
YAML:     import_path: serve_app.app           ← <filename>.<variable>
              ↓
YAML:     working_dir: /app                    ← where Ray looks for the module
              ↓
Docker:   WORKDIR /app + COPY serve_app.py .   ← code is at /app/serve_app.py
              ↓
YAML:     deployments[].name: EnhanceNLP       ← must match Python class name
              ↓
initCont: /shared-packages                     ← torch, transformers installed here
              ↓
env:      PYTHONPATH=/shared-packages:/app      ← Python finds both heavy deps + code
              ↓
PVC:      /tmp/hf-cache                        ← HuggingFace models cached here
              ↓
env:      HF_HOME=/tmp/hf-cache                ← transformers knows where to cache
```

---

## Deployment Timeline

| Time | Event |
|------|-------|
| 0 min | Push ray-service.yaml to draft-deployment |
| ~2 min | Flux detects and applies to K8s |
| ~2 min | KubeRay creates nlp-worker pod |
| ~5 min | initContainer installs torch + transformers |
| ~6 min | Main container starts, joins Ray cluster |
| ~7 min | Ray Serve deploys EnhanceNLP app |
| ~10 min | First request downloads models to PVC (~5.5 GB) |
| ~12 min | **READY** — subsequent requests: 15-30s each |

After first deployment, restarts are faster (models persist on PVC).

---

## Day-to-Day Operations

### Update Pipeline Code

```bash
# Edit pipeline.py, processor.py, etc. in Enhance-NLP-v2 repo
cd Enhance-NLP-v2
# make changes...
git add -A && git commit -m "Update pipeline"
git push origin main && git push origin dev
# Wait for Azure Pipeline (~5 min)
# Restart the pod to pick up new image:
kubectl delete pod -n ray -l component=nlp-worker
```

### Update Heavy Dependencies

Edit the initContainer `args` in `ray-service.yaml`:
```bash
cd draft-deployment
# edit dev/ray-service.yaml initContainer section
git add dev/ray-service.yaml && git commit -m "Update NLP deps" && git push origin main
# Flux applies → pod restarts → initContainer re-installs deps
```

### Scale Up

Edit `replicas` / `maxReplicas` in the nlp-worker-group:
```bash
cd draft-deployment
# change replicas: 2, maxReplicas: 3
git add dev/ray-service.yaml && git commit -m "Scale NLP workers" && git push origin main
```

### Remove the App

Remove from `ray-service.yaml`:
1. The `enhance_nlp_app` entry from `serveConfigV2`
2. The entire `nlp-worker-group` from `workerGroupSpecs`
3. The `enhance-nlp-models` PVC

Push to `main`.

---

## Troubleshooting

| Problem | Check | Fix |
|---------|-------|-----|
| Pod stuck in Init | `kubectl logs -c install-heavy-deps` | Pip timeout, network, disk space |
| App stuck in DEPLOYING | `kubectl logs -c ray-worker --tail=50` | Wrong import_path, working_dir mismatch |
| ModuleNotFoundError: torch | PYTHONPATH env var | Must be `/shared-packages:/app` |
| ModuleNotFoundError: pycld2 | Docker image build | Check pipeline built successfully |
| Models re-downloading on restart | PVC mount | Check `enhance-nlp-models` PVC is bound |
| OOM killed | Memory limits | Increase to 12Gi or 16Gi |
| Slow first request | Model download | Normal (~5 min first time), cached after |
| Ray version mismatch | All 3 locations | initContainer, Docker base, ray-service.yaml must all use 2.52.0 |
| protobuf/click conflict | initContainer cleanup | Ensure `rm -rf` lines are present |

### Useful Commands

```bash
# Pod status
kubectl get pods -n ray -l component=nlp-worker

# Full pod describe (events, scheduling)
kubectl describe pod -n ray -l component=nlp-worker

# initContainer logs
kubectl logs -n ray -l component=nlp-worker -c install-heavy-deps

# Main container logs
kubectl logs -n ray -l component=nlp-worker -c ray-worker --tail=50

# Ray dashboard
kubectl port-forward svc/rayservice-head-svc 8265:8265 -n ray &
open http://localhost:8265

# Ray Serve app status
curl -s http://localhost:8265/api/serve/applications/ | python3 -m json.tool

# Test endpoint
kubectl port-forward svc/rayservice-serve-svc 8000:8000 -n ray &
curl -X POST http://localhost:8000/enhance-nlp \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'

# Force Flux reconcile
flux reconcile kustomization draft-deployments -n flux-system

# Restart pod (pick up new image)
kubectl delete pod -n ray -l component=nlp-worker
```

---

## Version Constraints

All Ray components must match version `2.52.0`:

| Component | Location |
|-----------|----------|
| HEAD node image | `ray-service.yaml`: `rayproject/ray:2.52.0-py310` |
| CPU worker image | `ray-service.yaml`: `rayproject/ray:2.52.0-py310` |
| NLP worker base image | `Dockerfile`: `FROM rayproject/ray:2.52.0-py310` |
| NLP initContainer image | `ray-service.yaml`: `rayproject/ray:2.52.0-py310` |
| GPU worker base image | `Dockerfile`: `rayproject/ray:2.52.0-py310-cu121` |

**Change one → change all.**
