# ConfigMap Deployment Agent Guide

## What This Is

This guide lets you take any CPU-based Python code and deploy it as an HTTP API on a Kubernetes Ray cluster. The code is stored in a Kubernetes ConfigMap (a key-value store for small files), mounted as files into Ray pods, and served via Ray Serve (Ray's built-in HTTP serving framework). No Docker builds or CI/CD pipelines — one git push and Flux (a GitOps tool that watches the repo and auto-applies changes to the cluster) deploys it.

## When to Use This vs Docker

| Condition | ConfigMap (this guide) | Docker (agent-docker.md) |
|-----------|----------------------|--------------------------|
| CPU only | Yes | Yes |
| GPU/CUDA needed | No | Yes |
| pip dependencies | ≤500MB | Any size |
| Code size | ≤1MB (K8s limit) | Any size |
| System packages (apt) | No | Yes |
| Deploy speed | ~2 min (git push) | ~10 min (build + push + restart) |

---

## Skill: Deploy New Code

### Prerequisites

Clone the deployment repo:
```bash
git clone https://dev.azure.com/predictintel/Cybersmart-Next/_git/draft-deployment
cd draft-deployment
```

You will only edit one file: `dev/ray-service.yaml`. You will only push to one branch: `main`.

### 1. Convert Code to Ray Serve Format

Your code must be a Python file with:
- A class decorated with `@serve.deployment`
- An `async def __call__(self, request)` method that handles HTTP requests
- A module-level variable created by calling `.bind()` on the class

```python
# my_service.py
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.1}
)
class MyService:
    def __init__(self):
        # Runs once when replica starts. Load models, set up state, etc.
        pass

    async def __call__(self, request: Request) -> JSONResponse:
        body = await request.json()
        result = {"echo": body, "status": "processed"}
        return JSONResponse(result)

# This variable name ("app") is what you reference in the YAML config
app = MyService.bind()
```

See [Appendix B](#appendix-b-ray-serve-code-patterns) for more patterns (stateful, multi-deployment DAGs).

### 2. Create a ConfigMap in ray-service.yaml

Open `dev/ray-service.yaml`. This file has multiple YAML documents separated by `---`. Find the first `---` separator (after the PVC block) and insert a new ConfigMap document before the RayService:

```yaml
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-service-code
  namespace: ray
data:
  my_service.py: |
    from ray import serve
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @serve.deployment(
        num_replicas=1,
        ray_actor_options={"num_cpus": 0.1}
    )
    class MyService:
        def __init__(self):
            pass

        async def __call__(self, request: Request) -> JSONResponse:
            body = await request.json()
            return JSONResponse({"echo": body})

    app = MyService.bind()
---
```

Rules:
- `namespace` MUST be `ray`
- The `|` after the filename means multiline — indent all code lines consistently under it
- For multiple files, add more keys: `utils.py: |`, `helpers.py: |`, etc.
- ConfigMap name convention: `<app-name>-code`

### 3. Add App to serveConfigV2

Find `serveConfigV2: |` in the file. Under it is `applications:` with existing app entries (currently `fruit_app` at route `/fruit` and `math_app` at route `/calc`). Add your app as a new list item at the same indentation:

```yaml
      - name: my_service_app
        import_path: my_service.app
        route_prefix: /my-service
        runtime_env:
          working_dir: /tmp/my-service
          pip:
            - requests>=2.28.0
        deployments:
          - name: MyService
            num_replicas: 1
            ray_actor_options:
              num_cpus: 0.1
```

**Every field must connect correctly:**

| Field | Must match | Example |
|-------|-----------|---------|
| `import_path` | `<filename minus .py>.<bind variable>` | File is `my_service.py`, variable is `app` → `my_service.app` |
| `route_prefix` | Unique HTTP path, starts with `/` | `/my-service` |
| `working_dir` | Exactly the `mountPath` in steps 4 and 5 | `/tmp/my-service` |
| `deployments[].name` | Exactly the Python class name | `MyService` |
| `num_cpus` | 0.1-0.5 recommended (CPU worker has 2 total) | `0.1` |
| `pip` | Optional — packages Ray installs at deploy time | `- pandas>=1.5.0` |

### 4. Add Volume to HEAD Node

Find the HEAD node container by searching for `name: ray-head`. It has a `volumeMounts:` list and the pod has a `volumes:` list. Append to both.

**Add to `volumeMounts:` (under the `ray-head` container, after existing mounts like `log-volume` and `atom-kg-code`):**

```yaml
                - mountPath: /tmp/my-service
                  name: my-service-code
```

**Add to `volumes:` (under the HEAD `spec:`, after existing volumes):**

```yaml
            - name: my-service-code
              configMap:
                name: my-service-code
                items:
                  - key: my_service.py
                    path: my_service.py
```

See [Appendix A](#appendix-a-ray-serviceyaml-structure) for the exact structure and indentation.

### 5. Add Same Volume to CPU Worker

Find the CPU worker by searching for `groupName: default-worker-group`. Its `ray-worker` container also has `volumeMounts:` and the pod has `volumes:`. Add the **identical** entries:

**Add to `volumeMounts:`:**
```yaml
                  - mountPath: /tmp/my-service
                    name: my-service-code
```

**Add to `volumes:`:**
```yaml
              - name: my-service-code
                configMap:
                  name: my-service-code
                  items:
                    - key: my_service.py
                      path: my_service.py
```

Both HEAD and CPU worker MUST have the same mounts. Ray Serve can schedule replicas on either.

### 6. DO NOT TOUCH the GPU Worker

After the CPU worker group, there's a second worker group: `groupName: gpu-worker-group`. **Stop. Never edit anything in the GPU worker section.** It's managed via Docker images (see `agent-docker.md`).

### 7. Push

```bash
cd draft-deployment
git add dev/ray-service.yaml
git commit -m "Deploy my-service via ConfigMap"
git push origin main
```

Flux auto-applies within ~1-2 minutes.

### 8. Verify

```bash
# Flux applied?
kubectl get kustomization -n flux-system draft-deployments \
  -o jsonpath='{.status.lastAppliedRevision}'

# ConfigMap created?
kubectl get configmap -n ray my-service-code

# Pods running?
kubectl get pods -n ray

# App status (should be RUNNING)
kubectl port-forward svc/rayservice-head-svc 8265:8265 -n ray &
curl -s http://localhost:8265/api/serve/applications/ | python3 -m json.tool

# Test endpoint
kubectl port-forward svc/rayservice-serve-svc 8000:8000 -n ray &
curl -X POST http://localhost:8000/my-service \
  -H "Content-Type: application/json" \
  -d '{"test": "hello"}'
```

---

## Skill: Update Existing Code

1. Edit the Python code inside the ConfigMap `data:` in `dev/ray-service.yaml`
2. `git add dev/ray-service.yaml && git commit -m "Update my-service" && git push origin main`
3. Flux applies → K8s updates ConfigMap → Ray Serve reloads

If it doesn't reload: `kubectl delete pod -n ray -l component=head`

---

## Skill: Remove an App

Remove these 6 things from `dev/ray-service.yaml`:
1. The `- name: my_service_app` entry from `serveConfigV2`
2. The `volumeMount` from HEAD container
3. The `volume` from HEAD pod
4. The `volumeMount` from CPU worker container
5. The `volume` from CPU worker pod
6. The ConfigMap YAML document

Push to `main`.

---

## Troubleshooting

**App stuck in DEPLOYING:**
```bash
kubectl logs -n ray -l component=head -c ray-head --tail=100 | grep -i error
```
Causes: wrong `import_path`, `working_dir` ≠ `mountPath`, pip install failing.

**ModuleNotFoundError:**
`working_dir` ≠ `mountPath`, or filename has dashes (invalid Python module names — use underscores).

**Connection refused:**
```bash
kubectl port-forward svc/rayservice-serve-svc 8000:8000 -n ray &
curl http://localhost:8000/my-service  # Must match route_prefix
```

**Pods restarting:**
YAML syntax error (serveConfigV2 is YAML-inside-YAML — indentation must be exact), or ConfigMap doesn't exist in `ray` namespace.

---

## Appendix A: ray-service.yaml Structure

The file `dev/ray-service.yaml` in the `draft-deployment` repo has this structure. The `# <-- YOUR EDIT` comments show where you add things:

```yaml
# ─── Document 1: PVC (DO NOT TOUCH) ───────────────────────
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: face-ocr-models
  namespace: ray
spec:
  accessModes: [ReadWriteMany]
  storageClassName: azurefile
  resources:
    requests:
      storage: 1Gi
---
# ─── Document 2: YOUR CONFIGMAP (INSERT HERE) ─────────────
# apiVersion: v1
# kind: ConfigMap
# metadata:
#   name: my-service-code
#   namespace: ray
# data:
#   my_service.py: |
#     ...your code...
# ---
# ─── Document 3: RayService ───────────────────────────────
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: rayservice
  namespace: ray
spec:
  serviceUnhealthySecondThreshold: 1800
  deploymentUnhealthySecondThreshold: 1800

  serveConfigV2: |
    applications:
      - name: fruit_app
        import_path: fruit.deployment_graph
        route_prefix: /fruit
        runtime_env:
          working_dir: "https://github.com/ray-project/test_dag/archive/78b4a5da38796123d9f9ffff59bab2792a043e95.zip"
        deployments:
          - name: MangoStand
            num_replicas: 2
            max_replicas_per_node: 1
            user_config:
              price: 3
            ray_actor_options:
              num_cpus: 0.1
          - name: OrangeStand
            num_replicas: 1
            user_config:
              price: 2
            ray_actor_options:
              num_cpus: 0.1
          - name: PearStand
            num_replicas: 1
            user_config:
              price: 1
            ray_actor_options:
              num_cpus: 0.1
          - name: FruitMarket
            num_replicas: 1
            ray_actor_options:
              num_cpus: 0.1
      - name: math_app
        import_path: conditional_dag.serve_dag
        route_prefix: /calc
        runtime_env:
          working_dir: "https://github.com/ray-project/test_dag/archive/78b4a5da38796123d9f9ffff59bab2792a043e95.zip"
        deployments:
          - name: Adder
            num_replicas: 1
            user_config:
              increment: 3
            ray_actor_options:
              num_cpus: 0.1
          - name: Multiplier
            num_replicas: 1
            user_config:
              factor: 5
            ray_actor_options:
              num_cpus: 0.1
          - name: Router
            num_replicas: 1

      # <-- YOUR APP GOES HERE (same indentation as fruit_app/math_app)

  rayClusterConfig:
    rayVersion: '2.52.0'
    enableInTreeAutoscaling: false

    headGroupSpec:
      rayStartParams:
        dashboard-host: '0.0.0.0'
        dashboard-port: '8265'
        num-cpus: '0'
        ray-client-server-port: '10001'
      template:
        metadata:
          labels:
            app: rayservice-sample
            component: head
        spec:
          imagePullSecrets:
            - name: regcred
          tolerations:
            - key: "kubernetes.azure.com/scalesetpriority"
              operator: "Equal"
              value: "spot"
              effect: "NoSchedule"
          containers:
            - name: ray-head
              image: rayproject/ray:2.52.0-py310
              ports:
                - containerPort: 8000
                  name: serve
                - containerPort: 8265
                  name: dashboard
                - containerPort: 6379
                  name: gcs
                - containerPort: 10001
                  name: ray-client
              resources:
                limits:
                  cpu: "2"
                  memory: "8Gi"
                requests:
                  cpu: "1"
                  memory: "4Gi"
              volumeMounts:
                - mountPath: /tmp/ray
                  name: log-volume
                - mountPath: /tmp/atom-kg
                  name: atom-kg-code
                # <-- YOUR VOLUME MOUNT HERE
              env:
                - name: RAY_memory_monitor_refresh_ms
                  value: "0"
                - name: RAY_GRAFANA_HOST
                  value: "http://monitoring-prom-stack-grafana.monitoring.svc.cluster.local"
                - name: RAY_PROMETHEUS_HOST
                  value: "http://prom-stack-prometheus.monitoring.svc.cluster.local:9090"
                - name: RAY_GRAFANA_IFRAME_HOST
                  value: "http://monitoring-prom-stack-grafana.monitoring.svc.cluster.local"
                - name: RAY_PROMETHEUS_NAME
                  value: "prom-stack"
          volumes:
            - name: log-volume
              emptyDir: {}
            - name: atom-kg-code
              configMap:
                name: atom-kg-ray-tasks
                items:
                  - key: atom_tasks.py
                    path: atom_tasks.py
            # <-- YOUR VOLUME DEFINITION HERE

    workerGroupSpecs:
      - replicas: 1
        minReplicas: 1
        maxReplicas: 3
        groupName: default-worker-group
        rayStartParams:
          block: 'true'
        template:
          metadata:
            labels:
              app: rayservice-sample
              component: cpu-worker
          spec:
            imagePullSecrets:
              - name: regcred
            tolerations:
              - key: "kubernetes.azure.com/scalesetpriority"
                operator: "Equal"
                value: "spot"
                effect: "NoSchedule"
            containers:
              - name: ray-worker
                image: rayproject/ray:2.52.0-py310
                resources:
                  limits:
                    cpu: "2"
                    memory: "4Gi"
                  requests:
                    cpu: "1"
                    memory: "2Gi"
                volumeMounts:
                  - mountPath: /tmp/ray
                    name: log-volume
                  - mountPath: /tmp/atom-kg
                    name: atom-kg-code
                  # <-- YOUR VOLUME MOUNT HERE (same as HEAD)
            volumes:
              - name: log-volume
                emptyDir: {}
              - name: atom-kg-code
                configMap:
                  name: atom-kg-ray-tasks
                  items:
                    - key: atom_tasks.py
                      path: atom_tasks.py
              # <-- YOUR VOLUME DEFINITION HERE (same as HEAD)

      # ═══════════════════════════════════════════════════════
      # GPU WORKER GROUP BELOW — DO NOT TOUCH
      # Managed via Docker images (see agent-docker.md)
      # ═══════════════════════════════════════════════════════
      - replicas: 1
        groupName: gpu-worker-group
        # ... (GPU worker config, initContainer, etc.) ...
```

---

## Appendix B: Ray Serve Code Patterns

### Stateful (loads model/resource on init)

```python
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.5})
class Predictor:
    def __init__(self):
        import joblib
        self.model = joblib.load("/tmp/predictor/model.joblib")

    async def __call__(self, request: Request) -> JSONResponse:
        data = await request.json()
        prediction = self.model.predict([data["features"]])
        return JSONResponse({"prediction": prediction.tolist()})

app = Predictor.bind()
```

### Multi-deployment DAG

```python
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.1})
class Preprocessor:
    def clean(self, text: str) -> str:
        return text.lower().strip()

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.1})
class Analyzer:
    def analyze(self, text: str) -> dict:
        return {"length": len(text), "words": len(text.split())}

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.1})
class Pipeline:
    def __init__(self, preprocessor, analyzer):
        self.preprocessor = preprocessor
        self.analyzer = analyzer

    async def __call__(self, request: Request) -> JSONResponse:
        body = await request.json()
        cleaned = await self.preprocessor.clean.remote(body["text"])
        result = await self.analyzer.analyze.remote(cleaned)
        return JSONResponse(result)

pre = Preprocessor.bind()
ana = Analyzer.bind()
app = Pipeline.bind(pre, ana)
```

For DAGs, list every class in `deployments:`:
```yaml
        deployments:
          - name: Preprocessor
            num_replicas: 1
            ray_actor_options: { num_cpus: 0.1 }
          - name: Analyzer
            num_replicas: 1
            ray_actor_options: { num_cpus: 0.1 }
          - name: Pipeline
            num_replicas: 1
            ray_actor_options: { num_cpus: 0.1 }
```

---

## Appendix C: Connection Chain

Every field must link correctly. If any link breaks, the deployment fails:

```
Python: app = MyService.bind()        ← variable name
           ↓
YAML import_path: my_service.app      ← <filename_no_ext>.<variable>
           ↓
YAML working_dir: /tmp/my-service     ← Ray looks for module here
           ↓
K8s mountPath: /tmp/my-service        ← K8s mounts files here (MUST = working_dir)
           ↓
K8s volume name: my-service-code      ← connects mount to volume definition
           ↓
ConfigMap name: my-service-code       ← the K8s ConfigMap resource
           ↓
ConfigMap data key: my_service.py     ← the file stored in ConfigMap
           ↓
items[].path: my_service.py           ← filename inside mount directory
           ↓
YAML deployments[].name: MyService    ← MUST exactly match Python class name
```
