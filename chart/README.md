# OpenShift AI Model Evaluation Helm Chart

Deploys the full OpenShift AI Model Evaluation stack:

- FastAPI backend
- React frontend
- PostgreSQL + pgvector
- migration job
- OpenShift Routes

Chart metadata:

- chart name: `model-evaluation`
- chart path: `chart/`

## Prerequisites

- OpenShift 4.10+ (or compatible Kubernetes)
- Helm 3.8+
- `oc` CLI
- container images for API and UI available in your registry
- MaaS/LiteLLM API token if using remote model serving

## Quick Start

```bash
oc new-project model-evaluation || oc project model-evaluation

helm install model-eval ./chart \
  --namespace model-evaluation \
  --set secrets.API_TOKEN="<your-token>"
```

Check resources:

```bash
oc get pods -n model-evaluation
oc get routes -n model-evaluation
```

## Route Behavior

When `routes.enabled=true`, the chart creates:

- UI route to the UI service
- API route with path `/api` to the API service
- health route with path `/health` to the API service

If `routes.sharedHost` is set, routes share one host with path-based routing.

## Important Values (Current Defaults)

| Value | Default |
| --- | --- |
| `global.imageRegistry` | `quay.io` |
| `global.imageRepository` | `rh-ai-quickstart` |
| `global.imageTag` | `latest` |
| `models.maasEndpoint` | `https://litellm-litemaas.apps.prod.rhoai.rh-aiservices-bu.com` |
| `models.modelA.name` | `Granite-3.3-8B-Instruct` |
| `models.modelB.name` | `Llama-4-Scout-17B-16E-W4A16` |
| `models.embeddingModel` | `Nomic-embed-text-v2-moe` |
| `models.judgeModel` | `Mistral-Small-24B-W8A8` |
| `secrets.POSTGRES_DB` | `model-evaluation` |
| `secrets.POSTGRES_USER` | `user` |
| `secrets.POSTGRES_PASSWORD` | `changeme` |
| `secrets.API_TOKEN` | `""` (must be set for model calls) |
| `api.replicas` | `1` |
| `ui.replicas` | `1` |
| `database.persistence.size` | `10Gi` |

See [`values.yaml`](values.yaml) for full configuration.

## Deployment Modes

### MaaS / LiteLLM (default)

- keep `llm-service.enabled=false`
- set `models.*` names and `models.maasEndpoint`
- provide `secrets.API_TOKEN`

### Self-hosted serving

Enable subchart and set deployment mode:

```yaml
llm-service:
  enabled: true
models:
  modelA:
    deploymentMode: self-hosted
  modelB:
    deploymentMode: self-hosted
```

## Upgrade / Uninstall

```bash
helm upgrade model-eval ./chart -n model-evaluation --reuse-values
helm uninstall model-eval -n model-evaluation
```

Optional cleanup:

```bash
oc delete pvc -l app.kubernetes.io/instance=model-eval -n model-evaluation
```

## Troubleshooting

```bash
# API logs
oc logs -l app.kubernetes.io/component=api -n model-evaluation

# Migration job logs
oc logs job/model-evaluation-migration -n model-evaluation

# Route + service checks
oc get routes,svc -n model-evaluation
```

