"""Model serving endpoints -- list, status, and switch active model."""

import logging

from fastapi import APIRouter, HTTPException

from ..core.config import settings
from ..schemas.models import (
    ModelMetadata,
    ModelMetadataResponse,
    ModelResponse,
    ModelStatusResponse,
)
from ..services.litellm_metadata import fetch_model_metadata

logger = logging.getLogger(__name__)
router = APIRouter()


def _build_models() -> list[dict]:
    """Build model list from environment configuration.

    Models are configured via environment variables (MODEL_A_NAME, MODEL_B_NAME).
    The ModelConfig DB table exists for future extensibility but is not currently used by this endpoint.
    In MaaS mode, models are configured via environment variables (set in
    values.yaml).  This avoids requiring a database connection just to list
    available models, which keeps the QuickStart simple.
    """
    # IDs are hardcoded for the two-model QuickStart scope. Derive from DB if expanding to N models.
    return [
        {
            "id": 1,
            "name": settings.MODEL_A_NAME,
            "endpoint_url": f"{settings.MAAS_ENDPOINT}/v1",
            "deployment_mode": settings.MODEL_A_DEPLOYMENT_MODE,
            "is_active": True,
        },
        {
            "id": 2,
            "name": settings.MODEL_B_NAME,
            "endpoint_url": f"{settings.MAAS_ENDPOINT}/v1",
            "deployment_mode": settings.MODEL_B_DEPLOYMENT_MODE,
            "is_active": True,
        },
    ]


@router.get("/metadata", response_model=ModelMetadataResponse)
async def list_model_metadata() -> ModelMetadataResponse:
    """Fetch rich model metadata from the LiteLLM admin API.

    Returns cached metadata (5-min TTL). Returns available=false
    when LITELLM_ADMIN_URL is not configured.
    """
    if not settings.LITELLM_ADMIN_URL:
        return ModelMetadataResponse(models=[], available=False)

    raw = await fetch_model_metadata()
    models = [ModelMetadata(**m) for m in raw]
    return ModelMetadataResponse(models=models, available=True)


@router.get("/", response_model=list[ModelResponse])
async def list_models() -> list[dict]:
    """List all configured models for A/B comparison."""
    return _build_models()


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: int) -> dict:
    """Get a single model configuration by ID."""
    models = _build_models()
    for model in models:
        if model["id"] == model_id:
            return model
    raise HTTPException(status_code=404, detail=f"Model with id {model_id} not found")


@router.get("/{model_id}/status", response_model=ModelStatusResponse)
async def get_model_status(model_id: int) -> ModelStatusResponse:
    """Check health/availability of a specific model endpoint.

    In a full deployment this would call the vLLM /health endpoint.
    For now it returns the model config with an 'available' status,
    which will be replaced with a real health check when models are
    deployed.
    """
    models = _build_models()
    for model in models:
        if model["id"] == model_id:
            return ModelStatusResponse(
                name=model["name"],
                status="available",
                deployment_mode=model["deployment_mode"],
                endpoint_url=model["endpoint_url"],
            )
    raise HTTPException(status_code=404, detail=f"Model with id {model_id} not found")
