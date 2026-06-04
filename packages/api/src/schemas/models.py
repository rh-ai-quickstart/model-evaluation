"""Pydantic schemas for model serving endpoints."""

from pydantic import BaseModel, Field


class ModelResponse(BaseModel):
    """Response schema for a model configuration."""

    id: int
    name: str
    endpoint_url: str
    deployment_mode: str
    is_active: bool

    model_config = {"from_attributes": True}


class ModelStatusResponse(BaseModel):
    """Response for model health/status check."""

    name: str
    status: str
    deployment_mode: str
    endpoint_url: str


class ModelPricing(BaseModel):
    """Pricing info from LiteLLM catalog."""

    input: float | None = None
    output: float | None = None
    unit: str | None = None


class ModelMetadata(BaseModel):
    """Rich model metadata from LiteLLM admin API."""

    id: str
    name: str
    context_length: int | None = None
    max_tokens: int | None = None
    pricing: ModelPricing | None = None
    capabilities: list[str] = Field(default_factory=list)
    tpm: int | None = None
    rpm: int | None = None
    supports_vision: bool | None = None
    supports_function_calling: bool | None = None
    supports_embeddings: bool | None = None


class ModelMetadataResponse(BaseModel):
    """Response wrapper for model metadata list."""

    models: list[ModelMetadata] = Field(default_factory=list)
    available: bool = True
