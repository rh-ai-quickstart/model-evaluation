"""Health check response schemas per REQ-5-006."""

from pydantic import BaseModel


class LivenessResponse(BaseModel):
    """Response for /health/live -- process-level liveness only."""

    status: str
    service: str
    timestamp: str


class ReadinessResponse(BaseModel):
    """Response for /health/ready -- includes dependency checks.

    ``status`` is one of: ``ready``, ``degraded``, ``not_ready``.
    """

    status: str
    service: str
    timestamp: str
    dependencies: dict[str, str]
    message: str | None = None
