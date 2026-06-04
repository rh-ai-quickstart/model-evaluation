"""Health check endpoints per REQ-5-006.

/health/live  -- lightweight liveness probe (process is responsive)
/health/ready -- readiness probe with dependency checks and degraded status
"""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from starlette.responses import JSONResponse

from ..core.config import settings
from ..schemas.health import LivenessResponse, ReadinessResponse

try:
    from db import DatabaseService, get_db_service  # type: ignore[import-untyped]
except Exception:
    DatabaseService = None  # type: ignore[assignment]
    get_db_service = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
router = APIRouter()


async def _check_database(db_service: object | None) -> str:
    """Return 'healthy' or 'unhealthy' for the database dependency."""
    if db_service is None:
        return "unhealthy"
    try:
        result = await db_service.health_check()  # type: ignore[union-attr]
        return "healthy" if result.get("status") == "healthy" else "unhealthy"
    except Exception:
        logger.warning("Database health check failed", exc_info=True)
        return "unhealthy"


@router.get("/live", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    """Liveness probe -- only checks that the API process is responsive."""
    return LivenessResponse(
        status="healthy",
        service="api",
        timestamp=datetime.now(UTC).isoformat(),
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness(
    db_service: DatabaseService | None = (  # type: ignore[valid-type]
        Depends(get_db_service) if get_db_service else None
    ),
) -> JSONResponse:
    """Readiness probe -- checks all critical dependencies.

    Returns 200 with status ``ready`` or ``degraded``, or 503 with ``not_ready``
    when all critical dependencies (database + both models) are down.
    """
    deps: dict[str, str] = {}

    # Database check
    deps["database"] = await _check_database(db_service)

    # Model config check -- reports whether models are configured (not whether
    # the remote endpoint is reachable, which would add latency to the probe).
    deps["model_a"] = "healthy" if settings.MODEL_A_NAME else "unhealthy"
    deps["model_b"] = "healthy" if settings.MODEL_B_NAME else "unhealthy"

    unhealthy = [k for k, v in deps.items() if v == "unhealthy"]

    critical_deps = {"database", "model_a", "model_b"}
    critical_unhealthy = critical_deps & set(unhealthy)

    if not unhealthy:
        status = "ready"
        message = None
        http_status = 200
    elif critical_unhealthy == critical_deps:
        status = "not_ready"
        message = "All critical services are unavailable."
        http_status = 503
    else:
        status = "degraded"
        message = "Some services are unavailable. Some features may not work."
        http_status = 200

    body = ReadinessResponse(
        status=status,
        service="api",
        timestamp=datetime.now(UTC).isoformat(),
        dependencies=deps,
        message=message,
    )
    return JSONResponse(status_code=http_status, content=body.model_dump())
