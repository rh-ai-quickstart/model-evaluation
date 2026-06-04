"""Tests for health check endpoints (/health/live, /health/ready)."""

from unittest.mock import patch


def test_liveness_returns_healthy(client):
    """Liveness probe should always return 200 when the process is running."""
    response = client.get("/health/live")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "api"
    assert "timestamp" in data


def test_readiness_includes_dependency_status(client):
    """Readiness probe should list all dependency statuses."""
    response = client.get("/health/ready")
    data = response.json()

    assert "dependencies" in data
    assert "database" in data["dependencies"]
    assert "model_a" in data["dependencies"]
    assert "model_b" in data["dependencies"]


def test_readiness_degraded_when_db_unhealthy(client):
    """When the database is unreachable, readiness should report degraded (HTTP 200)."""
    with patch("src.routes.health._check_database", return_value="unhealthy"):
        response = client.get("/health/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["dependencies"]["database"] == "unhealthy"
    assert data["message"] is not None


def test_readiness_ready_when_all_healthy(client):
    """When all dependencies report healthy, readiness should return ready (200)."""
    with patch("src.routes.health._check_database", return_value="healthy"):
        response = client.get("/health/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["message"] is None


def test_liveness_does_not_check_dependencies(client):
    """Liveness probe must NOT call the database -- it only checks the process."""
    with patch("src.routes.health._check_database") as mock_db:
        response = client.get("/health/live")
        assert response.status_code == 200
        mock_db.assert_not_called()


def test_root_endpoint(client):
    """Root endpoint returns welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
