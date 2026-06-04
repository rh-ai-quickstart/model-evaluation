"""Tests for model serving endpoints (/models)."""

from src.core.config import settings


def test_list_models_returns_two(client):
    """Should return both Model A and Model B from config."""
    response = client.get("/models/")
    assert response.status_code == 200

    data = response.json()
    assert len(data) == 2
    names = {m["name"] for m in data}
    assert settings.MODEL_A_NAME in names
    assert settings.MODEL_B_NAME in names


def test_list_models_includes_required_fields(client):
    """Each model should have id, name, endpoint_url, deployment_mode, is_active."""
    response = client.get("/models/")
    data = response.json()

    for model in data:
        assert "id" in model
        assert "name" in model
        assert "endpoint_url" in model
        assert "deployment_mode" in model
        assert "is_active" in model


def test_get_model_by_id(client):
    """Should return a specific model by ID."""
    response = client.get("/models/1")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == 1
    assert data["name"] == settings.MODEL_A_NAME


def test_get_model_not_found(client):
    """Should return 404 for a non-existent model ID."""
    response = client.get("/models/999")
    assert response.status_code == 404


def test_model_status_returns_available(client):
    """Model status should return available for configured models."""
    response = client.get("/models/1/status")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "available"
    assert data["name"] == settings.MODEL_A_NAME
    assert "endpoint_url" in data


def test_model_status_not_found(client):
    """Model status should return 404 for non-existent model."""
    response = client.get("/models/999/status")
    assert response.status_code == 404


def test_model_metadata_endpoint_exists(client):
    """Metadata endpoint should return 200."""
    response = client.get("/models/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "available" in data


def test_model_metadata_unavailable_when_no_url(client, monkeypatch):
    """Should return available=false when LITELLM_ADMIN_URL is empty."""
    monkeypatch.setattr(settings, "LITELLM_ADMIN_URL", "")
    response = client.get("/models/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is False
    assert data["models"] == []
