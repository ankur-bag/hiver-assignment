import os
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend.api.app import app

client = TestClient(app)


def test_health_is_quota_free_and_reports_configuration():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "key", "GEMINI_FILE_SEARCH_STORE": "fileSearchStores/test"}):
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["services"] == {"application": "ready", "file_search_store": "configured", "gemini": "configured"}


def test_security_headers_present():
    response = client.get("/api/v1/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_edge_auth_required_in_production():
    with patch.dict(os.environ, {"APP_ENV": "production", "EDGE_SHARED_SECRET": "secret"}, clear=False):
        response = client.post("/api/v1/chat", json={"text": "hello"})
    assert response.status_code == 401


def test_valid_edge_auth_reaches_route():
    with patch.dict(os.environ, {"APP_ENV": "production", "EDGE_SHARED_SECRET": "secret"}, clear=False):
        response = client.post("/api/v1/chat", headers={"X-Hiver-Edge-Auth": "secret"}, json={"text": "   "})
    assert response.status_code == 400
