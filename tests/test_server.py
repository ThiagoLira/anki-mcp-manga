import os
import tempfile

import pytest
from starlette.testclient import TestClient

# Set test env vars before importing server
os.environ.setdefault("MCP_AUTH_TOKEN", "test-token-123")
os.environ.setdefault("COLLECTION_PATH", "/tmp/test_server.anki2")
os.environ.setdefault("SYNC_ENDPOINT", "http://localhost:8080")


class TestAuth:
    """Test the auth middleware independently."""

    def test_missing_token_returns_401(self):
        from src.server import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/mcp")
        assert resp.status_code == 401

    def test_wrong_token_returns_401(self):
        from src.server import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/mcp", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

    def test_valid_bearer_token(self):
        from src.server import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/mcp", headers={"Authorization": "Bearer test-token-123"}
        )
        # Should not be 401 (may be 405 or other, but auth passes)
        assert resp.status_code != 401

    def test_valid_query_token(self):
        from src.server import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/mcp?token=test-token-123")
        assert resp.status_code != 401

    def test_health_endpoint_no_auth(self):
        from src.server import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        # Health check should not require auth (404 is fine — no handler, but not 401)
        assert resp.status_code != 401
