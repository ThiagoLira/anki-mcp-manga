import os

import pytest
from starlette.testclient import TestClient

# Set test env vars before importing server
os.environ.setdefault("MCP_AUTH_TOKEN", "test-token-123")
os.environ.setdefault("COLLECTION_PATH", "/tmp/test_server.anki2")
os.environ.setdefault("SYNC_ENDPOINT", "http://localhost:8080")


class TestSecretPathAuth:
    """Test that the MCP endpoint is only reachable via the secret path."""

    def test_wrong_path_returns_404(self):
        from src.server import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/mcp")
        assert resp.status_code == 404

    def test_correct_path_not_404(self):
        from src.server import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test-token-123/mcp")
        assert resp.status_code != 404

    def test_trailing_slash_stripped(self):
        from src.server import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test-token-123/mcp/")
        # Should not be a redirect (307) — slash is stripped
        assert resp.status_code != 307
