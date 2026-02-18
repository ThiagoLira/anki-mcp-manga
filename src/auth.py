from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import settings

# Paths that don't require authentication
PUBLIC_PATHS = {"/health"}


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Authenticate requests via Bearer token header or ?token= query param."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        token = None

        # Check Authorization header
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

        # Fallback to query parameter (for Claude Web connectors)
        if not token:
            token = request.query_params.get("token")

        if token != settings.mcp_auth_token:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or missing authentication token"},
            )

        return await call_next(request)
