import logging
from typing import Set

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger("uvicorn")


class FrontendProxyMiddleware:
    """
    ASGI middleware that proxies requests to a frontend development server.
    
    Key features:
    - Routes non-API requests to frontend dev server
    - Handles request/response streaming
    - Manages headers and query parameters
    - Excludes specified API paths
    - Supports all HTTP methods
    
    Example usage:
        app.add_middleware(
            FrontendProxyMiddleware,
            frontend_endpoint="http://localhost:3000",
            excluded_paths={"/api", "/docs"}
        )
    """

    def __init__(
        self,
        app,
        frontend_endpoint: str,  # URL of frontend dev server (e.g. http://localhost:3000)
        excluded_paths: Set[str],  # Paths to not proxy (e.g. API routes)
    ):
        self.app = app  # ASGI application instance
        self.excluded_paths = excluded_paths
        self.frontend_endpoint = frontend_endpoint

    async def _request_frontend(
        self,
        request: Request,  # Incoming FastAPI request
        path: str,  # Request path to proxy
        timeout: float = 60.0,  # Max time to wait for frontend response
    ):
        """
        Proxies request to frontend server and streams response back.
        
        Steps:
        1. Creates HTTP client with timeout
        2. Forwards original request method, headers and body
        3. Streams response back with original status and headers
        4. Handles errors and connection issues
        """
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Construct full URL with query params
            url = f"{self.frontend_endpoint}/{path}"
            if request.query_params:
                url = f"{url}?{request.query_params}"

            # Forward original headers
            headers = dict(request.headers)
            try:
                # Get request body for non-GET requests
                body = await request.body() if request.method != "GET" else None

                # Make request to frontend
                response = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    content=body,
                    follow_redirects=True,
                )

                # Clean response headers
                response_headers = dict(response.headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)

                # Stream response back as it becomes available
                return StreamingResponse(
                    response.iter_bytes(),
                    status_code=response.status_code,
                    headers=response_headers,
                )
            except Exception as e:
                logger.error(f"Proxy error: {str(e)}")
                raise

    def _is_excluded_path(self, path: str) -> bool:
        """
        Checks if path should be excluded from proxying.
        Used to prevent proxying API requests to frontend.
        """
        return any(
            path.startswith(excluded_path) for excluded_path in self.excluded_paths
        )

    async def __call__(self, scope, receive, send):
        """
        ASGI interface implementation.
        Decides whether to proxy request or pass to next middleware.
        
        Flow:
        1. Check if HTTP request
        2. Check if path should be excluded
        3. Proxy to frontend or pass to app
        """
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request = Request(scope, receive)
        path = request.url.path

        if self._is_excluded_path(path):
            return await self.app(scope, receive, send)

        response = await self._request_frontend(request, path.lstrip("/"))
        return await response(scope, receive, send)
