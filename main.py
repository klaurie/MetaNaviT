"""
MetaNaviT FastAPI Application Entry Point

This module serves as the main entry point for the MetaNaviT application.
It handles server configuration, middleware setup, and static file mounting
based on the environment (development or production).

Features:
    - FastAPI application initialization
    - Environment-based configuration
    - Static file serving
    - Frontend proxy middleware (dev mode)

Environment Variables:
    ENVIRONMENT: Running environment (dev/prod)
    FRONTEND_ENDPOINT: Frontend server URL for dev proxy
    APP_HOST: Server host address
    APP_PORT: Server port number

Dependencies:
    - FastAPI for API framework
    - Uvicorn for ASGI server
    - Custom middleware for frontend proxying
"""
    
# flake8: noqa: E402
from app.config import DATA_DIR, STATIC_DIR
from dotenv import load_dotenv

load_dotenv()

import logging
import os

import uvicorn
from app.api.routers import api_router
from app.middlewares.frontend import FrontendProxyMiddleware
from app.settings import init_settings
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

# Initialize FastAPI app
app = FastAPI()

init_settings()

environment = os.getenv("ENVIRONMENT", "dev")
logger = logging.getLogger("uvicorn")


def mount_static_files(directory, path, html=False):
    """
    Mount static file directories to serve through FastAPI.
    
    Args:
        directory: Local directory path to mount
        path: URL path to mount the directory at
        html: Whether to serve index.html for directory roots
    """
    if os.path.exists(directory):
        logger.info(f"Mounting static files '{directory}' at '{path}'")
        app.mount(
            path,
            StaticFiles(directory=directory, check_dir=False, html=html),
            name=f"{directory}-static",
        )


app.include_router(api_router, prefix="/api")

# Mount the data files to serve the file viewer
mount_static_files(DATA_DIR, "/api/files/data")
# Mount the output files from tools
mount_static_files("output", "/api/files/output")
 # Development mode: Use frontend proxy if configured
if environment == "dev":
    frontend_endpoint = os.getenv("FRONTEND_ENDPOINT")
    if frontend_endpoint:
        app.add_middleware(
            FrontendProxyMiddleware,
            frontend_endpoint=frontend_endpoint,
            excluded_paths=set(
                route.path for route in app.routes if hasattr(route, "path")
            ),
        )
    else:
        # No frontend in dev: redirect to API docs
        logger.warning("No frontend endpoint - starting API server only")

        @app.get("/")
        async def redirect_to_docs():
            return RedirectResponse(url="/docs")
else:
    # Mount the frontend static files (production)
    mount_static_files(STATIC_DIR, "/", html=True)

if __name__ == "__main__":
    app_host = os.getenv("APP_HOST", "0.0.0.0")
    app_port = int(os.getenv("APP_PORT", "8000"))
    reload = True if environment == "dev" else False

    uvicorn.run(app="main:app", host=app_host, port=app_port, reload=reload)
