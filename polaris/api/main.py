"""FastAPI application for the Polaris Todo/Scheduling system."""

import logging
from contextlib import asynccontextmanager

# Load .env before anything reads os.getenv()
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from polaris.api.config import get_api_config
from polaris.api.dependencies import init_managers, shutdown_managers

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize and clean up shared resources."""
    config = get_api_config()
    logger.info(f"Starting Polaris API on {config.host}:{config.port}")

    # Initialize Redis-backed managers
    await init_managers(config.redis_url)
    logger.info("Managers initialized")

    yield

    # Shutdown
    await shutdown_managers()
    logger.info("Polaris API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_api_config()

    app = FastAPI(
        title="Polaris API",
        description="Personal productivity backend — todos, reminders, scheduling",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS (development only)
    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Register routers
    from polaris.api.routers import auth, todos, reminders, push, websocket

    app.include_router(auth.router)
    app.include_router(todos.router)
    app.include_router(reminders.router)
    app.include_router(push.router)
    app.include_router(websocket.router)

    # Mount static files (Next.js export) if directory exists
    if config.static_dir:
        from pathlib import Path

        if Path(config.static_dir).exists():
            from fastapi.staticfiles import StaticFiles

            app.mount(
                "/",
                StaticFiles(directory=config.static_dir, html=True),
                name="static",
            )
            logger.info(f"Serving static files from {config.static_dir}")

    # OpenAPI security scheme so Swagger UI shows the Authorize button
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
            }
        }
        schema["security"] = [{"BearerAuth": []}]
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi

    # Health check
    @app.get("/api/health")
    async def health():
        return {"status": "ok", "service": "polaris"}

    # Root — shows API info when no static files are mounted
    @app.get("/")
    async def root():
        return {
            "service": "polaris",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    from pathlib import Path

    config = get_api_config()

    # Use HTTPS if certs exist
    project_root = Path(__file__).parent.parent.parent
    cert_file = project_root / "hyperion-cert.pem"
    key_file = project_root / "hyperion-key.pem"

    ssl_kwargs = {}
    if cert_file.exists() and key_file.exists():
        ssl_kwargs["ssl_certfile"] = str(cert_file)
        ssl_kwargs["ssl_keyfile"] = str(key_file)

    uvicorn.run(
        "polaris.api.main:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level="info",
        **ssl_kwargs,
    )
