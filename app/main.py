"""
Main FastAPI application for the NexusRAG Multi-Tenant System.

Startup sequence:
  1. Configure structured logging (structlog).
  2. Initialize the database (create tables).
  3. Build the ServiceContainer — all heavy singletons created once here.
  4. Store container on ``app.state`` for dependency injection.
"""
import structlog
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.config import settings
from app.database import init_db
from app.core.container import ServiceContainer
from app.api import auth_router, documents_router, queries_router, tenants_router

# ── Logging configuration ─────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# ── Application lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application startup and shutdown.

    All heavy services (embedding model, Qdrant client, LLM providers) are
    initialized here ONCE and stored on ``app.state.container``.  Route
    handlers retrieve them via lightweight dependency functions — no service
    is ever instantiated per-request.
    """
    logger.info("Starting NexusRAG", version=settings.app_version)

    # 1. Initialize relational database (create tables if absent)
    await init_db()
    logger.info("Database initialized")

    # 2. Build singleton service container
    container = await ServiceContainer.create()
    app.state.container = container
    logger.info("Service container ready")

    logger.info("Application startup complete")
    yield

    # Shutdown — nothing to explicitly teardown for now, but the hook is here.
    logger.info("Shutting down NexusRAG")


# ── Application factory ───────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "A production-grade Multi-Tenant Retrieval-Augmented Generation (RAG) System "
        "with strict tenant isolation, pluggable LLM providers, and semantic vector search."
    ),
    lifespan=lifespan,
    # Only expose interactive docs in debug mode
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_hosts,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Emit a structured log entry for every HTTP exception and return JSON."""
    logger.warning(
        "HTTP exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler: log full traceback, return a safe message to the client."""
    logger.error(
        "Unhandled exception",
        exc_info=True,
        path=request.url.path,
        method=request.method,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc) if settings.debug else "Internal server error",
            "status_code": 500,
        },
    )


# ── Health endpoints ──────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Lightweight liveness probe — always returns 200 if the process is alive."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "app_name": settings.app_name,
    }


@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check(request: Request) -> dict:
    """
    Readiness probe that checks every downstream dependency.

    Uses the shared ServiceContainer from ``app.state`` so no new connections
    are created per call.
    """
    container: ServiceContainer = request.app.state.container
    component_status = await container.health_check()

    # Check PostgreSQL connectivity using SQLAlchemy 2.x-compatible syntax
    from app.database.session import engine
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        component_status["database"] = "healthy"
    except Exception as exc:
        component_status["database"] = f"unhealthy: {exc}"

    overall = "healthy" if all(
        v.startswith("healthy") for v in component_status.values()
    ) else "degraded"

    return {
        "status": overall,
        "version": settings.app_version,
        "app_name": settings.app_name,
        "components": component_status,
    }


# ── API routers ───────────────────────────────────────────────────────────────

app.include_router(auth_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(queries_router, prefix="/api/v1")
app.include_router(tenants_router, prefix="/api/v1")


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root() -> dict:
    """API discovery endpoint."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs" if settings.debug else "Disabled in production",
        "health": "/health",
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )