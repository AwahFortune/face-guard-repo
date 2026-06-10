"""
FastAPI application entry point.
Lifespan initialises all models and DB connections once at startup.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.exceptions import register_handlers
from .core.logging_setup import configure_logging
from .api.middleware import RateLimitMiddleware, SecurityHeadersMiddleware
from .api.routes import auth, health, register, recognize
from .db.pool import init_pool
from .db.vector import init_milvus
from .services.face_service import init_models

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup — initialising subsystems")
    init_pool()
    logger.info("MySQL pool ready")
    try:
        init_milvus()
        logger.info("Milvus ready")
    except Exception as exc:
        logger.warning("Milvus unavailable — recognition endpoints will return 503: %s", exc)
    init_models()
    logger.info("All subsystems ready")
    yield
    logger.info("Shutdown complete")


app = FastAPI(
    title="Face-Guard Face Recognition API",
    version="2.0.0",
    docs_url="/docs" if settings.ENV != "production" else None,
    redoc_url=None,
    lifespan=lifespan,
)

# Middleware — outermost is applied last in the stack, so add in reverse priority
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# Exception handlers
register_handlers(app)

# API routes
app.include_router(auth.router,      prefix="/api/v1")
app.include_router(health.router,    prefix="/api/v1")
app.include_router(register.router,  prefix="/api/v1")
app.include_router(recognize.router, prefix="/api/v1")
