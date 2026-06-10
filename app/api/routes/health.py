"""
GET /api/v1/health — liveness + readiness probe.
"""
import logging
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ...db.pool import get_conn
from ...db.vector import get_collection
from ...services.face_service import get_models

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", summary="Health check")
def health() -> JSONResponse:
    checks: dict = {}
    ok = True

    # MySQL
    try:
        with get_conn() as (_, cur):
            cur.execute("SELECT 1")
            cur.fetchone()
        checks["mysql"] = "ok"
    except Exception as exc:
        checks["mysql"] = f"error: {exc}"
        ok = False

    # Milvus
    try:
        col = get_collection()
        checks["milvus"] = f"ok (entities={col.num_entities})"
    except Exception as exc:
        checks["milvus"] = f"error: {exc}"
        ok = False

    # ML models
    try:
        models = get_models()
        checks["insightface"] = "ok" if models["insightface"] else "not loaded"
        checks["image_processor"] = "ok" if models["image_processor"] else "not loaded"
        checks["liveness"] = "ok" if models["liveness"] else "not loaded"
        checks["encryptor"] = "ok" if models["encryptor"] else "not loaded"
    except Exception as exc:
        checks["models"] = f"error: {exc}"
        ok = False

    checks["timestamp"] = int(time.time())
    status_str = "healthy" if ok else "degraded"
    checks["status"] = status_str

    return JSONResponse(
        status_code=200 if ok else 503,
        content=checks,
    )
