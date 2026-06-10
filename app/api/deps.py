"""
FastAPI dependencies injected into route handlers.
"""
import io
import logging

import numpy as np
from fastapi import HTTPException, UploadFile, status
from PIL import Image

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


async def parse_image(file: UploadFile) -> np.ndarray:
    """
    Validate and decode an uploaded image.
    Returns an RGB uint8 numpy array.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type '{file.content_type}'. "
                   f"Use JPEG, PNG, or WebP.",
        )

    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds 10 MB limit",
        )

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception as exc:
        logger.warning("Image decode failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Cannot decode image") from exc


def validate_user_id(user_id: str) -> str:
    """
    Sanitise user_id: alphanumeric + hyphens/underscores, 1–64 chars.
    Prevents injection and unexpected keys in MySQL / Milvus.
    """
    import re
    if not user_id or not re.match(r'^[A-Za-z0-9_\-]{1,64}$', user_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="user_id must be 1–64 alphanumeric characters (hyphens/underscores allowed)",
        )
    return user_id
