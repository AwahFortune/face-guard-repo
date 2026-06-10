"""
POST /api/v1/verify   — 1:1 verification (user_id + image)
POST /api/v1/identify — 1:N identification (image only, returns best match)
"""
import asyncio
import logging

from fastapi import APIRouter, Depends, Form, Request, UploadFile, File
from pydantic import BaseModel
from typing import Optional

from ...core.security import require_auth
from ...api.deps import parse_image, validate_user_id
from ...services.face_service import verify_face, identify_face

logger = logging.getLogger(__name__)
router = APIRouter(tags=["recognition"])


class VerifyResponse(BaseModel):
    authorized: bool
    similarity: float
    confidence: str
    user_id: str


class IdentifyResponse(BaseModel):
    identified: bool
    user_id: Optional[str]
    score: float
    confidence: str


@router.post("/verify", response_model=VerifyResponse, summary="1:1 face verification")
async def verify(
    request: Request,
    user_id: str = Form(...),
    image: UploadFile = File(...),
    _subject: str = Depends(require_auth),
) -> VerifyResponse:
    uid = validate_user_id(user_id)
    img_array = await parse_image(image)
    ip = request.client.host if request.client else "unknown"
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: verify_face(uid, img_array, ip))
    return VerifyResponse(
        authorized=result["authorized"],
        similarity=result["similarity"],
        confidence=f"{result['similarity'] * 100:.1f}%",
        user_id=uid,
    )


@router.post("/identify", response_model=IdentifyResponse, summary="1:N face identification")
async def identify(
    image: UploadFile = File(...),
    _subject: str = Depends(require_auth),
) -> IdentifyResponse:
    img_array = await parse_image(image)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: identify_face(img_array))
    return IdentifyResponse(
        identified=result["identified"],
        user_id=result.get("user_id"),
        score=result["score"],
        confidence=f"{result['score'] * 100:.1f}%",
    )
