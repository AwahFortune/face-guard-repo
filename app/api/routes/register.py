"""
POST /api/v1/register  — enroll a user's face.

Accepts:
  multipart/form-data:
    user_id : str
    image   : image file (JPEG/PNG/WebP)

Liveness is checked passively via DeepFace anti-spoofing.
"""
import asyncio
import logging

from fastapi import APIRouter, Depends, Form, Request, UploadFile, File
from pydantic import BaseModel

from ...core.security import require_auth
from ...api.deps import parse_image, validate_user_id
from ...services.face_service import register_face

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/register", tags=["register"])


class RegisterResponse(BaseModel):
    status: str
    user_id: str
    det_score: float
    message: str


@router.post("", response_model=RegisterResponse, status_code=201,
             summary="Register a face")
async def register(
    request: Request,
    user_id: str = Form(...),
    image: UploadFile = File(..., description="Face image JPEG/PNG/WebP"),
    _subject: str = Depends(require_auth),
) -> RegisterResponse:
    uid = validate_user_id(user_id)
    img_array = await parse_image(image)
    ip = request.client.host if request.client else "unknown"
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: register_face(uid, img_array, ip))
    return RegisterResponse(
        status="success",
        user_id=uid,
        det_score=result["det_score"],
        message="Face registered successfully",
    )
