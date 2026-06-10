"""
POST /api/v1/auth/token  — exchange API key for a short-lived JWT.
"""
import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ...core.security import create_access_token, verify_api_key

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


class TokenRequest(BaseModel):
    api_key: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/token", response_model=TokenResponse, summary="Exchange API key for JWT")
def get_token(body: TokenRequest) -> TokenResponse:
    if not verify_api_key(body.api_key):
        logger.warning("Failed API key attempt")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid API key")
    token = create_access_token(subject="service")
    return TokenResponse(access_token=token)
