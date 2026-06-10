"""
Application-level exceptions and FastAPI exception handlers.
Never expose internal details to the caller.
"""
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class FaceNotDetected(Exception):
    pass


class LivenessFailed(Exception):
    pass


class UserNotFound(Exception):
    pass


class UserAlreadyExists(Exception):
    pass


class EncryptionError(Exception):
    pass


class DatabaseError(Exception):
    pass


class ImageQualityError(Exception):
    pass


class AccountLocked(Exception):
    def __init__(self, seconds_remaining: int):
        self.seconds_remaining = seconds_remaining
        super().__init__(f"Account locked for {seconds_remaining}s")


def register_handlers(app: FastAPI) -> None:
    @app.exception_handler(FaceNotDetected)
    async def face_not_detected(_req: Request, exc: FaceNotDetected):
        return JSONResponse(status_code=422,
                            content={"error": "FACE_NOT_DETECTED",
                                     "message": "No face detected in the image"})

    @app.exception_handler(LivenessFailed)
    async def liveness_failed(_req: Request, exc: LivenessFailed):
        return JSONResponse(status_code=422,
                            content={"error": "LIVENESS_FAILED",
                                     "message": "Liveness check failed — real face required"})

    @app.exception_handler(UserNotFound)
    async def user_not_found(_req: Request, exc: UserNotFound):
        return JSONResponse(status_code=404,
                            content={"error": "USER_NOT_FOUND",
                                     "message": "User not registered"})

    @app.exception_handler(UserAlreadyExists)
    async def user_exists(_req: Request, exc: UserAlreadyExists):
        return JSONResponse(status_code=409,
                            content={"error": "USER_EXISTS",
                                     "message": "User ID already registered"})

    @app.exception_handler(AccountLocked)
    async def account_locked(_req: Request, exc: AccountLocked):
        return JSONResponse(
            status_code=423,
            content={"error": "ACCOUNT_LOCKED",
                     "message": f"Too many failed attempts. Try again in {exc.seconds_remaining}s"},
        )

    @app.exception_handler(ImageQualityError)
    async def image_quality(_req: Request, exc: ImageQualityError):
        return JSONResponse(status_code=422,
                            content={"error": "IMAGE_QUALITY",
                                     "message": str(exc)})

    @app.exception_handler(Exception)
    async def generic(_req: Request, exc: Exception):
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(status_code=500,
                            content={"error": "INTERNAL_ERROR",
                                     "message": "An internal error occurred"})
