"""
Security headers + rate limiting middleware.
"""
import logging
import time
from collections import defaultdict
from threading import Lock
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ── Security headers ──────────────────────────────────────────────────────────

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'none'",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value
        return response


# ── In-process rate limiter (per IP, sliding window) ─────────────────────────

class _Window:
    __slots__ = ("count", "start")

    def __init__(self):
        self.count = 0
        self.start = time.time()


_windows: dict[str, _Window] = defaultdict(_Window)
_lock = Lock()


def _check_rate(ip: str, limit: int, window_secs: float) -> bool:
    """Return True if allowed, False if over limit."""
    with _lock:
        w = _windows[ip]
        now = time.time()
        if now - w.start > window_secs:
            w.count = 1
            w.start = now
            return True
        w.count += 1
        return w.count <= limit


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Global 120 req/min per IP with fast 429 response."""

    LIMIT = 120
    WINDOW = 60.0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        ip = request.client.host if request.client else "unknown"
        if not _check_rate(f"global:{ip}", self.LIMIT, self.WINDOW):
            logger.warning("Rate limit exceeded for %s", ip)
            return Response(
                content='{"error":"RATE_LIMITED","message":"Too many requests"}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )
        return await call_next(request)
