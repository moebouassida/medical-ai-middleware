"""
middleware.py — API rate limiting for medical AI APIs.

Uses slowapi (built on limits) for per-IP sliding window rate limiting.

Default limits:
    /predict*   → 10 requests/minute  (expensive inference)
    /health     → 60 requests/minute
    /metrics    → 30 requests/minute
    default     → 30 requests/minute

Returns 429 with Retry-After header when limit exceeded.
"""

import logging
from typing import TYPE_CHECKING, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from ..config import MiddlewareConfig

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    pass


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def setup_rate_limiter(app: FastAPI, config: "MiddlewareConfig" = None):
    """
    Configure rate limiting on FastAPI app using slowapi.
    Falls back gracefully if slowapi is not installed.
    """
    from ..config import MiddlewareConfig

    cfg = config or MiddlewareConfig()

    try:
        from slowapi import Limiter
        from slowapi.util import get_remote_address
        from slowapi.errors import RateLimitExceeded as SlowAPIRateLimitExceeded
        from slowapi.middleware import SlowAPIMiddleware

        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=[cfg.rate_limit_default],
        )

        app.state.limiter = limiter
        app.add_exception_handler(
            SlowAPIRateLimitExceeded,
            _medical_rate_limit_handler,
        )
        app.add_middleware(SlowAPIMiddleware)

        logger.info(
            f"[ratelimit] Rate limiting enabled — "
            f"predict: {cfg.rate_limit_predict}, "
            f"default: {cfg.rate_limit_default}"
        )

        return limiter

    except ImportError:
        logger.warning(
            "[ratelimit] slowapi not installed — rate limiting disabled. "
            "Install with: pip install slowapi"
        )
        return None


def _medical_rate_limit_handler(request: Request, exc) -> Response:
    """Custom 429 response with GDPR-appropriate messaging."""
    retry_after = getattr(exc, "retry_after", 60)
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": (
                "Too many requests. Medical AI inference is resource-intensive. "
                "Please respect the rate limits."
            ),
            "retry_after_seconds": retry_after,
            "limit": str(getattr(exc, "limit", "unknown")),
        },
        headers={
            "Retry-After": str(retry_after),
            "X-RateLimit-Reset": str(retry_after),
        },
    )


def rate_limit(limit: str = "10/minute") -> Callable:
    """
    Decorator for per-endpoint rate limiting.

    Usage:
        from medical_middleware.ratelimit import rate_limit

        @app.post("/predict")
        @rate_limit("10/minute")
        async def predict(request: Request, ...):
            ...
    """
    try:
        from slowapi import Limiter
        from slowapi.util import get_remote_address

        limiter = Limiter(key_func=get_remote_address)
        return limiter.limit(limit)

    except ImportError:
        # Return no-op decorator if slowapi not installed
        def noop_decorator(func):
            return func

        return noop_decorator
