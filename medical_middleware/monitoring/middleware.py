"""
middleware.py — Prometheus metrics collection middleware.
Tracks every request automatically.
"""

import time
import logging
from typing import TYPE_CHECKING

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .metrics import get_metrics

if TYPE_CHECKING:
    from ..config import MiddlewareConfig

logger = logging.getLogger(__name__)

# Endpoints to exclude from metrics (noisy, low-value)
EXCLUDE_PATHS = {
    "/metrics",
    "/health",
    "/favicon.ico",
    "/docs",
    "/openapi.json",
    "/redoc",
}


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware that records Prometheus metrics for every request.

    Tracks:
    - Request count by endpoint/method/status
    - Request latency histogram
    - Active requests gauge
    """

    def __init__(self, app, config: "MiddlewareConfig" = None):
        super().__init__(app)
        from ..config import MiddlewareConfig

        self.config = config or MiddlewareConfig()
        self.metrics = get_metrics(self.config.app_name)
        logger.info("[monitoring] Prometheus middleware initialized")

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        # Skip excluded paths
        if path in EXCLUDE_PATHS:
            return await call_next(request)

        method = request.method
        start_time = time.time()

        with self.metrics.track_request(path, method):
            try:
                response = await call_next(request)
                status_code = response.status_code
            except Exception as e:
                self.metrics.record_error(path, type(e).__name__)
                raise

        duration = time.time() - start_time
        self.metrics.record_request(path, method, status_code, duration)

        return response
