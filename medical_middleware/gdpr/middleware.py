"""
middleware.py — GDPR FastAPI middleware.

Intercepts every request to:
  1. Assign a unique request ID
  2. Check consent header on sensitive endpoints
  3. Add security response headers
  4. Log to audit trail (S3 if enabled, local fallback)
  5. Enforce data retention on uploaded files (S3 if enabled, local fallback)
"""

import time
import uuid
import logging
from typing import TYPE_CHECKING

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .audit import AuditLogger
from .retention import DataRetentionManager

if TYPE_CHECKING:
    from ..config import MiddlewareConfig

logger = logging.getLogger(__name__)

_audit_logger = None
_retention_manager = None


def get_audit_logger():
    return _audit_logger


def get_retention_manager():
    return _retention_manager


class GDPRMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, config: "MiddlewareConfig" = None):
        super().__init__(app)
        from ..config import MiddlewareConfig

        self.config = config or MiddlewareConfig()
        global _audit_logger, _retention_manager

        if self.config.s3_enabled:
            try:
                from ..storage.s3 import get_s3_client
                from ..storage.audit_s3 import S3AuditLogger
                from ..storage.retention_s3 import S3RetentionManager

                s3 = get_s3_client()
                if s3.available:
                    _audit_logger = S3AuditLogger(
                        app_name=self.config.app_name,
                        s3_client=s3,
                        local_fallback=self.config.audit_log_path,
                    )
                    _retention_manager = S3RetentionManager(
                        s3_client=s3,
                        retention_seconds=self.config.data_retention_seconds,
                        local_temp_path=self.config.temp_storage_path,
                    )
                    logger.info("[gdpr] S3-backed audit + retention initialized")
                else:
                    raise RuntimeError("S3 not available")
            except Exception as e:
                logger.warning(f"[gdpr] S3 init failed ({e}) — using local storage")
                self._init_local()
        else:
            self._init_local()

        logger.info(
            f"[gdpr] Ready — storage={'s3' if self.config.s3_enabled else 'local'}, app={self.config.app_name}"
        )

    def _init_local(self):
        global _audit_logger, _retention_manager
        _audit_logger = AuditLogger(self.config.audit_log_path)
        _retention_manager = DataRetentionManager(
            retention_seconds=self.config.data_retention_seconds,
            storage_path=self.config.temp_storage_path,
        )

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.time()

        consent_given = True
        if self.config.require_consent_header:
            for path in self.config.consent_required_paths:
                if request.url.path.startswith(path):
                    consent = request.headers.get("X-Data-Consent", "").lower()
                    if consent != "true":
                        consent_given = False
                        duration_ms = (time.time() - start_time) * 1000
                        _audit_logger.log(
                            request_id=request_id,
                            endpoint=request.url.path,
                            method=request.method,
                            status_code=403,
                            duration_ms=duration_ms,
                            client_ip=self._get_client_ip(request),
                            consent_given=False,
                        )
                        return JSONResponse(
                            status_code=403,
                            content={
                                "error": "consent_required",
                                "message": "Processing medical images requires explicit consent. Add header: X-Data-Consent: true",
                                "gdpr_article": "Article 9 GDPR — Special categories of personal data",
                            },
                            headers={"X-Request-ID": request_id},
                        )
                    break

        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"[gdpr] Unhandled error for {request_id}: {e}")
            raise

        duration_ms = (time.time() - start_time) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Content-Security-Policy"] = self.config.csp_policy
        response.headers["Strict-Transport-Security"] = (
            f"max-age={self.config.hsts_max_age}; includeSubDomains"
        )

        _audit_logger.log(
            request_id=request_id,
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration_ms=duration_ms,
            client_ip=self._get_client_ip(request),
            consent_given=consent_given,
            user_agent=request.headers.get("user-agent"),
        )

        return response

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"
