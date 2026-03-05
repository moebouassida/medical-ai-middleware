"""
core.py — Main setup functions for medical AI middleware.
"""

from fastapi import FastAPI
from .gdpr.middleware import GDPRMiddleware
from .gdpr.router import gdpr_router
from .monitoring.middleware import PrometheusMiddleware
from .monitoring.router import monitoring_router
from .ratelimit.middleware import setup_rate_limiter
from .config import MiddlewareConfig


def setup_middleware(
    app: FastAPI,
    config: MiddlewareConfig = None,
) -> FastAPI:
    """
    One-line setup for all production middleware.

    Args:
        app:    FastAPI application instance
        config: Optional MiddlewareConfig — uses safe defaults if not provided

    Returns:
        app with all middleware configured

    Example:
        app = FastAPI()
        setup_middleware(app)
    """
    cfg = config or MiddlewareConfig()
    setup_gdpr(app, cfg)
    setup_monitoring(app, cfg)
    setup_ratelimit(app, cfg)
    return app


def setup_gdpr(app: FastAPI, config: MiddlewareConfig = None) -> FastAPI:
    """
    Add GDPR compliance layer:
      - Audit logging (every request logged)
      - Data retention enforcement (auto-delete after TTL)
      - Right to erasure endpoint (DELETE /gdpr/erase/{request_id})
      - Image anonymization (strip metadata)
      - Consent header enforcement
      - Security response headers
    """
    cfg = config or MiddlewareConfig()
    app.add_middleware(GDPRMiddleware, config=cfg)
    app.include_router(gdpr_router, prefix="/gdpr", tags=["GDPR"])
    return app


def setup_monitoring(app: FastAPI, config: MiddlewareConfig = None) -> FastAPI:
    """
    Add Prometheus + Grafana monitoring:
      - GET /metrics — Prometheus scrape endpoint
      - Request count, latency, error rate per endpoint
      - Inference duration tracking
      - Active request gauge
    """
    cfg = config or MiddlewareConfig()
    app.add_middleware(PrometheusMiddleware, config=cfg)
    app.include_router(monitoring_router, tags=["Monitoring"])
    return app


def setup_ratelimit(app: FastAPI, config: MiddlewareConfig = None) -> FastAPI:
    """
    Add API rate limiting:
      - Per-IP sliding window rate limiting
      - Configurable limits per endpoint
      - 429 responses with Retry-After header
    """
    cfg = config or MiddlewareConfig()
    setup_rate_limiter(app, cfg)
    return app
