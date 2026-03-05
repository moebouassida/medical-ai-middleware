"""
medical-ai-middleware
=====================
Production-grade security, compliance, and observability middleware
for medical AI APIs. GDPR compliant. Prometheus + Grafana ready.

Usage:
    from medical_middleware import setup_middleware
    setup_middleware(app)

Or selectively:
    from medical_middleware import setup_gdpr, setup_monitoring, setup_ratelimit
    setup_gdpr(app)
    setup_monitoring(app)
    setup_ratelimit(app)
"""

from .core import setup_middleware, setup_gdpr, setup_monitoring, setup_ratelimit
from .xai import GradCAM, AttentionMap, make_explain_router

__version__ = "1.1.0"
__author__ = "Moez Bouassida"

__all__ = [
    "setup_middleware",
    "setup_gdpr",
    "setup_monitoring",
    "setup_ratelimit",
    "GradCAM",
    "AttentionMap",
    "make_explain_router",
]
