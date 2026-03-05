"""
router.py — Monitoring endpoints.

GET /metrics           — Prometheus scrape endpoint
GET /health            — Liveness + readiness check
GET /grafana/dashboard — Grafana dashboard JSON (import directly)
"""

import logging
from fastapi import APIRouter, Response
from .metrics import get_metrics

monitoring_router = APIRouter()
logger = logging.getLogger(__name__)


@monitoring_router.get("/health", include_in_schema=True)
def health_check():
    """Liveness + readiness probe. Used by Docker, K8s, HF Spaces."""
    metrics = get_metrics()
    return {
        "status": "healthy",
        "metrics": metrics.available,
    }


@monitoring_router.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    """Prometheus scrape endpoint."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        return Response(
            content="# prometheus_client not installed\n",
            media_type="text/plain",
        )


@monitoring_router.get("/grafana/dashboard")
def grafana_dashboard():
    """
    Ready-to-import Grafana dashboard JSON.
    Import at: Grafana → Dashboards → Import → Paste JSON
    """
    return {
        "title": "Medical AI API",
        "uid": "medical-ai-api",
        "timezone": "browser",
        "refresh": "10s",
        "panels": [
            {
                "id": 1,
                "title": "Request Rate (req/min)",
                "type": "graph",
                "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                "targets": [
                    {
                        "expr": "rate(http_requests_total[1m]) * 60",
                        "legendFormat": "{{endpoint}} {{method}}",
                    }
                ],
            },
            {
                "id": 2,
                "title": "Request Latency P95 (s)",
                "type": "graph",
                "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                        "legendFormat": "P95 {{endpoint}}",
                    }
                ],
            },
            {
                "id": 3,
                "title": "Error Rate (%)",
                "type": "graph",
                "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
                "targets": [
                    {
                        "expr": "rate(http_requests_total{status_code=~'5..'}[5m]) / rate(http_requests_total[5m]) * 100",
                        "legendFormat": "Error % {{endpoint}}",
                    }
                ],
            },
            {
                "id": 4,
                "title": "Inference Duration P95 (s)",
                "type": "graph",
                "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8},
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m]))",
                        "legendFormat": "P95 inference {{model}}",
                    }
                ],
            },
            {
                "id": 5,
                "title": "Active Requests",
                "type": "stat",
                "gridPos": {"x": 0, "y": 16, "w": 6, "h": 4},
                "targets": [
                    {"expr": "sum(http_requests_in_progress)", "legendFormat": "Active"}
                ],
            },
            {
                "id": 6,
                "title": "Total Requests (24h)",
                "type": "stat",
                "gridPos": {"x": 6, "y": 16, "w": 6, "h": 4},
                "targets": [
                    {
                        "expr": "sum(increase(http_requests_total[24h]))",
                        "legendFormat": "Total",
                    }
                ],
            },
            {
                "id": 7,
                "title": "Total Inferences (24h)",
                "type": "stat",
                "gridPos": {"x": 12, "y": 16, "w": 6, "h": 4},
                "targets": [
                    {
                        "expr": "sum(increase(inference_requests_total[24h]))",
                        "legendFormat": "Inferences",
                    }
                ],
            },
            {
                "id": 8,
                "title": "Total Errors (24h)",
                "type": "stat",
                "gridPos": {"x": 18, "y": 16, "w": 6, "h": 4},
                "targets": [
                    {
                        "expr": "sum(increase(errors_total[24h]))",
                        "legendFormat": "Errors",
                    }
                ],
            },
        ],
        "templating": {"list": []},
        "time": {"from": "now-6h", "to": "now"},
    }
