"""
metrics.py — Prometheus metrics definitions for medical AI APIs.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge, REGISTRY

    _prometheus_available = True
except ImportError:
    _prometheus_available = False
    logger.warning("[metrics] prometheus_client not installed — metrics disabled")

_metrics_instance = None


class MedicalAIMetrics:
    """
    Prometheus metrics for medical AI APIs.
    Singleton — one instance per application.
    Safe against duplicate registration (test environments / hot reload).
    """

    def __init__(self, app_name: str = "medical_ai_api", registry=None):
        if not _prometheus_available:
            self._available = False
            return

        self._available = True
        self.app_name = app_name
        labels = ["app", "endpoint", "method"]

        # Use _safe_metric() to avoid duplicate registration errors
        self.request_count = self._safe_metric(
            Counter,
            "http_requests_total",
            "Total HTTP requests",
            labels + ["status_code"],
            registry=registry,
        )
        self.request_latency = self._safe_metric(
            Histogram,
            "http_request_duration_seconds",
            "HTTP request latency",
            labels,
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            registry=registry,
        )
        self.requests_in_progress = self._safe_metric(
            Gauge,
            "http_requests_in_progress",
            "Active HTTP requests",
            ["app", "endpoint", "method"],
            registry=registry,
        )
        self.inference_duration = self._safe_metric(
            Histogram,
            "inference_duration_seconds",
            "Model inference duration",
            ["app", "model", "endpoint"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=registry,
        )
        self.inference_count = self._safe_metric(
            Counter,
            "inference_requests_total",
            "Total inference requests",
            ["app", "model", "endpoint", "status"],
            registry=registry,
        )
        self.error_count = self._safe_metric(
            Counter,
            "errors_total",
            "Total errors by type",
            ["app", "endpoint", "error_type"],
            registry=registry,
        )

    @staticmethod
    def _safe_metric(metric_class, name, description, labels, registry=None, **kwargs):
        """
        Create a Prometheus metric, or return the existing one if already registered.
        Prevents ValueError: Duplicated timeseries in hot-reload / test environments.
        """
        try:
            return metric_class(name, description, labels, registry=registry, **kwargs)
        except ValueError:
            # Already registered — retrieve from default registry
            collectors = list(REGISTRY._names_to_collectors.get(name, [None]))
            if collectors and collectors[0] is not None:
                return collectors[0]
            # Fallback: try direct lookup
            for collector in REGISTRY._collectors:
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            raise

    def record_request(
        self, endpoint: str, method: str, status_code: int, duration: float
    ):
        if not self._available:
            return
        labels = [self.app_name, endpoint, method]
        self.request_count.labels(*labels, str(status_code)).inc()
        self.request_latency.labels(*labels).observe(duration)

    def record_inference(
        self, endpoint: str, model: str, duration: float, success: bool = True
    ):
        if not self._available:
            return
        status = "success" if success else "error"
        self.inference_duration.labels(self.app_name, model, endpoint).observe(duration)
        self.inference_count.labels(self.app_name, model, endpoint, status).inc()

    def record_error(self, endpoint: str, error_type: str):
        if not self._available:
            return
        self.error_count.labels(self.app_name, endpoint, error_type).inc()

    def track_request(self, endpoint: str, method: str):
        if not self._available:
            return _NoopContext()
        return _InProgressContext(
            self.requests_in_progress, [self.app_name, endpoint, method]
        )

    @property
    def available(self) -> bool:
        return self._available


class _InProgressContext:
    def __init__(self, gauge, labels):
        self.gauge = gauge
        self.labels = labels

    def __enter__(self):
        self.gauge.labels(*self.labels).inc()
        return self

    def __exit__(self, *args):
        self.gauge.labels(*self.labels).dec()


class _NoopContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def get_metrics(app_name: str = "medical_ai_api") -> MedicalAIMetrics:
    """Get or create the global metrics singleton."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MedicalAIMetrics(app_name)
    return _metrics_instance
