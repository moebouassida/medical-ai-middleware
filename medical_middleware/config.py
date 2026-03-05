"""
config.py — Configuration for medical AI middleware.
All settings have safe production defaults.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class MiddlewareConfig:
    # ── App ───────────────────────────────────────────────────────
    app_name: str = os.getenv("APP_NAME", "medical-ai-api")

    # ── GDPR ──────────────────────────────────────────────────────
    data_retention_seconds: int = int(os.getenv("DATA_RETENTION_SECONDS", 86400))
    require_consent_header: bool = (
        os.getenv("REQUIRE_CONSENT", "true").lower() == "true"
    )
    consent_required_paths: List[str] = field(
        default_factory=lambda: [
            "/predict",
            "/predict/upload",
            "/predict/slices",
            "/explain/predict",
        ]
    )
    audit_log_path: str = os.getenv("AUDIT_LOG_PATH", "/tmp/logs/audit.jsonl")
    anonymize_images: bool = os.getenv("ANONYMIZE_IMAGES", "true").lower() == "true"

    # ── S3 ────────────────────────────────────────────────────────
    # Set S3_ENABLED=true + AWS credentials to enable S3-backed storage
    # Falls back to local files if S3 is disabled or unavailable
    s3_enabled: bool = os.getenv("S3_ENABLED", "false").lower() == "true"
    s3_bucket_logs: str = os.getenv("S3_BUCKET_LOGS", "medical-ai-logs")
    s3_bucket_uploads: str = os.getenv("S3_BUCKET_UPLOADS", "medical-ai-uploads")
    s3_region: str = os.getenv("AWS_REGION", "us-east-1")

    # ── Rate Limiting ─────────────────────────────────────────────
    rate_limit_predict: str = os.getenv("RATE_LIMIT_PREDICT", "10/minute")
    rate_limit_default: str = os.getenv("RATE_LIMIT_DEFAULT", "30/minute")

    # ── Monitoring ────────────────────────────────────────────────
    metrics_path: str = "/metrics"
    track_latency: bool = True

    # ── Security Headers ──────────────────────────────────────────
    hsts_max_age: int = 31536000
    csp_policy: str = "default-src 'self'"

    # ── Storage ───────────────────────────────────────────────────
    temp_storage_path: str = os.getenv("TEMP_STORAGE_PATH", "/tmp/medical_ai_uploads")
    encryption_key: Optional[str] = os.getenv("ENCRYPTION_KEY")
