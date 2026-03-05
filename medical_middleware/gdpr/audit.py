"""
audit.py — GDPR-compliant local audit logger (fallback when S3 is disabled).

Every request is logged with:
  - Timestamp (UTC)
  - Request ID (UUID)
  - App name
  - Endpoint, method, status code
  - Anonymized IP (last octet zeroed)
  - Processing duration
  - Data consent status

Logs are written as JSONL (one JSON object per line).
Patient data is NEVER logged.

When S3_ENABLED=true, audit_s3.py is used instead.
This file is the local fallback.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _anonymize_ip(ip: str) -> str:
    """Zero out last octet of IPv4, last group of IPv6."""
    if not ip:
        return "unknown"
    parts = ip.split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.{parts[2]}.0"
    parts = ip.split(":")
    if len(parts) > 1:
        parts[-1] = "0000"
        return ":".join(parts)
    return "anonymized"


class AuditLogger:
    """
    Local JSONL audit logger.
    Used when S3 is disabled or unavailable.
    Matches the same interface as S3AuditLogger so they're interchangeable.
    """

    def __init__(
        self,
        log_path: str = "/tmp/logs/audit.jsonl",
        app_name: str = "medical-ai",
    ):
        self.log_path = Path(log_path)
        self.app_name = app_name
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            fallback = Path("logs/audit.jsonl")
            fallback.parent.mkdir(parents=True, exist_ok=True)
            self.log_path = fallback
            logger.warning(
                f"[audit] Permission denied — using fallback: {self.log_path}"
            )

    def log(
        self,
        request_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        client_ip: str,
        consent_given: bool,
        user_agent: Optional[str] = None,
        extra: Optional[dict] = None,
    ):
        """Write a single audit log entry to local JSONL file."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "app": self.app_name,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "client_ip": _anonymize_ip(client_ip),
            "consent_given": consent_given,
            "user_agent": user_agent[:200] if user_agent else None,
        }
        if extra:
            entry["extra"] = extra

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.error(f"[audit] Failed to write log: {e}")

    def get_logs_for_request(self, request_id: str) -> list:
        """Retrieve all log entries for a specific request ID."""
        results = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("request_id") == request_id:
                            results.append(entry)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        return results

    def erase_request(self, request_id: str) -> int:
        """
        Right to erasure — remove all log entries for a request ID.
        Returns number of entries erased.
        """
        if not self.log_path.exists():
            return 0

        kept = []
        erased = 0

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("request_id") == request_id:
                        erased += 1
                    else:
                        kept.append(line)
                except json.JSONDecodeError:
                    kept.append(line)

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.writelines(kept)

        return erased

    def generate_request_id(self) -> str:
        return str(uuid.uuid4())
