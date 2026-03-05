"""
audit_s3.py — S3-backed GDPR audit logger.

Writes audit logs to:
    s3://medical-ai-logs/{app_name}/audit/YYYY/MM/DD/{app_name}-audit.jsonl

Falls back to local file if S3 is unavailable.
Each log entry is one JSON line (JSONL format).
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .s3 import S3Client, get_s3_client

logger = logging.getLogger(__name__)


def _anonymize_ip(ip: str) -> str:
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


class S3AuditLogger:
    """
    GDPR-compliant audit logger backed by AWS S3.
    Also writes to local fallback file.
    Interface is identical to AuditLogger (local) — fully interchangeable.

    S3 path: s3://{bucket_logs}/{app_name}/audit/YYYY/MM/DD/{app_name}-audit.jsonl
    """

    def __init__(
        self,
        app_name: str = "medical-ai",
        s3_client: S3Client = None,
        local_fallback: str = "/tmp/audit_fallback.jsonl",
    ):
        self.app_name = app_name
        self.s3 = s3_client or get_s3_client()
        self.local_fallback = Path(local_fallback)
        self.local_fallback.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[audit_s3] Initialized — app={app_name}, "
            f"S3={'enabled' if self.s3.available else 'disabled (local fallback)'}"
        )

    def _current_s3_key(self) -> str:
        """S3 key: {app_name}/audit/YYYY/MM/DD/{app_name}-audit.jsonl"""
        now = datetime.now(timezone.utc)
        return f"{self.app_name}/audit/{now.year}/{now.month:02d}/{now.day:02d}/{self.app_name}-audit.jsonl"

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
        """Write a single audit log entry to S3 + local fallback."""
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

        line = json.dumps(entry) + "\n"

        # Always write local fallback first (fast, reliable)
        try:
            with open(self.local_fallback, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            logger.error(f"[audit_s3] Local fallback write failed: {e}")

        # Append to S3 (download existing → append → re-upload)
        if self.s3.available:
            self._append_to_s3(line)

    def _append_to_s3(self, line: str):
        key = self._current_s3_key()
        try:
            existing = self.s3.download_bytes(key) or b""
            updated = existing + line.encode("utf-8")
            self.s3.upload_bytes(
                updated,
                key,
                content_type="application/x-ndjson",
                metadata={"app": self.app_name, "type": "audit-log"},
            )
        except Exception as e:
            logger.error(f"[audit_s3] S3 append failed: {e}")

    def get_logs_for_request(self, request_id: str) -> list:
        """Retrieve all audit entries for a request ID."""
        seen_timestamps = set()
        results = []

        def _parse_lines(text: str):
            for line in text.splitlines():
                try:
                    entry = json.loads(line)
                    if entry.get("request_id") == request_id:
                        # Deduplicate by timestamp
                        ts = entry.get("timestamp", "")
                        if ts not in seen_timestamps:
                            seen_timestamps.add(ts)
                            results.append(entry)
                except json.JSONDecodeError:
                    continue

        # Try S3 first
        if self.s3.available:
            data = self.s3.download_bytes(self._current_s3_key())
            if data:
                _parse_lines(data.decode("utf-8"))

        # Also check local fallback (catches entries that failed S3 upload)
        if self.local_fallback.exists():
            with open(self.local_fallback, "r", encoding="utf-8") as f:
                _parse_lines(f.read())

        return results

    def erase_request(self, request_id: str) -> int:
        """
        Right to erasure — remove all log entries for a request ID.
        Erases from both S3 and local fallback.
        Returns total unique entries erased.
        """
        erased_ids = set()

        # Erase from S3
        if self.s3.available:
            key = self._current_s3_key()
            data = self.s3.download_bytes(key)
            if data:
                kept = []
                for line in data.decode("utf-8").splitlines():
                    try:
                        entry = json.loads(line)
                        if entry.get("request_id") == request_id:
                            erased_ids.add(entry.get("timestamp", line))
                        else:
                            kept.append(line)
                    except json.JSONDecodeError:
                        kept.append(line)

                updated = "\n".join(kept) + "\n" if kept else b""
                self.s3.upload_bytes(
                    updated.encode("utf-8") if isinstance(updated, str) else updated,
                    key,
                    content_type="application/x-ndjson",
                )

        # Erase from local fallback
        if self.local_fallback.exists():
            kept = []
            with open(self.local_fallback, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("request_id") == request_id:
                            erased_ids.add(entry.get("timestamp", line.strip()))
                        else:
                            kept.append(line)
                    except json.JSONDecodeError:
                        kept.append(line)
            with open(self.local_fallback, "w", encoding="utf-8") as f:
                f.writelines(kept)

        return len(erased_ids)

    def generate_request_id(self) -> str:
        return str(uuid.uuid4())
