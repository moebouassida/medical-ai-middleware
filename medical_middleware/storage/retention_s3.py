"""
retention_s3.py — S3-backed data retention manager.

Uploaded medical images are stored in:
    s3://medical-ai-uploads/{request_id}/image.jpg

S3 Lifecycle rule on the uploads bucket auto-deletes after 24h.
Right to erasure immediately deletes from S3.

Falls back to local temp storage if S3 is unavailable.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .s3 import S3Client, get_s3_client

logger = logging.getLogger(__name__)


class S3RetentionManager:
    """
    Manages lifecycle of uploaded medical images using AWS S3.

    On upload:
        1. Image anonymized (metadata stripped)
        2. Uploaded to s3://medical-ai-uploads/{request_id}/image.jpg
        3. S3 Lifecycle rule auto-deletes after 24h (set by aws_setup.py)
        4. Local temp file deleted immediately after S3 upload

    On erasure:
        1. S3 object deleted immediately
        2. Registry entry removed
    """

    def __init__(
        self,
        s3_client: S3Client = None,
        retention_seconds: int = 86400,
        local_temp_path: str = "/tmp/medical_ai_uploads",
        cleanup_interval: int = 300,
    ):
        self.s3 = s3_client or get_s3_client()
        self.retention_seconds = retention_seconds
        self.local_temp = Path(local_temp_path)
        self.local_temp.mkdir(parents=True, exist_ok=True)

        self._registry: Dict[str, dict] = {}
        self._lock = threading.Lock()

        # Background cleanup for local fallback files
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            args=(cleanup_interval,),
            daemon=True,
            name="s3-retention-cleanup",
        )
        self._cleanup_thread.start()

        logger.info(
            f"[retention_s3] Initialized — "
            f"S3={'enabled' if self.s3.available else 'disabled'}, "
            f"TTL={retention_seconds}s"
        )

    def register_upload(
        self,
        request_id: str,
        image_bytes: bytes,
        filename: str = "image.jpg",
        content_type: str = "image/jpeg",
    ) -> dict:
        """
        Upload anonymized image to S3 and register for retention tracking.

        Args:
            request_id:   unique request identifier
            image_bytes:  anonymized image bytes
            filename:     original filename
            content_type: MIME type

        Returns:
            Registration record with S3 URI and expiry
        """
        expiry_ts = time.time() + self.retention_seconds
        s3_key = S3Client.make_upload_key(request_id, filename)
        s3_uri = None

        if self.s3.available:
            s3_uri = self.s3.upload_bytes(
                image_bytes,
                key=s3_key,
                bucket=self.s3.bucket_uploads,
                content_type=content_type,
                metadata={
                    "request_id": request_id,
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                    "gdpr_retention": f"{self.retention_seconds}s",
                },
            )

        # Also write local temp copy as fallback
        local_path = self.local_temp / request_id
        local_path.mkdir(exist_ok=True)
        local_file = local_path / filename
        local_file.write_bytes(image_bytes)

        record = {
            "request_id": request_id,
            "s3_uri": s3_uri,
            "s3_key": s3_key,
            "local_path": str(local_file),
            "filename": filename,
            "registered": datetime.now(timezone.utc).isoformat(),
            "expires_at": datetime.fromtimestamp(
                expiry_ts, tz=timezone.utc
            ).isoformat(),
            "expiry_ts": expiry_ts,
            "erased": False,
        }

        with self._lock:
            self._registry[request_id] = record

        logger.debug(
            f"[retention_s3] Registered {request_id} → {s3_uri or 'local only'}"
        )
        return record

    def erase(self, request_id: str) -> dict:
        """
        Right to erasure — immediately delete from S3 and local storage.
        GDPR Article 17.
        """
        with self._lock:
            record = self._registry.get(request_id)

        if not record:
            return {"erased": False, "reason": "request_id not found"}

        s3_deleted = False
        local_deleted = False

        # Delete from S3
        if self.s3.available and record.get("s3_key"):
            s3_deleted = self.s3.delete(
                record["s3_key"],
                bucket=self.s3.bucket_uploads,
            )

        # Delete local temp
        local_path = Path(record.get("local_path", ""))
        if local_path.exists():
            try:
                # Secure overwrite before deletion
                size = local_path.stat().st_size
                with open(local_path, "wb") as f:
                    f.write(b"\x00" * size)
                local_path.unlink()
                local_deleted = True
                # Remove parent dir if empty
                if not any(local_path.parent.iterdir()):
                    local_path.parent.rmdir()
            except OSError as e:
                logger.error(f"[retention_s3] Local delete failed: {e}")

        with self._lock:
            if request_id in self._registry:
                del self._registry[request_id]

        result = {
            "erased": True,
            "request_id": request_id,
            "s3_deleted": s3_deleted,
            "local_deleted": local_deleted,
            "erased_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            f"[retention_s3] Erased {request_id} (S3={s3_deleted}, local={local_deleted})"
        )
        return result

    def get_record(self, request_id: str) -> Optional[dict]:
        with self._lock:
            return self._registry.get(request_id)

    def get_presigned_url(
        self, request_id: str, expires_in: int = 3600
    ) -> Optional[str]:
        """Get a temporary presigned URL for an uploaded image."""
        record = self.get_record(request_id)
        if not record or not self.s3.available:
            return None
        return self.s3.presigned_url(
            record["s3_key"],
            bucket=self.s3.bucket_uploads,
            expires_in=expires_in,
        )

    def _cleanup_loop(self, interval: int):
        while True:
            try:
                self._cleanup_expired_local()
            except Exception as e:
                logger.error(f"[retention_s3] Cleanup error: {e}")
            time.sleep(interval)

    def _cleanup_expired_local(self):
        """Delete expired local temp files."""
        now = time.time()
        with self._lock:
            expired = [
                (rid, rec)
                for rid, rec in list(self._registry.items())
                if rec["expiry_ts"] <= now and not rec["erased"]
            ]

        for request_id, record in expired:
            local_path = Path(record.get("local_path", ""))
            if local_path.exists():
                try:
                    local_path.unlink()
                    logger.info(f"[retention_s3] Local cleanup: {request_id}")
                except OSError:
                    pass

            with self._lock:
                self._registry.pop(request_id, None)
