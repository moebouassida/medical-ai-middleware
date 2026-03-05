"""
retention.py — GDPR data retention enforcement.

GDPR Article 5(1)(e): Data must not be kept longer than necessary.

This module:
  - Tracks uploaded files with TTL
  - Auto-deletes files after retention period
  - Provides right-to-erasure endpoint support
  - Maintains an in-memory registry (production: use Redis)
"""

import hashlib
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DataRetentionManager:
    """
    Manages lifecycle of uploaded medical images.

    Files are:
    1. Registered on upload with TTL
    2. Auto-deleted by background thread after TTL expires
    3. Immediately deletable via erase(request_id)

    Thread-safe.
    """

    def __init__(
        self,
        retention_seconds: int = 86400,  # 24 hours
        storage_path: str = "/tmp/medical_ai_uploads",
        cleanup_interval: int = 300,  # check every 5 min
    ):
        self.retention_seconds = retention_seconds
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._registry: Dict[str, dict] = {}
        self._lock = threading.Lock()

        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            args=(cleanup_interval,),
            daemon=True,
            name="data-retention-cleanup",
        )
        self._cleanup_thread.start()
        logger.info(
            f"[retention] Started — TTL={retention_seconds}s, "
            f"path={storage_path}, cleanup_interval={cleanup_interval}s"
        )

    def register(self, request_id: str, file_path: str) -> dict:
        """
        Register a file for retention tracking.

        Args:
            request_id: unique request identifier
            file_path:  absolute path to the uploaded file

        Returns:
            Registration record with expiry time
        """
        expiry = time.time() + self.retention_seconds
        record = {
            "request_id": request_id,
            "file_path": file_path,
            "registered": datetime.now(timezone.utc).isoformat(),
            "expires_at": datetime.fromtimestamp(expiry, tz=timezone.utc).isoformat(),
            "expiry_ts": expiry,
            "erased": False,
        }

        with self._lock:
            self._registry[request_id] = record

        logger.debug(
            f"[retention] Registered {request_id} → expires {record['expires_at']}"
        )
        return record

    def erase(self, request_id: str) -> dict:
        """
        Right to erasure — immediately delete file and registry entry.

        Args:
            request_id: request ID to erase

        Returns:
            Erasure confirmation record
        """
        with self._lock:
            record = self._registry.get(request_id)

            if not record:
                return {"erased": False, "reason": "request_id not found"}

            file_path = Path(record["file_path"])
            deleted_file = False

            if file_path.exists():
                try:
                    # Overwrite with zeros before deletion (secure erase)
                    size = file_path.stat().st_size
                    with open(file_path, "wb") as f:
                        f.write(b"\x00" * size)
                    file_path.unlink()
                    deleted_file = True
                except OSError as e:
                    logger.error(f"[retention] Failed to delete {file_path}: {e}")

            record["erased"] = True
            record["erased_at"] = datetime.now(timezone.utc).isoformat()
            del self._registry[request_id]

        logger.info(f"[retention] Erased {request_id} (file_deleted={deleted_file})")
        return {
            "erased": True,
            "request_id": request_id,
            "file_deleted": deleted_file,
            "erased_at": record["erased_at"],
        }

    def get_record(self, request_id: str) -> Optional[dict]:
        """Get retention record for a request ID."""
        with self._lock:
            return self._registry.get(request_id)

    def list_active(self) -> list:
        """List all active (non-expired) registrations."""
        now = time.time()
        with self._lock:
            return [r for r in self._registry.values() if r["expiry_ts"] > now]

    def _cleanup_loop(self, interval: int):
        """Background thread — deletes expired files."""
        while True:
            try:
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"[retention] Cleanup error: {e}")
            time.sleep(interval)

    def _cleanup_expired(self):
        """Delete all expired files."""
        now = time.time()
        expired = []

        with self._lock:
            for request_id, record in list(self._registry.items()):
                if record["expiry_ts"] <= now and not record["erased"]:
                    expired.append((request_id, record))

        for request_id, record in expired:
            file_path = Path(record["file_path"])
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"[retention] Auto-deleted expired file: {request_id}")
                except OSError as e:
                    logger.error(f"[retention] Failed to auto-delete {request_id}: {e}")

            with self._lock:
                if request_id in self._registry:
                    del self._registry[request_id]

    @staticmethod
    def hash_file(file_path: str) -> str:
        """SHA256 hash of file contents (for integrity verification)."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
