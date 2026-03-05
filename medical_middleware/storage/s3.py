"""
s3.py — AWS S3 client wrapper for medical AI middleware.

Handles:
  - Uploading files and objects to S3
  - Downloading and deleting objects
  - Generating presigned URLs
  - Graceful fallback if boto3 not installed

All credentials come from environment variables:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_REGION          (default: us-east-1)
    S3_BUCKET_LOGS      (default: medical-ai-logs)
    S3_BUCKET_UPLOADS   (default: medical-ai-uploads)
    S3_ENABLED          (default: false)
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_s3_instance = None


class S3Client:
    """
    Thin wrapper around boto3 S3 client.
    Handles missing boto3 gracefully.
    """

    def __init__(
        self,
        bucket_logs: str = None,
        bucket_uploads: str = None,
        region: str = None,
        endpoint_url: str = None,
    ):
        self.bucket_logs = bucket_logs or os.getenv("S3_BUCKET_LOGS", "medical-ai-logs")
        self.bucket_uploads = bucket_uploads or os.getenv(
            "S3_BUCKET_UPLOADS", "medical-ai-uploads"
        )
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.endpoint_url = endpoint_url or os.getenv("S3_ENDPOINT_URL", None)
        self._client = None
        self._available = False
        self._init_client()

    def _init_client(self):
        try:
            import boto3
            
            kwargs = {"region_name": self.region}
            if self.endpoint_url:
                kwargs["endpoint_url"] = self.endpoint_url

            self._client = boto3.client("s3", **kwargs)

            # Verify credentials
            self._client.list_buckets()
            self._available = True
            logger.info(
                f"[s3] Connected — logs: {self.bucket_logs}, uploads: {self.bucket_uploads}"
            )

        except ImportError:
            logger.warning("[s3] boto3 not installed — S3 disabled. pip install boto3")
        except Exception as e:
            logger.warning(
                f"[s3] Could not connect to S3: {e} — falling back to local storage"
            )

    @property
    def available(self) -> bool:
        return self._available

    def upload_bytes(
        self,
        data: bytes,
        key: str,
        bucket: str = None,
        content_type: str = "application/octet-stream",
        metadata: dict = None,
    ) -> Optional[str]:
        """
        Upload bytes to S3.

        Args:
            data:         raw bytes to upload
            key:          S3 object key (path within bucket)
            bucket:       bucket name (default: bucket_logs)
            content_type: MIME type
            metadata:     optional key-value metadata

        Returns:
            S3 URI (s3://bucket/key) or None on failure
        """
        if not self._available:
            return None

        bucket = bucket or self.bucket_logs
        try:
            kwargs = {
                "Body": data,
                "Bucket": bucket,
                "Key": key,
                "ContentType": content_type,
            }
            if metadata:
                kwargs["Metadata"] = {k: str(v) for k, v in metadata.items()}

            self._client.put_object(**kwargs)
            uri = f"s3://{bucket}/{key}"
            logger.debug(f"[s3] Uploaded → {uri}")
            return uri

        except Exception as e:
            logger.error(f"[s3] Upload failed ({bucket}/{key}): {e}")
            return None

    def upload_file(
        self,
        file_path: str,
        key: str,
        bucket: str = None,
        content_type: str = "application/octet-stream",
    ) -> Optional[str]:
        """Upload a local file to S3."""
        if not self._available:
            return None

        bucket = bucket or self.bucket_logs
        try:
            self._client.upload_file(
                file_path,
                bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
            return f"s3://{bucket}/{key}"
        except Exception as e:
            logger.error(f"[s3] File upload failed ({bucket}/{key}): {e}")
            return None

    def download_bytes(self, key: str, bucket: str = None) -> Optional[bytes]:
        """Download object from S3 as bytes."""
        if not self._available:
            return None

        bucket = bucket or self.bucket_logs
        try:
            resp = self._client.get_object(Bucket=bucket, Key=key)
            return resp["Body"].read()
        except Exception as e:
            logger.error(f"[s3] Download failed ({bucket}/{key}): {e}")
            return None

    def delete(self, key: str, bucket: str = None) -> bool:
        """Delete object from S3."""
        if not self._available:
            return False

        bucket = bucket or self.bucket_logs
        try:
            self._client.delete_object(Bucket=bucket, Key=key)
            logger.debug(f"[s3] Deleted s3://{bucket}/{key}")
            return True
        except Exception as e:
            logger.error(f"[s3] Delete failed ({bucket}/{key}): {e}")
            return False

    def delete_prefix(self, prefix: str, bucket: str = None) -> int:
        """Delete all objects with a given prefix. Returns count deleted."""
        if not self._available:
            return 0

        bucket = bucket or self.bucket_logs
        deleted = 0
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                objects = page.get("Contents", [])
                if objects:
                    self._client.delete_objects(
                        Bucket=bucket,
                        Delete={"Objects": [{"Key": o["Key"]} for o in objects]},
                    )
                    deleted += len(objects)
        except Exception as e:
            logger.error(f"[s3] delete_prefix failed ({bucket}/{prefix}): {e}")

        return deleted

    def list_keys(self, prefix: str = "", bucket: str = None) -> list:
        """List all object keys with a given prefix."""
        if not self._available:
            return []

        bucket = bucket or self.bucket_logs
        keys = []
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                keys.extend(o["Key"] for o in page.get("Contents", []))
        except Exception as e:
            logger.error(f"[s3] list_keys failed ({bucket}/{prefix}): {e}")

        return keys

    def presigned_url(
        self,
        key: str,
        bucket: str = None,
        expires_in: int = 3600,
    ) -> Optional[str]:
        """
        Generate a presigned URL for temporary access.
        Default expiry: 1 hour.
        """
        if not self._available:
            return None

        bucket = bucket or self.bucket_logs
        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires_in,
            )
        except Exception as e:
            logger.error(f"[s3] presigned_url failed ({bucket}/{key}): {e}")
            return None

    def ensure_bucket(self, bucket: str, lifecycle_days: int = None):
        """
        Create bucket if it doesn't exist.
        Optionally set lifecycle rule to auto-delete after N days.
        """
        if not self._available:
            return

        try:
            self._client.head_bucket(Bucket=bucket)
            logger.debug(f"[s3] Bucket exists: {bucket}")
        except Exception:
            try:
                if self.region == "us-east-1":
                    self._client.create_bucket(Bucket=bucket)
                else:
                    self._client.create_bucket(
                        Bucket=bucket,
                        CreateBucketConfiguration={"LocationConstraint": self.region},
                    )
                logger.info(f"[s3] Created bucket: {bucket}")
            except Exception as e:
                logger.error(f"[s3] Failed to create bucket {bucket}: {e}")
                return

        if lifecycle_days:
            try:
                self._client.put_bucket_lifecycle_configuration(
                    Bucket=bucket,
                    LifecycleConfiguration={
                        "Rules": [
                            {
                                "ID": f"auto-delete-after-{lifecycle_days}d",
                                "Status": "Enabled",
                                "Filter": {"Prefix": ""},
                                "Expiration": {"Days": lifecycle_days},
                            }
                        ]
                    },
                )
                logger.info(
                    f"[s3] Lifecycle rule set: {bucket} → delete after {lifecycle_days}d"
                )
            except Exception as e:
                logger.error(f"[s3] Failed to set lifecycle on {bucket}: {e}")

    @staticmethod
    def make_audit_key(app_name: str = "medical-ai") -> str:
        """Generate dated S3 key for audit logs."""
        now = datetime.now(timezone.utc)
        return f"audit/{now.year}/{now.month:02d}/{now.day:02d}/{app_name}-audit.jsonl"

    @staticmethod
    def make_xai_key(request_id: str) -> str:
        """Generate S3 key for XAI heatmap."""
        now = datetime.now(timezone.utc)
        return f"xai/{now.year}/{now.month:02d}/{now.day:02d}/{request_id}.png"

    @staticmethod
    def make_upload_key(request_id: str, filename: str = "image.jpg") -> str:
        """Generate S3 key for uploaded medical image."""
        return f"{request_id}/{filename}"


def get_s3_client() -> S3Client:
    """Get or create the global S3 client singleton."""
    global _s3_instance
    if _s3_instance is None:
        _s3_instance = S3Client()
    return _s3_instance
