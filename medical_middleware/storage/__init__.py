"""
storage/__init__.py

Lazy imports — boto3 is optional.
If boto3 is not installed, importing this module is safe.
S3 classes will only fail when actually instantiated.
"""

# These imports are deferred to avoid crashing if boto3 is missing.
# The actual ImportError will only surface when you try to USE them.

try:
    from .s3 import S3Client, get_s3_client
    from .audit_s3 import S3AuditLogger
    from .retention_s3 import S3RetentionManager

    __all__ = [
        "S3Client",
        "get_s3_client",
        "S3AuditLogger",
        "S3RetentionManager",
    ]

except ImportError:
    # boto3 not installed — S3 classes unavailable
    # middleware will fall back to local storage automatically
    import logging

    logging.getLogger(__name__).warning(
        "[storage] boto3 not installed — S3 storage disabled. "
        "Install with: pip install boto3"
    )

    def get_s3_client(*args, **kwargs):
        raise ImportError("boto3 is required for S3 storage. pip install boto3")

    __all__ = ["get_s3_client"]
