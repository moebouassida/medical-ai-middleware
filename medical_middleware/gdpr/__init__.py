from .middleware import GDPRMiddleware
from .router import gdpr_router
from .audit import AuditLogger
from .anonymizer import ImageAnonymizer
from .retention import DataRetentionManager

__all__ = [
    "GDPRMiddleware",
    "gdpr_router",
    "AuditLogger",
    "ImageAnonymizer",
    "DataRetentionManager",
]


# S3-backed versions — imported lazily to avoid requiring boto3
def get_s3_audit_logger(*args, **kwargs):
    from ..storage.audit_s3 import S3AuditLogger

    return S3AuditLogger(*args, **kwargs)


def get_s3_retention_manager(*args, **kwargs):
    from ..storage.retention_s3 import S3RetentionManager

    return S3RetentionManager(*args, **kwargs)
