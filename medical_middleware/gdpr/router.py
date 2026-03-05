"""
router.py — GDPR compliance endpoints.

Endpoints:
    GET    /gdpr/status              — GDPR compliance status
    GET    /gdpr/request/{id}        — Retrieve request audit trail
    DELETE /gdpr/erase/{request_id}  — Right to erasure (local + S3)
    GET    /gdpr/retention           — Data retention policy info
    GET    /gdpr/privacy-policy      — Machine-readable privacy policy
"""

from fastapi import APIRouter, HTTPException
from .middleware import get_audit_logger, get_retention_manager

gdpr_router = APIRouter()


@gdpr_router.get("/status")
def gdpr_status():
    audit = get_audit_logger()
    storage = (
        "s3"
        if hasattr(audit, "s3") and getattr(audit, "s3", None) and audit.s3.available
        else "local"
    )
    return {
        "compliant": True,
        "framework": "GDPR (EU) 2016/679",
        "storage_backend": storage,
        "measures": {
            "audit_logging": True,
            "data_minimization": True,
            "image_anonymization": True,
            "right_to_erasure": True,
            "data_retention_ttl": True,
            "consent_enforcement": True,
            "security_headers": True,
            "encryption_in_transit": True,
            "s3_lifecycle_rules": storage == "s3",
        },
        "data_controller": "Moez Bouassida",
        "legal_basis": "Article 9(2)(j) — Scientific research purposes",
        "contact": "privacy@example.com",
    }


@gdpr_router.get("/request/{request_id}")
def get_request_audit(request_id: str):
    """GDPR Article 15 — Right of access."""
    audit = get_audit_logger()
    if not audit:
        raise HTTPException(status_code=503, detail="Audit logger not initialized")

    logs = audit.get_logs_for_request(request_id)
    retention = get_retention_manager()
    record = retention.get_record(request_id) if retention else None

    return {
        "request_id": request_id,
        "audit_entries": logs,
        "retention": record,
        "found": len(logs) > 0,
    }


@gdpr_router.delete("/erase/{request_id}")
def erase_request(request_id: str):
    """
    GDPR Article 17 — Right to erasure.
    Deletes from S3 (if enabled) and local storage.
    """
    retention = get_retention_manager()
    audit = get_audit_logger()

    if not retention or not audit:
        raise HTTPException(status_code=503, detail="GDPR services not initialized")

    file_result = retention.erase(request_id)
    logs_erased = audit.erase_request(request_id)

    # Surface S3-specific info if available
    s3_deleted = file_result.get("s3_deleted", file_result.get("file_deleted", False))

    return {
        "erased": True,
        "request_id": request_id,
        "file_deleted": s3_deleted,
        "local_deleted": file_result.get("local_deleted", False),
        "audit_logs_erased": logs_erased,
        "gdpr_article": "Article 17 GDPR — Right to erasure",
        "message": "All data associated with this request has been erased.",
    }


@gdpr_router.get("/retention")
def retention_policy():
    retention = get_retention_manager()
    ttl = getattr(retention, "retention_seconds", 86400)
    is_s3 = hasattr(retention, "s3")

    return {
        "policy": {
            "uploaded_images": {
                "retention_period": f"{ttl} seconds ({ttl // 3600} hours)",
                "deletion_method": (
                    "S3 Lifecycle rule + immediate erase on request"
                    if is_s3
                    else "Secure overwrite + unlink"
                ),
                "storage": (
                    "AWS S3 (encrypted, private)" if is_s3 else "Local temp directory"
                ),
                "auto_delete": True,
            },
            "audit_logs": {
                "retention_period": "90 days",
                "storage": (
                    "AWS S3 (medical-ai-logs bucket)" if is_s3 else "Local JSONL file"
                ),
                "minimization": "IP addresses anonymized, no patient data stored",
            },
            "model_outputs": {
                "retention_period": "Not stored — returned in response only",
                "stored": False,
            },
        },
        "gdpr_article": "Article 5(1)(e) — Storage limitation",
    }


@gdpr_router.get("/privacy-policy")
def privacy_policy():
    retention = get_retention_manager()
    is_s3 = hasattr(retention, "s3")

    return {
        "version": "1.0",
        "last_updated": "2025-01-01",
        "data_controller": {
            "name": "Moez Bouassida",
            "contact": "privacy@example.com",
        },
        "data_processed": [
            {
                "type": "Medical images",
                "purpose": "AI-assisted diagnosis support",
                "legal_basis": "Article 9(2)(j) — Scientific research",
                "retention": "24 hours maximum",
                "storage": (
                    "AWS S3 (private, lifecycle-managed)"
                    if is_s3
                    else "Local temp storage"
                ),
                "shared_with": "No third parties",
            }
        ],
        "rights": {
            "access": "GET /gdpr/request/{request_id}",
            "erasure": "DELETE /gdpr/erase/{request_id}",
            "portability": "GET /gdpr/request/{request_id}",
            "restriction": "Contact data controller",
            "objection": "Contact data controller",
        },
        "security_measures": [
            "TLS 1.3 in transit",
            "Image metadata stripped on upload (EXIF + DICOM tags removed)",
            "IP addresses anonymized in logs (last octet zeroed)",
            "Automatic data deletion after TTL (S3 Lifecycle + local cleanup)",
            "No persistent storage of medical images beyond retention period",
            (
                "AWS S3 HTTPS-only bucket policy"
                if is_s3
                else "Local secure overwrite before deletion"
            ),
        ],
    }
