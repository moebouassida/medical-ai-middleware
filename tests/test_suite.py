"""
Unit tests for medical-ai-middleware.
Run: pytest tests/ -v
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient


# ── GDPR Audit Logger ─────────────────────────────────────────────────────────
class TestAuditLogger:
    def test_log_and_retrieve(self, tmp_path):
        from medical_middleware.gdpr.audit import AuditLogger

        logger = AuditLogger(str(tmp_path / "audit.jsonl"))
        logger.log(
            request_id="test-123",
            endpoint="/predict",
            method="POST",
            status_code=200,
            duration_ms=150.0,
            client_ip="192.168.1.100",
            consent_given=True,
        )

        logs = logger.get_logs_for_request("test-123")
        assert len(logs) == 1
        assert logs[0]["request_id"] == "test-123"
        assert logs[0]["endpoint"] == "/predict"
        assert logs[0]["status_code"] == 200

    def test_ip_anonymization(self, tmp_path):
        from medical_middleware.gdpr.audit import _anonymize_ip

        assert _anonymize_ip("192.168.1.100") == "192.168.1.0"
        assert _anonymize_ip("10.0.0.50") == "10.0.0.0"
        assert _anonymize_ip("") == "unknown"

    def test_erase_request(self, tmp_path):
        from medical_middleware.gdpr.audit import AuditLogger

        logger = AuditLogger(str(tmp_path / "audit.jsonl"))
        logger.log("erase-me", "/predict", "POST", 200, 100.0, "1.2.3.4", True)
        logger.log("keep-me", "/predict", "POST", 200, 100.0, "1.2.3.4", True)

        erased = logger.erase_request("erase-me")
        assert erased == 1

        assert len(logger.get_logs_for_request("erase-me")) == 0
        assert len(logger.get_logs_for_request("keep-me")) == 1

    def test_patient_data_not_logged(self, tmp_path):
        import json
        from medical_middleware.gdpr.audit import AuditLogger

        logger = AuditLogger(str(tmp_path / "audit.jsonl"))
        logger.log("req-1", "/predict", "POST", 200, 100.0, "1.2.3.4", True)

        with open(tmp_path / "audit.jsonl") as f:
            entry = json.loads(f.read())

        # No patient data fields
        for field in ["image", "prediction", "patient", "name", "dob"]:
            assert field not in entry


# ── Image Anonymizer ──────────────────────────────────────────────────────────
class TestImageAnonymizer:
    def test_anonymize_pil(self):
        from PIL import Image
        from medical_middleware.gdpr.anonymizer import ImageAnonymizer

        # Create image with fake EXIF
        img = Image.new("RGB", (64, 64), color=(128, 64, 32))
        clean = ImageAnonymizer.anonymize_pil(img)

        assert clean.size == img.size
        assert clean.mode in ("RGB", "L")
        # No EXIF data
        assert not getattr(clean, "_getexif", lambda: None)()

    def test_anonymize_bytes(self):
        import io
        from PIL import Image
        from medical_middleware.gdpr.anonymizer import ImageAnonymizer

        img = Image.new("RGB", (32, 32))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        data = buf.getvalue()

        clean = ImageAnonymizer.anonymize_bytes(data, "image/jpeg")
        assert len(clean) > 0

        # Re-open cleaned image
        restored = Image.open(io.BytesIO(clean))
        assert restored.size == (32, 32)


# ── Data Retention ────────────────────────────────────────────────────────────
class TestDataRetention:
    def test_register_and_retrieve(self, tmp_path):
        from medical_middleware.gdpr.retention import DataRetentionManager

        mgr = DataRetentionManager(
            retention_seconds=3600,
            storage_path=str(tmp_path),
            cleanup_interval=999999,
        )

        # Create a fake file
        fake_file = tmp_path / "upload_123.jpg"
        fake_file.write_bytes(b"fake image data")

        record = mgr.register("req-123", str(fake_file))
        assert record["request_id"] == "req-123"
        assert "expires_at" in record

        retrieved = mgr.get_record("req-123")
        assert retrieved is not None

    def test_erase(self, tmp_path):
        from medical_middleware.gdpr.retention import DataRetentionManager

        mgr = DataRetentionManager(
            retention_seconds=3600,
            storage_path=str(tmp_path),
            cleanup_interval=999999,
        )

        fake_file = tmp_path / "upload_456.jpg"
        fake_file.write_bytes(b"sensitive medical data")
        mgr.register("req-456", str(fake_file))

        result = mgr.erase("req-456")
        assert result["erased"] is True
        assert result["file_deleted"] is True
        assert not fake_file.exists()
        assert mgr.get_record("req-456") is None

    def test_erase_nonexistent(self, tmp_path):
        from medical_middleware.gdpr.retention import DataRetentionManager

        mgr = DataRetentionManager(storage_path=str(tmp_path), cleanup_interval=999999)
        result = mgr.erase("nonexistent-id")
        assert result["erased"] is False


# ── Middleware Integration ─────────────────────────────────────────────────────
class TestGDPRMiddleware:
    def _make_app(self, tmp_path):
        from medical_middleware import setup_gdpr
        from medical_middleware.config import MiddlewareConfig

        app = FastAPI()

        @app.post("/predict")
        def predict():
            return {"result": "ok"}

        @app.get("/health")
        def health():
            return {"status": "ok"}

        cfg = MiddlewareConfig(
            audit_log_path=str(tmp_path / "audit.jsonl"),
            temp_storage_path=str(tmp_path / "uploads"),
            require_consent_header=True,
        )
        setup_gdpr(app, cfg)
        return app

    def test_consent_required(self, tmp_path):
        app = self._make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        # Without consent header
        resp = client.post("/predict")
        assert resp.status_code == 403
        assert resp.json()["error"] == "consent_required"

    def test_consent_accepted(self, tmp_path):
        app = self._make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/predict", headers={"X-Data-Consent": "true"})
        assert resp.status_code == 200

    def test_request_id_header(self, tmp_path):
        app = self._make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/health")
        assert "x-request-id" in resp.headers

    def test_security_headers(self, tmp_path):
        app = self._make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/health")
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "DENY"
        assert "cache-control" in resp.headers

    def test_gdpr_status_endpoint(self, tmp_path):
        app = self._make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/gdpr/status")
        assert resp.status_code == 200
        assert resp.json()["compliant"] is True

    def test_right_to_erasure_endpoint(self, tmp_path):
        app = self._make_app(tmp_path)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.delete("/gdpr/erase/nonexistent-request-id")
        assert resp.status_code == 200
        data = resp.json()
        assert "erased" in data


# ── Config ────────────────────────────────────────────────────────────────────
class TestConfig:
    def test_defaults(self):
        from medical_middleware.config import MiddlewareConfig

        cfg = MiddlewareConfig()
        assert cfg.data_retention_seconds == 86400
        assert cfg.rate_limit_predict == "10/minute"
        assert cfg.anonymize_images is True
        assert cfg.require_consent_header is True

    def test_custom_values(self):
        from medical_middleware.config import MiddlewareConfig

        cfg = MiddlewareConfig(
            data_retention_seconds=3600,
            rate_limit_predict="5/minute",
        )
        assert cfg.data_retention_seconds == 3600
        assert cfg.rate_limit_predict == "5/minute"
