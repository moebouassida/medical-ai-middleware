# 🏥 medical-ai-middleware

> Production-grade GDPR compliance, Prometheus monitoring, and API rate limiting for medical AI APIs.  
> One import. Three layers. Fully production ready.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![GDPR](https://img.shields.io/badge/GDPR-compliant-brightgreen)
![Prometheus](https://img.shields.io/badge/Prometheus-ready-orange)

---

## ⚡ Quick Start

```python
from fastapi import FastAPI
from medical_middleware import setup_middleware

app = FastAPI()
setup_middleware(app)  # ← GDPR + Prometheus + Rate Limiting. Done.
```

That's it. Your API is now:
- ✅ GDPR compliant (audit logs, right to erasure, consent enforcement)
- ✅ Monitored (Prometheus metrics + Grafana dashboard)
- ✅ Rate limited (per-IP sliding window)

---

## 📦 Installation

```bash
# Basic
pip install git+https://github.com/moebouassida/medical-ai-middleware.git

# With all optional dependencies
pip install "medical-ai-middleware[all] @ git+https://github.com/moebouassida/medical-ai-middleware.git"
```

Add to `requirements.txt`:
```
medical-ai-middleware[all] @ git+https://github.com/moebouassida/medical-ai-middleware.git
```

---

## 🔒 GDPR Layer

### What it does

| Feature | GDPR Article | Implementation |
|---|---|---|
| Consent enforcement | Art. 9 | `X-Data-Consent: true` header required |
| Audit logging | Art. 5(2) | Every request logged to JSONL |
| IP anonymization | Art. 4(1) | Last octet zeroed (e.g. `192.168.1.0`) |
| Image anonymization | Art. 4(1) | EXIF/DICOM metadata stripped |
| Data retention | Art. 5(1)(e) | Files auto-deleted after 24h |
| Right to erasure | Art. 17 | `DELETE /gdpr/erase/{request_id}` |
| Security headers | Art. 32 | HSTS, CSP, X-Content-Type, etc. |

### Endpoints

```
GET    /gdpr/status              → compliance status overview
GET    /gdpr/request/{id}        → audit trail for a request
DELETE /gdpr/erase/{request_id}  → right to erasure
GET    /gdpr/retention           → data retention policy
GET    /gdpr/privacy-policy      → machine-readable privacy policy
```

### Usage

Every predict request must include:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-Data-Consent: true" \
  -F "file=@image.jpg"
```

Without the consent header → `403 Forbidden`:
```json
{
  "error": "consent_required",
  "message": "Processing medical images requires explicit consent.",
  "gdpr_article": "Article 9 GDPR — Special categories of personal data"
}
```

Right to erasure:
```bash
curl -X DELETE "http://localhost:8000/gdpr/erase/YOUR-REQUEST-ID"
```

---

## 📊 Prometheus + Grafana

### Metrics exposed at `GET /metrics`

| Metric | Type | Description |
|---|---|---|
| `http_requests_total` | Counter | Request count by endpoint/method/status |
| `http_request_duration_seconds` | Histogram | Latency by endpoint |
| `http_requests_in_progress` | Gauge | Active requests |
| `inference_duration_seconds` | Histogram | Model inference time |
| `inference_requests_total` | Counter | Inference count by model |
| `errors_total` | Counter | Errors by type/endpoint |

### Track inference time in your code

```python
from medical_middleware.monitoring.metrics import get_metrics
import time

metrics = get_metrics()

start = time.time()
result = model.predict(image)
metrics.record_inference(
    endpoint="/predict",
    model="swinunetr-brats2021",
    duration=time.time() - start,
    success=True,
)
```

### Grafana Dashboard

Import the ready-made dashboard:
```bash
curl http://localhost:8000/grafana/dashboard > dashboard.json
# Grafana → Dashboards → Import → Upload JSON
```

### Prometheus config (`prometheus.yml`)

```yaml
scrape_configs:
  - job_name: medical-ai-api
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: /metrics
    scrape_interval: 15s
```

---

## 🛡️ Rate Limiting

Default limits:
```
/predict*  → 10 requests/minute  (inference is expensive)
/health    → 60 requests/minute
default    → 30 requests/minute
```

Per-endpoint override:
```python
from medical_middleware.ratelimit import rate_limit

@app.post("/predict")
@rate_limit("10/minute")
async def predict(request: Request):
    ...
```

Rate limit exceeded → `429 Too Many Requests`:
```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Please respect the rate limits.",
  "retry_after_seconds": 60
}
```

---

## ⚙️ Configuration

All settings configurable via env vars or `MiddlewareConfig`:

```python
from medical_middleware import setup_middleware
from medical_middleware.config import MiddlewareConfig

cfg = MiddlewareConfig(
    data_retention_seconds=3600,     # 1 hour (default: 24h)
    require_consent_header=True,
    rate_limit_predict="5/minute",
    app_name="my-medical-api",
    audit_log_path="/var/log/my_api/audit.jsonl",
)

setup_middleware(app, cfg)
```

Or via environment variables:
```bash
DATA_RETENTION_SECONDS=3600
REQUIRE_CONSENT=true
RATE_LIMIT_PREDICT=10/minute
RATE_LIMIT_DEFAULT=30/minute
APP_NAME=medical-ai-api
AUDIT_LOG_PATH=/var/log/medical_ai/audit.jsonl
ANONYMIZE_IMAGES=true
```

---

## 🧪 Testing

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## 📦 Used In

- [SwinUNETR 3D Brain Tumor Segmentation](https://github.com/moebouassida/SwinUNETR-3D-Brain-Segmentation)
- [Path-VQA Med-GaMMa Fine-Tuning](https://github.com/moebouassida/Path-VQA-Med-GaMMa-Fine-Tuning)
- [Breast Cancer Segmentation](https://github.com/moebouassida/Breast-Cancer-Segmentation)

---

## 📜 License

MIT License — free to use in research and commercial projects.

---

## 🙋 Author

**Moez Bouassida** — AI/ML Engineer · Medical Imaging  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/moezbouassida/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/moebouassida)
