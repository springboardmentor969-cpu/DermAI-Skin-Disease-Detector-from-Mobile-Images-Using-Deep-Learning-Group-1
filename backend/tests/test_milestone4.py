from __future__ import annotations

import io
import uuid
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def test_system_health_endpoint():
    # Test both bare /health and API-prefixed /api/v1/health
    resp1 = client.get("/health")
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["status"] in ("ok", "healthy", "degraded")
    assert data1["database"] == "connected"
    assert "models" in data1
    assert "mobilenet_v2" in data1["models"]

    resp2 = client.get("/api/v1/health")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["database"] == "connected"


def test_milestone4_report_exports_and_security():
    uid = uuid.uuid4().hex[:8]
    p1_email = f"patient1_{uid}@example.com"
    p2_email = f"patient2_{uid}@example.com"
    doc_email = f"doctor_{uid}@example.com"
    pwd = "SecurePassword123!"

    # 1. Register accounts
    r1 = client.post("/api/v1/auth/register", json={"email": p1_email, "password": pwd, "full_name": "Varun Test", "role": "patient"})
    assert r1.status_code == 201, r1.text
    r2 = client.post("/api/v1/auth/register", json={"email": p2_email, "password": pwd, "full_name": "Other Patient", "role": "patient"})
    assert r2.status_code == 201, r2.text
    rd = client.post("/api/v1/auth/register", json={"email": doc_email, "password": pwd, "full_name": "Dr. Arjun Sharma", "role": "doctor"})
    assert rd.status_code == 201, rd.text

    p1_token = client.post("/api/v1/auth/login", json={"email": p1_email, "password": pwd}).json()["access_token"]
    p2_token = client.post("/api/v1/auth/login", json={"email": p2_email, "password": pwd}).json()["access_token"]
    doc_token = client.post("/api/v1/auth/login", json={"email": doc_email, "password": pwd}).json()["access_token"]

    p1_hdr = {"Authorization": f"Bearer {p1_token}"}
    p2_hdr = {"Authorization": f"Bearer {p2_token}"}
    doc_hdr = {"Authorization": f"Bearer {doc_token}"}

    # 2. Patient 1 creates a case
    c_resp = client.post("/api/v1/cases", headers=p1_hdr, json={"title": "Pigmented Nevus Evaluation", "description": "Border assessment on upper shoulder"})
    assert c_resp.status_code == 201
    case_id = c_resp.json()["id"]

    # 3. Patient 1 uploads an image
    img = Image.new("RGB", (224, 224), color=(190, 130, 95))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    up_resp = client.post(
        f"/api/v1/upload/{case_id}",
        headers=p1_hdr,
        data={"architecture": "mobilenet_v2"},
        files={"file": ("lesion_test.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert up_resp.status_code == 201
    image_id = up_resp.json()["id"]

    # 4. Export JSON report
    json_resp = client.get(f"/api/v1/reports/cases/{case_id}?format=json", headers=p1_hdr)
    assert json_resp.status_code == 200
    report_json = json_resp.json()
    assert "report_id" in report_json
    assert report_json["case"]["id"] == case_id
    assert report_json["patient"]["name"] == "Varun Test"
    assert len(report_json["images"]) >= 1
    assert "disclaimer" in report_json

    # 5. Export CSV report
    csv_resp = client.get(f"/api/v1/reports/cases/{case_id}?format=csv", headers=p1_hdr)
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers.get("content-type", "")
    csv_text = csv_resp.text
    assert "Report ID" in csv_text
    assert "Case ID" in csv_text
    assert str(case_id) in csv_text

    # 6. Export PDF report
    pdf_resp = client.get(f"/api/v1/reports/cases/{case_id}?format=pdf", headers=p1_hdr)
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers.get("content-type") == "application/pdf"
    assert pdf_resp.content.startswith(b"%PDF-"), "Generated report must be a valid binary PDF"
    assert len(pdf_resp.content) > 1000

    # 7. Single image export
    img_pdf_resp = client.get(f"/api/v1/reports/images/{image_id}?format=pdf", headers=p1_hdr)
    assert img_pdf_resp.status_code == 200
    assert img_pdf_resp.content.startswith(b"%PDF-")

    # 8. Security & RBAC: Patient 2 is blocked from accessing Patient 1's report
    p2_attempt = client.get(f"/api/v1/reports/cases/{case_id}?format=pdf", headers=p2_hdr)
    assert p2_attempt.status_code in (403, 404), "Unauthorized patient must be denied access to another user's report"

    # 9. Doctor RBAC: Doctor can access patient's report for clinical consultation
    doc_report = client.get(f"/api/v1/reports/cases/{case_id}?format=pdf", headers=doc_hdr)
    assert doc_report.status_code == 200
    assert doc_report.content.startswith(b"%PDF-")
