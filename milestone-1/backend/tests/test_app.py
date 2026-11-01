from __future__ import annotations

import os
import tempfile
import uuid
from datetime import timedelta
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

test_database = Path(tempfile.gettempdir()) / f"dermai_test_{uuid.uuid4().hex}.db"
# API tests use an isolated SQLite database unless an explicit dedicated test database is provided.
os.environ["DATABASE_URL"] = os.getenv(
    "DERMAI_TEST_DATABASE_URL",
    f"sqlite:///{test_database.as_posix()}",
)
os.environ["JWT_SECRET_KEY"] = "test-only-secret-key-with-more-than-32-characters"
os.environ["UPLOAD_DIR"] = str(Path(__file__).resolve().parent / "test-uploads")

from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

client = TestClient(app)


def make_email() -> str:
    return f"user-{uuid.uuid4().hex[:8]}@example.com"


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_register_and_login() -> None:
    email = make_email()
    payload = {"email": email, "password": "strongpass123", "full_name": "User Name"}
    register_response = client.post("/api/v1/auth/register", json=payload)
    assert register_response.status_code == 201

    login_response = client.post("/api/v1/auth/login", json={"email": payload["email"], "password": payload["password"]})
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]
    assert token

    protected = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert protected.status_code == 200
    assert protected.json()["email"] == payload["email"]


def test_registration_rejects_passwords_over_bcrypt_byte_limit() -> None:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": make_email(),
            "password": "é" * 37,
            "full_name": "Long Password",
        },
    )
    assert response.status_code == 422


def test_case_and_upload_flow(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)
    from app.services.preprocessing_service import ImagePreprocessor

    original_preprocess = ImagePreprocessor.preprocess
    preprocessing_calls: list[bytes] = []

    def track_preprocessing(self, file_bytes: bytes):
        preprocessing_calls.append(file_bytes)
        return original_preprocess(self, file_bytes)

    monkeypatch.setattr(ImagePreprocessor, "preprocess", track_preprocessing)
    email = make_email()
    register = client.post("/api/v1/auth/register", json={"email": email, "password": "strongpass123", "full_name": "Case User"})
    assert register.status_code == 201
    auth_response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "strongpass123"},
    )
    assert auth_response.status_code == 200
    token = auth_response.json()["access_token"]

    create_case = client.post(
        "/api/v1/cases",
        json={"title": "Test case", "description": "A single upload test"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert create_case.status_code == 201
    case_id = create_case.json()["id"]

    detail_response = client.get(
        f"/api/v1/cases/{case_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert detail_response.status_code == 200
    assert detail_response.json()["title"] == "Test case"

    file_path = Path(__file__).resolve().parent.parent.parent / "HAM10000_images_part_1" / "ISIC_0024306.jpg"
    assert file_path.exists(), file_path

    with open(file_path, "rb") as image_file:
        upload_response = client.post(
            f"/api/v1/upload/{case_id}",
            files={"file": ("../../outside.jpg", image_file.read(), "image/jpeg")},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert upload_response.status_code == 201, upload_response.text
    uploaded = upload_response.json()
    assert uploaded["filename"] == "outside.jpg"
    assert uploaded["file_size"] == file_path.stat().st_size
    assert uploaded["status"] == "processed"
    assert uploaded["preprocessing"]["target_size"] == [224, 224]
    assert uploaded["preprocessing"]["channels"] == "RGB"
    assert uploaded["preprocessing"]["normalization"] == "scale_0_1"
    assert uploaded["preprocessing"]["pixel_value_range"][0] >= 0.0
    assert uploaded["preprocessing"]["pixel_value_range"][1] <= 1.0
    assert preprocessing_calls == [file_path.read_bytes()]
    assert "storage_path" not in uploaded
    stored_files = list(tmp_path.iterdir())
    assert len(stored_files) == 1
    assert stored_files[0].parent == tmp_path
    assert stored_files[0].name.endswith(".jpg")

    list_response = client.get("/api/v1/cases", headers={"Authorization": f"Bearer {token}"})
    assert list_response.status_code == 200
    assert len(list_response.json()) >= 1


def test_upload_records_preprocessing_failure_and_preserves_original(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)
    from app.services.preprocessing_service import ImagePreprocessor

    def fail_preprocessing(self, file_bytes: bytes):
        raise ValueError("test preprocessing failure")

    monkeypatch.setattr(ImagePreprocessor, "preprocess", fail_preprocessing)
    email = make_email()
    assert client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "strongpass123", "full_name": "Failure User"},
    ).status_code == 201
    token = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "strongpass123"},
    ).json()["access_token"]
    case_id = client.post(
        "/api/v1/cases",
        json={"title": "Preprocessing failure"},
        headers={"Authorization": f"Bearer {token}"},
    ).json()["id"]
    original_path = (
        Path(__file__).resolve().parent.parent.parent
        / "HAM10000_images_part_1"
        / "ISIC_0024306.jpg"
    )

    response = client.post(
        f"/api/v1/upload/{case_id}",
        files={"file": ("original.jpg", original_path.read_bytes(), "image/jpeg")},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 201
    uploaded = response.json()
    assert uploaded["status"] == "processing_failed"
    assert uploaded["preprocessing"] is None
    assert "original upload was preserved" in uploaded["processing_error"]
    stored_files = list(tmp_path.glob("*.jpg"))
    assert len(stored_files) == 1
    assert stored_files[0].read_bytes() == original_path.read_bytes()


def test_upload_rejects_invalid_and_oversized_images(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)
    email = make_email()
    client.post("/api/v1/auth/register", json={"email": email, "password": "strongpass123", "full_name": "Upload User"})
    token = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "strongpass123"},
    ).json()["access_token"]
    case_id = client.post(
        "/api/v1/cases",
        json={"title": "Upload validation"},
        headers={"Authorization": f"Bearer {token}"},
    ).json()["id"]

    invalid = client.post(
        f"/api/v1/upload/{case_id}",
        files={"file": ("broken.jpg", b"not an image", "image/jpeg")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert invalid.status_code == 400
    assert "Corrupted" in invalid.json()["detail"]
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE", 16)
    too_large = client.post(
        f"/api/v1/upload/{case_id}",
        files={"file": ("large.jpg", b"x" * 17, "image/jpeg")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert too_large.status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_upload_rejects_mime_mismatch_and_oversized_dimensions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)
    email = make_email()
    client.post("/api/v1/auth/register", json={"email": email, "password": "strongpass123", "full_name": "Format User"})
    token = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "strongpass123"},
    ).json()["access_token"]
    case_id = client.post(
        "/api/v1/cases",
        json={"title": "Format validation"},
        headers={"Authorization": f"Bearer {token}"},
    ).json()["id"]

    jpeg_path = Path(__file__).resolve().parent.parent.parent / "HAM10000_images_part_1" / "ISIC_0024306.jpg"
    mismatch = client.post(
        f"/api/v1/upload/{case_id}",
        files={"file": ("image.png", jpeg_path.read_bytes(), "image/png")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert mismatch.status_code == 400
    assert "does not match" in mismatch.json()["detail"]

    png_buffer = BytesIO()
    Image.new("RGB", (33, 16), color=(30, 120, 90)).save(png_buffer, format="PNG")
    monkeypatch.setattr(settings, "MAX_IMAGE_DIMENSION", 32)
    oversized = client.post(
        f"/api/v1/upload/{case_id}",
        files={"file": ("wide.png", png_buffer.getvalue(), "image/png")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert oversized.status_code == 400
    assert "dimensions" in oversized.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_upload_removes_file_when_database_commit_fails(tmp_path: Path, monkeypatch) -> None:
    from app.services.image_service import save_upload

    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path)

    class FailingSession:
        def add(self, record) -> None:
            self.record = record

        def flush(self) -> None:
            return None

        def commit(self) -> None:
            raise RuntimeError("database unavailable")

        def rollback(self) -> None:
            return None

    image_path = Path(__file__).resolve().parent.parent.parent / "HAM10000_images_part_1" / "ISIC_0024306.jpg"
    try:
        save_upload(
            FailingSession(),
            case_id=1,
            filename="cleanup.jpg",
            content_type="image/jpeg",
            file_bytes=image_path.read_bytes(),
        )
    except RuntimeError as exc:
        assert str(exc) == "database unavailable"
    else:
        raise AssertionError("Expected the simulated database failure to propagate.")
    assert list(tmp_path.iterdir()) == []


def test_upload_requires_authentication_and_case_ownership() -> None:
    unauthenticated = client.post(
        "/api/v1/upload/1",
        files={"file": ("broken.jpg", b"not an image", "image/jpeg")},
    )
    assert unauthenticated.status_code in {401, 403}

    owner_email = make_email()
    other_email = make_email()
    for email in (owner_email, other_email):
        registered = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": "strongpass123", "full_name": "Case Owner"},
        )
        assert registered.status_code == 201
    owner_token = client.post(
        "/api/v1/auth/login",
        json={"email": owner_email, "password": "strongpass123"},
    ).json()["access_token"]
    other_token = client.post(
        "/api/v1/auth/login",
        json={"email": other_email, "password": "strongpass123"},
    ).json()["access_token"]
    case = client.post(
        "/api/v1/cases",
        json={"title": "Owned case"},
        headers={"Authorization": f"Bearer {owner_token}"},
    ).json()

    detail = client.get(
        f"/api/v1/cases/{case['id']}",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert detail.status_code == 404
    upload = client.post(
        f"/api/v1/upload/{case['id']}",
        files={"file": ("broken.jpg", b"not an image", "image/jpeg")},
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert upload.status_code == 404


def test_expired_access_token_is_rejected() -> None:
    from app.services.auth_service import create_access_token

    expired_token = create_access_token("expired@example.com", timedelta(seconds=-1))
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert response.status_code == 401


def test_dataset_statistics_are_read_from_existing_report() -> None:
    response = client.get("/api/v1/dataset/stats")
    assert response.status_code == 200
    result = response.json()
    assert result["dataset_name"] == "HAM10000"
    assert result["image_count"] == 10015
    assert result["class_count"] == 7
    assert result["class_counts"] == {
        "akiec": 327,
        "bcc": 514,
        "bkl": 1099,
        "df": 115,
        "mel": 1113,
        "nv": 6705,
        "vasc": 142,
    }


def test_frontend_cors_origin_is_allowed() -> None:
    response = client.options(
        "/api/v1/cases",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    ip_origin_response = client.options(
        "/api/v1/cases",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert ip_origin_response.status_code == 200
    assert ip_origin_response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_invalid_credentials() -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "missing@example.com", "password": "wrongpass"},
    )
    assert response.status_code == 401


def test_preprocessing_output() -> None:
    from app.services.preprocessing_service import ImagePreprocessor

    file_path = Path(__file__).resolve().parent.parent.parent / "HAM10000_images_part_1" / "ISIC_0024306.jpg"
    data = file_path.read_bytes()
    image_array, metadata = ImagePreprocessor().preprocess(data)
    assert settings.TARGET_IMAGE_SIZE == (224, 224)
    assert image_array.shape == (224, 224, 3)
    assert image_array.dtype == np.float32
    assert metadata["channels"] == "RGB"
    assert metadata["array_shape"] == [224, 224, 3]
    assert metadata["target_size"] == [224, 224]
    assert metadata["normalization"] == "scale_0_1"
    assert image_array.min() >= 0.0
    assert image_array.max() <= 1.0


def test_preprocessing_converts_grayscale_and_scales_pixel_values() -> None:
    from app.services.preprocessing_service import ImagePreprocessor

    source = Image.new("L", (1, 1), color=128)
    encoded = BytesIO()
    source.save(encoded, format="PNG")

    image_array, metadata = ImagePreprocessor((3, 2)).preprocess(encoded.getvalue())

    assert image_array.shape == (2, 3, 3)
    assert metadata["channels"] == "RGB"
    assert metadata["source_format"] == "RGB"
    assert metadata["target_size"] == [3, 2]
    assert metadata["normalization"] == "scale_0_1"
    assert np.allclose(image_array, 128 / 255.0)
    assert image_array.min() >= 0.0
    assert image_array.max() <= 1.0
