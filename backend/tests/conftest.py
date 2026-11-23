"""
Pytest configuration and test database fixtures using an isolated SQLite database.
"""

import sys
import io
from pathlib import Path
import pytest
from PIL import Image as PILImage

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.database import Base, get_db
from app.main import app
from app.models.user import User, UserRole
from app.services.auth_service import hash_password, create_access_token

# In-memory SQLite database isolated per test session
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def mock_db_check(monkeypatch):
    """Mocks database connectivity check to avoid network timeouts during tests."""
    from app import database
    monkeypatch.setattr(database, "check_database_connection", lambda: True)


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Resets the rate limiter state before and after each test."""
    try:
        from app.core.limiter import limiter
        limiter.reset()
    except Exception:
        pass
    yield
    try:
        from app.core.limiter import limiter
        limiter.reset()
    except Exception:
        pass



@pytest.fixture
def db_session():
    """Provides a transactional database session per test and cleans up data."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session):
    """Provides an unauthenticated TestClient with database session overridden."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def patient_user(db_session):
    """Creates a patient user record in test database."""
    user = User(
        email="patient_fixture@example.com",
        password_hash=hash_password("Password123!"),
        full_name="Jane Patient",
        role=UserRole.PATIENT,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def clinician_user(db_session):
    """Creates a clinician user record in test database."""
    user = User(
        email="clinician_fixture@example.com",
        password_hash=hash_password("Password123!"),
        full_name="Dr. Alex Clinician",
        role=UserRole.CLINICIAN,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def authenticated_patient_client(db_session, patient_user):
    """Authenticated TestClient for patient role."""
    def override_get_db():
        yield db_session
    app.dependency_overrides[get_db] = override_get_db
    token = create_access_token({"sub": patient_user.email, "role": patient_user.role.value, "user_id": patient_user.id})
    with TestClient(app, headers={"Authorization": f"Bearer {token}"}) as c:
        c.auth_user = patient_user
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def patient_client(authenticated_patient_client):
    """Alias for authenticated patient test client."""
    return authenticated_patient_client


@pytest.fixture
def authenticated_clinician_client(db_session, clinician_user):
    """Authenticated TestClient for clinician role."""
    def override_get_db():
        yield db_session
    app.dependency_overrides[get_db] = override_get_db
    token = create_access_token({"sub": clinician_user.email, "role": clinician_user.role.value, "user_id": clinician_user.id})
    with TestClient(app, headers={"Authorization": f"Bearer {token}"}) as c:
        c.auth_user = clinician_user
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def clinician_client(authenticated_clinician_client):
    """Alias for authenticated clinician test client."""
    return authenticated_clinician_client


@pytest.fixture
def sample_test_image() -> bytes:
    """Fixture that creates a sample test image (3x3 pixel PNG bytes)."""
    buf = io.BytesIO()
    img = PILImage.new("RGB", (3, 3), color=(255, 120, 80))
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def valid_test_image() -> bytes:
    """Fixture that creates a valid 250x250 RGB JPEG image."""
    buf = io.BytesIO()
    img = PILImage.new("RGB", (250, 250), color=(180, 120, 90))
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def mock_storage_service(monkeypatch):
    """In-memory mock for MinIO S3 storage so tests run hermetically without external dependencies."""
    from app.services.storage_service import storage_service

    storage_dict = {}

    def mock_ensure_bucket(bucket_name=None):
        return bucket_name or "dermai-images"

    def mock_upload_file(file_bytes, storage_key, content_type, bucket_name=None):
        storage_dict[storage_key] = file_bytes
        return storage_key

    def mock_get_file(storage_key, bucket_name=None):
        if storage_key in storage_dict:
            return storage_dict[storage_key]
        return b"test_binary_payload"

    def mock_get_presigned_url(storage_key, expires_in=3600, bucket_name=None):
        return f"http://testserver/storage/{storage_key}"

    monkeypatch.setattr(storage_service, "ensure_bucket_exists", mock_ensure_bucket)
    monkeypatch.setattr(storage_service, "upload_file", mock_upload_file)
    monkeypatch.setattr(storage_service, "get_file", mock_get_file)
    monkeypatch.setattr(storage_service, "get_presigned_url", mock_get_presigned_url)
