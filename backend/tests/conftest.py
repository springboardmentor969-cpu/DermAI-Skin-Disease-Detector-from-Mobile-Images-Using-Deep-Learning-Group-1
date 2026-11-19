"""
Pytest configuration and test database fixtures using an isolated SQLite database.
"""

import sys
from pathlib import Path
import pytest

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.database import Base, get_db
from app.main import app

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
    """Provides a TestClient with the database session overridden."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


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
