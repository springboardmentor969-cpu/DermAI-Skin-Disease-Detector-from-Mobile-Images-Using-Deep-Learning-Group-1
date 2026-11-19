"""
Database connection, session management, and Base class definition for SQLAlchemy.
"""

from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.config import settings

# Create database engine
# connect_args={"check_same_thread": False} only needed if sqlite is used
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,  # Automatically check if connection is alive before querying
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base model from which all database models inherit
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a transactional database session per request.
    Ensures session is properly closed when the request is completed.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_connection() -> bool:
    """
    Utility function to test database connectivity (e.g. for health checks).
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
