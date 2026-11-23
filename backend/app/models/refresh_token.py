"""
SQLAlchemy model for storing and revoking JWT refresh tokens.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class RefreshToken(Base):
    """
    Persisted refresh tokens for secure session management and token revocation.
    """
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(512), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    revoked = Column(Boolean, default=False, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    user = relationship("User", backref="refresh_tokens")
