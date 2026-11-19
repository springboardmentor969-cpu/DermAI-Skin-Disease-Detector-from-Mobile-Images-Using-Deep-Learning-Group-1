"""
AuditLog model for recording critical system events, access logs, and workflow actions.
Ensures accountability and compliance with health information security standards.
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    action = Column(String(150), nullable=False)  # e.g., "USER_LOGIN", "CASE_CREATED", "IMAGE_UPLOADED"
    details = Column(JSON, nullable=True)  # Contextual payload or metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog id={self.id} user_id={self.user_id} action={self.action}>"
