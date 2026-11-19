"""
User model representing system accounts and role-based permissions.
"""

from sqlalchemy import Column, Integer, String, DateTime, Enum as SAEnum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from app.database import Base


class UserRole(str, enum.Enum):
    PATIENT = "patient"
    CLINICIAN = "clinician"
    RESEARCHER = "researcher"
    ML_ENGINEER = "ml_engineer"
    ADMIN = "admin"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    role = Column(
        SAEnum(UserRole, values_callable=lambda obj: [e.value for e in obj]),
        default=UserRole.PATIENT,
        nullable=False,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    created_cases = relationship("Case", back_populates="creator", foreign_keys="Case.created_by")
    audit_logs = relationship("AuditLog", back_populates="user", foreign_keys="AuditLog.user_id")

    def __repr__(self):
        return f"<User id={self.id} email={self.email} role={self.role}>"
