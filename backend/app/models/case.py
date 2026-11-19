"""
Case model representing a clinical screening submission for a patient.
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class Case(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False, index=True)
    body_location = Column(String(100), nullable=False)  # e.g., "Left Forearm", "Back", "Cheek"
    status = Column(String(50), default="submitted", nullable=False)  # submitted, reviewing, flagged, completed
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    patient = relationship("Patient", back_populates="cases")
    creator = relationship("User", back_populates="created_cases")
    images = relationship("Image", back_populates="case", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Case id={self.id} patient_id={self.patient_id} status={self.status}>"
