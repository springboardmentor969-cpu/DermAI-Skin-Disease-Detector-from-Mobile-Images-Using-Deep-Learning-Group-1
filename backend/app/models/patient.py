"""
Patient model holding strictly de-identified, case-level demographic data.
Contains NO personally identifiable information (PII) to ensure patient privacy.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    # Strictly non-identifying demographic attributes
    age = Column(Integer, nullable=True)
    gender = Column(String(50), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    cases = relationship("Case", back_populates="patient", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Patient id={self.id} age={self.age} gender={self.gender}>"
