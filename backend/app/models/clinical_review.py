"""
ClinicalReview model representing expert clinician evaluations and confirmations of AI skin lesion predictions.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from app.database import Base


class ClinicalReview(Base):
    __tablename__ = "clinical_reviews"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True)
    clinician_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    notes = Column(Text, nullable=True)
    confirmed_class = Column(String(10), nullable=False)
    overridden = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    image = relationship("Image", back_populates="reviews")
    clinician = relationship("User", backref="clinical_reviews")

    def __repr__(self):
        return (
            f"<ClinicalReview id={self.id} image_id={self.image_id} "
            f"clinician_id={self.clinician_id} confirmed={self.confirmed_class} "
            f"overridden={self.overridden}>"
        )
