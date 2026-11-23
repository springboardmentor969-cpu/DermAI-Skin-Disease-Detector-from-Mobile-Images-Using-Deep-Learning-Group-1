"""
Image model representing skin lesion photographs stored in MinIO S3 object storage.
Tracks source, capture metadata, and automated image quality assessment results.
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    storage_key = Column(String(500), nullable=False)  # MinIO object path, e.g. "cases/1/abc123_thumb.jpg"
    source = Column(String(50), nullable=False, default="camera")  # "camera" or "gallery"
    captured_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    quality_label = Column(String(50), nullable=True)  # "good", "acceptable", "poor"
    quality_details = Column(JSON, nullable=True)  # blurriness, contrast, brightness, dimensions

    # Relationships
    case = relationship("Case", back_populates="images")
    predictions = relationship("Prediction", back_populates="image", cascade="all, delete-orphan")
    reviews = relationship("ClinicalReview", back_populates="image", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Image id={self.id} case_id={self.case_id} quality={self.quality_label}>"
