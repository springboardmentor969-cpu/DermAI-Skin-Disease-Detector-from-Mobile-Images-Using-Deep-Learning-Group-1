"""
Prediction model storing deep learning inference outputs and confidence scores.
DISCLAIMER: Serves for decision support and risk stratification, NOT medical diagnosis.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True)
    model_version = Column(String(100), nullable=False)  # e.g., "efficientnet_b0_v1.0"
    predicted_class = Column(String(100), nullable=False)  # e.g., "melanocytic_nevus", "melanoma"
    confidence = Column(Float, nullable=False)  # e.g., 0.942
    risk_level = Column(String(50), nullable=True)  # e.g., "HIGH", "MEDIUM", "LOW"
    top_n = Column(JSON, nullable=True)  # List of top-K classes and their softmax probabilities
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    image = relationship("Image", back_populates="predictions")

    def __repr__(self):
        return f"<Prediction id={self.id} image_id={self.image_id} class={self.predicted_class} conf={self.confidence}>"
