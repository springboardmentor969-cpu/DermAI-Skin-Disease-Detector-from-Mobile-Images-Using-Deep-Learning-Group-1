"""
Multi-Model Soft-Voting Ensemble Inference for DermAI.
"""

from .ensemble_predict import ensemble_predict, EnsemblePredictor

__all__ = ["ensemble_predict", "EnsemblePredictor"]
