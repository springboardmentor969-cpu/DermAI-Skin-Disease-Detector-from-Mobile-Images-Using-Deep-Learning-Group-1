"""
Temperature Scaling for Probability Calibration in DermAI.

Provides post-hoc probability calibration (Guo et al., 2017) to optimize
the temperature parameter T > 0, minimizing Negative Log-Likelihood (NLL)
and bringing the Expected Calibration Error (ECE) below 0.10.
"""

from .temperature_scaling import TemperatureScaler, compute_ece

__all__ = ["TemperatureScaler", "compute_ece"]
