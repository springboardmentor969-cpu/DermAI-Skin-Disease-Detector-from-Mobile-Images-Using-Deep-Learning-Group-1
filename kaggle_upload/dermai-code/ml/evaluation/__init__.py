"""DermAI Evaluation Module — Milestone 2."""

from ml.evaluation.evaluate import (
    load_model,
    run_inference,
    compute_all_metrics,
    plot_confusion_matrices,
    plot_roc_curves,
    write_melanoma_summary,
    CLASS_NAMES,
    ALL_LABELS,
)
from ml.evaluation.calibrate_confidence import (
    compute_ece,
    fit_temperature_scaling,
    evaluate_threshold_sweep,
)

__all__ = [
    "load_model",
    "run_inference",
    "compute_all_metrics",
    "plot_confusion_matrices",
    "plot_roc_curves",
    "write_melanoma_summary",
    "CLASS_NAMES",
    "ALL_LABELS",
    "compute_ece",
    "fit_temperature_scaling",
    "evaluate_threshold_sweep",
]

