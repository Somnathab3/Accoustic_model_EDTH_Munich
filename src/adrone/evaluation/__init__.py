"""Evaluation module initialization"""
from .metrics import (
    compute_metrics,
    compute_roc_metrics,
    compute_calibration_error,
    TemperatureScaling,
    evaluate_model,
    print_evaluation_report
)

__all__ = [
    'compute_metrics',
    'compute_roc_metrics',
    'compute_calibration_error',
    'TemperatureScaling',
    'evaluate_model',
    'print_evaluation_report'
]
