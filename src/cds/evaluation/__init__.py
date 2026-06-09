"""Evaluation and model comparison tools.

Strong, research-grade components for discriminating between
scientific models/hypotheses using information criteria.
"""
from cds.evaluation.comparison import ModelComparison, compute_delta_bic_strength

__all__ = ["ModelComparison", "compute_delta_bic_strength"]
