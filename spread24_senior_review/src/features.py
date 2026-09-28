"""Backward-compatible import shim for the unchanged LightGBM baseline."""

from .lightgbm_源码.features import feature_columns, matrix

__all__ = ["feature_columns", "matrix"]

