"""Backward-compatible import shim for the unchanged LightGBM baseline."""

from .lightgbm_源码.leakage import audit, training_last_day

__all__ = ["audit", "training_last_day"]

