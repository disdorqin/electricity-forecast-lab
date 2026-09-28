"""Backward-compatible import shim for the unchanged LightGBM baseline."""

from .lightgbm_源码.model import fit_predict

__all__ = ["fit_predict"]

