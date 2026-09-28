"""Backward-compatible import shim for the unchanged LightGBM baseline."""

from .lightgbm_源码.data import REQUIRED, load_frozen, load_raw

__all__ = ["REQUIRED", "load_frozen", "load_raw"]

