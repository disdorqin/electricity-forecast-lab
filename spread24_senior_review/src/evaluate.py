"""Backward-compatible import shim for the unchanged LightGBM baseline."""

from .lightgbm_源码.evaluate import metrics, summarize

__all__ = ["metrics", "summarize"]

