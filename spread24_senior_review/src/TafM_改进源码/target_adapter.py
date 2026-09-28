"""Explicit source-native DA-RT to V2 model-target RT-DA conversion."""
from __future__ import annotations

import numpy as np


SOURCE_TARGET = "DA-RT"
MODEL_TARGET = "RT-DA"


def source_to_model_target(y_source):
    """Convert target values only; frozen source and engineered predictors stay untouched."""
    try:
        import torch
        if isinstance(y_source, torch.Tensor):
            return -y_source
    except ImportError:
        pass
    return -np.asarray(y_source)


def model_direction(y_model):
    try:
        import torch
        if isinstance(y_model, torch.Tensor):
            return y_model > 0
    except ImportError:
        pass
    return np.asarray(y_model) > 0
