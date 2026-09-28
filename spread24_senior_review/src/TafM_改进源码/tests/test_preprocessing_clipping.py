from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.TafM_改进源码.contracts import TEMPORAL_FEATURES
from src.TafM_改进源码.preprocessing import fit_preprocessor, transform_future, transform_hist


class _Store:
    def __init__(self, test_outlier):
        self.days = np.asarray([date(2025, 1, 1) + timedelta(days=i) for i in range(3)], dtype=object)
        self.day_index = pd.DataFrame({"target_day": self.days})
        self.x_future = np.ones((3, 24, 2), dtype=np.float32)
        self.x_future[2, :, 0] = test_outlier
        self.x_hist = np.ones((3, 168, 7), dtype=np.float32)
        self.x_hist[2, :, 0] = test_outlier
        self.y_source = np.tile(np.arange(24, dtype=np.float32), (3, 1))
        self.source_sha256 = "source"
        self.sequence_sha256 = "sequence"

    def selector_indices(self, _manifest):
        return ["feature_a", "feature_b"], np.asarray([0, 1]), None

    def eligibility(self, _manifest, requested_days=None):
        if requested_days is None:
            return {"eligible_indices": [0, 1]}
        requested = set(requested_days)
        return {"eligible_indices": [i for i, d in enumerate(self.days) if d in requested]}


def _fit(store):
    manifest = {"status": "FROZEN"}
    return fit_preprocessor(store, [0, 1], manifest, selector_sha256="selector", n_bins=4,
                            future_clip_abs=10.0, temporal_clip_abs=10.0)


def test_extreme_outlier_clips_after_train_only_scaling_and_ple_fit():
    store = _Store(1e30)
    state = _fit(store)
    future = transform_future(store.x_future[2:3, :, [0, 1]], state)
    temporal = transform_hist(store.x_hist[2:3], state)
    assert np.isfinite(future).all() and np.isfinite(temporal).all()
    assert np.max(np.abs(future)) <= 10.000001
    assert np.max(np.abs(temporal)) <= 10.000001
    assert np.isfinite(np.asarray(state.ple_bins)).all()
    assert np.max(np.asarray(state.ple_bins)) <= 10.000001
    assert state.future_preclip_abs_max <= 10.000001
    assert state.future_clip_fraction_fit == 0.0


def test_nontraining_extreme_cannot_change_train_scaler_or_bins():
    state_a = _fit(_Store(1e6))
    state_b = _fit(_Store(1e30))
    assert state_a.future_median == state_b.future_median
    assert state_a.future_scale == state_b.future_scale
    assert state_a.ple_bins == state_b.ple_bins
    assert state_a.temporal_scale == state_b.temporal_scale

