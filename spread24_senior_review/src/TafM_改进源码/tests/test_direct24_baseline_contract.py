from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch

from src.TafM_改进源码.config import V2Config
from src.TafM_改进源码.contracts import TEMPORAL_FEATURES
from src.TafM_改进源码.evaluate import run_direct24_baselines, summarize_regression_prediction_frame
from src.TafM_改进源码.models.direct24_baseline import FutureOnlyDirect24, SameInputDirect24


class _BaselineStore:
    def __init__(self):
        self.days = np.asarray([date(2026, 4, 1) + timedelta(days=i) for i in range(33)], dtype=object)
        self.day_index = pd.DataFrame({"target_day": self.days})
        rng = np.random.default_rng(5)
        self.x_future = rng.normal(size=(33, 24, 2)).astype(np.float32)
        self.x_future[:, :, 0] *= 1e6  # exercises shared scaling/clipping
        self.x_hist = rng.normal(size=(33, 168, 7)).astype(np.float32)
        self.y_source = rng.normal(size=(33, 24)).astype(np.float32)
        self.source_sha256 = "source"
        self.sequence_sha256 = "sequence"
        self.quarantined_index = 5

    def selector_indices(self, _selector):
        return ["f1", "f2"], np.asarray([0, 1]), None

    def eligibility(self, _selector, current_target_day=None, requested_days=None):
        if requested_days is not None:
            days = set(requested_days)
            return {"eligible_indices": [i for i, day in enumerate(self.days)
                                        if day in days and i != self.quarantined_index]}
        cutoff = current_target_day - timedelta(days=2)
        return {"eligible_indices": [i for i, day in enumerate(self.days)
                                    if day <= cutoff and i != self.quarantined_index]}


def test_sameinput_direct24_consumes_history_and_future_and_history_changes_output():
    model = SameInputDirect24(n_features=3, hidden=8, history_dim=4)
    hist = torch.zeros(1, 168, 7)
    future = torch.zeros(1, 24, 3)
    first = model(hist, future)
    changed = model(hist + 1.0, future)
    assert first.shape == (1, 24)
    assert not torch.equal(first, changed)
    future_only = FutureOnlyDirect24(n_features=3, hidden=8)
    assert future_only(future).shape == (1, 24)
    assert "FutureOnly" in type(future_only).__name__


def test_regression_baseline_has_no_fake_probability_or_brier():
    frame = pd.DataFrame({"y_true_model": [1.0, -2.0, 3.0, -4.0],
        "prediction_model": [0.5, -1.0, -2.0, 4.0], "target_day": ["2026-06-01"] * 4,
        "hour_business": [1, 2, 3, 4]})
    metrics = summarize_regression_prediction_frame(frame)
    assert metrics["auc"] == "N/A"
    assert metrics["auc_status"] == "NOT_A_PROBABILITY_METRIC"
    assert metrics["rank_auc"] is not None
    assert metrics["brier"] == "N/A"
    assert metrics["signed_mae"] >= 0
    assert metrics["magnitude_mae"] >= 0


def test_direct24_models_follow_d2_quarantine_and_shared_preprocessing(monkeypatch):
    import src.TafM_改进源码.selector as selector_module
    monkeypatch.setattr(selector_module, "load_selector_manifest", lambda: ({"status": "FROZEN"}, "selector-sha"))
    store = _BaselineStore()
    target = date(2026, 5, 3)
    selector = {"status": "FROZEN"}
    config = V2Config(max_epochs=1, batch_size=8, ple_enabled=False)
    runs = run_direct24_baselines(store, [target], selector, config=config, profile="default")
    assert set(runs) == {"FutureOnly-Direct24", "SameInput-Direct24"}
    for result in runs.values():
        audit = result["preprocessing_audits"][0]
        state = audit["preprocessor_state"]
        assert state["fit_day_end"] <= (target - timedelta(days=2)).isoformat()
        assert state["selector_sha256"] == "selector-sha"
        assert state["fit_sample_count"] == 30  # 31 D-2-eligible days minus one masked feature/day
        assert audit["numerical_audit"]["future"]["postclip_abs_max"] <= 10.000001
        assert audit["numerical_audit"]["temporal"]["postclip_abs_max"] <= 10.000001
        assert np.isfinite(result["rows"][0]["prediction"]).all()
