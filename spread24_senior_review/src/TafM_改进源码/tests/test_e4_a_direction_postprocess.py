import inspect
import numpy as np
import pytest

from src.TafM_改进源码.direction_postprocess import (
    POSTPROCESS_MODES, REGIME_FEATURES, build_meta_features, fit_logistic_stacker,
)
from src.TafM_改进源码.train import train_target_day


def _fixture(n_days=40):
    rng = np.random.default_rng(7)
    names = list(REGIME_FEATURES)
    x = rng.normal(size=(n_days, 24, len(names))).astype(np.float64)
    base_signal = 0.6 * x[:, :, 0] - 0.4 * x[:, :, 1] + 0.2 * x[:, :, 3]
    y = (base_signal + rng.normal(scale=1.0, size=(n_days, 24)) > 0).astype(np.float64)
    p = 1 / (1 + np.exp(-(0.35 * base_signal + rng.normal(scale=.35, size=(n_days, 24)))))
    return p, x, y, names


def test_modes_and_default_contract():
    assert POSTPROCESS_MODES == {"none", "segment_logit", "regime_logit"}
    sig = inspect.signature(train_target_day)
    assert sig.parameters["direction_postprocess_mode"].default == "none"


@pytest.mark.parametrize("mode,dim", [("segment_logit", 5), ("regime_logit", 5 + len(REGIME_FEATURES))])
def test_meta_feature_shape_and_finite(mode, dim):
    p, x, _, names = _fixture()
    X, cols = build_meta_features(p, x, names, mode)
    assert X.shape == (40 * 24, dim)
    assert len(cols) == dim
    assert np.isfinite(X).all()


def test_regime_mode_requires_exact_feature_family():
    p, x, _, names = _fixture()
    with pytest.raises(ValueError, match="missing required features"):
        build_meta_features(p, x[:, :, :-1], names[:-1], "regime_logit")


@pytest.mark.parametrize("mode", ["segment_logit", "regime_logit"])
def test_logistic_stacker_fits_and_predicts(mode):
    p, x, y, names = _fixture()
    stacker = fit_logistic_stacker(p, x, y, names, mode, max_iter=40)
    q = stacker.predict_proba(p, x, names)
    assert q.shape == p.shape
    assert np.isfinite(q).all()
    assert np.all((q > 0) & (q < 1))
    audit = stacker.audit()
    assert audit["mode"] == mode
    assert audit["monitor_positive_rate"] > 0
    assert len(audit["feature_names"]) == len(stacker.weight)


def test_stacker_rejects_short_monitor():
    p, x, y, names = _fixture(n_days=10)
    with pytest.raises(ValueError, match="at least 20 monitor days"):
        fit_logistic_stacker(p, x, y, names, "segment_logit")
