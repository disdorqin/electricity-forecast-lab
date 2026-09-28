from __future__ import annotations

import numpy as np
from datetime import timedelta

from src.TafM_改进源码.dataset import SequenceStore
from src.TafM_改进源码.target_adapter import source_to_model_target


def _day_index(store, day):
    matches = np.flatnonzero(np.asarray([str(x) == day for x in store.days]))
    assert len(matches) == 1
    return int(matches[0])


def test_real_sequence_selected_manifest_quarantine_is_feature_level():
    store = SequenceStore.load()
    i = _day_index(store, "2026-08-17")
    bad = np.flatnonzero(np.asarray(store.quarantine_mask[i]).any(axis=0))
    good = np.flatnonzero(~np.asarray(store.quarantine_mask[i]).any(axis=0))
    assert len(bad) and len(good)
    unsafe = {"status":"FROZEN", "selected_features":[store.feature_names[int(bad[0])]],
              "selected_indices":[int(bad[0])]}
    safe = {"status":"FROZEN", "selected_features":[store.feature_names[int(good[0])]],
            "selected_indices":[int(good[0])]}
    assert i not in store.eligibility(unsafe, requested_days=["2026-08-17"])["eligible_indices"]
    assert i in store.eligibility(safe, requested_days=["2026-08-17"])["eligible_indices"]


def test_real_sequence_target_adapter_and_d2_cutoff():
    store = SequenceStore.load()
    source = np.asarray(store.y_source[100:101], dtype=np.float32)
    np.testing.assert_array_equal(source_to_model_target(source), -source)
    selector = {"status":"FROZEN", "selected_features":[store.feature_names[0]],
                "selected_indices":[0]}
    day = store.days[100]
    got = store.eligibility(selector, current_target_day=day + timedelta(days=2))
    assert 100 in got["eligible_indices"] or 100 in got["quarantine_excluded_indices"]
    # The sample exactly at D-2 is considered by the chronological cutoff.
    assert 100 not in got["temporal_or_cutoff_excluded_indices"]
