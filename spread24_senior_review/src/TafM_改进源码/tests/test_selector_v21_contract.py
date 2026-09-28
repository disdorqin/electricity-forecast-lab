from src.TafM_改进源码.dataset import SequenceStore
from src.TafM_改进源码.selector import chronological_folds


def test_selector_v21_folds_are_oos_with_d2_label_gap():
    store=SequenceStore.load()
    folds=chronological_folds(store.days,"2025-12-31")
    assert len(folds)==4
    assert all(f["valid_end"]<="2025-12-31" for f in folds)
    for fold in folds:
        valid_start=__import__("datetime").date.fromisoformat(fold["valid_start"])
        train_end=__import__("datetime").date.fromisoformat(fold["train_end"])
        assert (valid_start-train_end).days>=2
