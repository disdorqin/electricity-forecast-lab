"""Read-only causal F providers. Never request target truth or D-1 price suffix."""
from datetime import timedelta
import hashlib
import numpy as np
from .contracts import allowed_day, finite_curve, label_cutoff, ContractError
from .frontier_f import make_candidates, FrontierHistoryRow, FrontierFConfig


def context_candidates(store, target, full, ddv1):
    d = allowed_day(target)
    cutoff = label_cutoff(d)
    dates = [(cutoff-timedelta(days=i)).isoformat() for i in range(28)]
    mature = np.stack([finite_curve(store.get(x, "rt"), "F mature RT")-
                       finite_curve(store.get(x, "da"), "F mature DA") for x in dates])
    previous = (d-timedelta(days=1)).isoformat()
    prefix = finite_curve(store.get(previous, "rt", 56), "F prefix RT", 56)-finite_curve(
        store.get(previous, "da", 56), "F prefix DA", 56)
    s = np.asarray(full["spread_scenarios"]).reshape(256, 96).astype(float)
    if not np.isfinite(s).all():
        raise ContractError("nonfinite FULL scenarios")
    return make_candidates(target, np.asarray(full["p_positive"]).reshape(96).astype(float),
        np.asarray(full["spread_median"]).reshape(96).astype(float), s.mean(0),
        np.asarray(ddv1["final_direction"]).reshape(96), prefix, mature,
        dict(mature_days=dates, max_mature_label_day=cutoff.isoformat(), d1_day=previous,
             d1_prefix_end=56, mature_spread_sha256=hashlib.sha256(mature.tobytes()).hexdigest(),
             prefix_spread_sha256=hashlib.sha256(prefix.tobytes()).hexdigest(),
             source_identity=getattr(store, "source_identity", {})))


def scored_history(store, target, dense_start, reader):
    """reader(day)->(sealed F arrays, input/artifact hashes) or None for missing.

    Missing calendar rows are not compressed into observations at different lags.
    The policy falls back to V1; corrupt artifacts propagate an exception.
    """
    cutoff = label_cutoff(target)
    start = max(allowed_day(dense_start), cutoff-timedelta(days=FrontierFConfig().window-1))
    rows = []
    complete = True
    for i in range(max(0, (cutoff-start).days+1)):
        d = (start+timedelta(days=i)).isoformat()
        record = reader(d)
        if record is None:
            complete = False
            continue
        arrays, hashes = record
        spread = finite_curve(store.get(d, "rt"), "F performance RT")-finite_curve(
            store.get(d, "da"), "F performance DA")
        y = spread > 0
        pool = np.asarray(arrays["candidate_pool"])
        if pool.shape != (6, 96) or pool.dtype != np.bool_:
            raise ContractError("invalid historical F candidate pool")
        rows.append(FrontierHistoryRow(d, arrays["features"], (pool == y).mean(1),
            float((arrays["positive_candidate"] == y).mean()),
            float((arrays["negative_candidate"] == y).mean()),
            float((arrays["post_positive"] == y).mean()),
            dict(**hashes, performance_truth_sha256=hashlib.sha256(spread.tobytes()).hexdigest())))
    return rows, complete
