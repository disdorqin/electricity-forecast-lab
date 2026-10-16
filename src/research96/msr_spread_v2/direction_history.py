"""Frozen docs/95 mature-history decision context, independent of any predictor.

Timeline: full curves D-254..D-2, then ONLY D-1 RT/DA p1..p56.
84 reliability target days end D-2; each expert on j ends j-2. Missing
calendar days fail closed, never compress dates into a different lag definition.
"""
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import numpy as np
from .contracts import allowed_day, ContractError, finite_curve

EXPERTS = ("d2", "w5", "w7", "w10", "w14", "e3", "e5")
SHORT_EXPERTS = ("w5", "w7", "w10", "w14")
RELIABILITY_DAYS = 84
EXP_HISTORY_DAYS = 168
REQUIRED_MATURE_DAYS = RELIABILITY_DAYS + EXP_HISTORY_DAYS + 1


@dataclass(frozen=True)
class DirectionHistoryContext:
    target_day: str
    prefix_spread: np.ndarray
    history_direction: np.ndarray
    override_permission: np.ndarray
    selected_expert: str
    reliability: float
    expert_reliabilities: dict
    provenance: dict


def expert_prediction(mature_signs, name):
    """Rows are complete consecutive days ending prediction-day minus two."""
    a = np.asarray(mature_signs)
    if a.ndim != 2 or a.shape[1] != 96 or a.dtype != np.bool_ or not len(a):
        raise ContractError("complete mature Boolean sign history [days,96] required")
    if name == "d2":
        return a[-1].copy()
    if name in SHORT_EXPERTS:
        w = int(name[1:])
        if len(a) < w:
            raise ContractError("incomplete mature history for " + name)
        return a[-w:].mean(0) >= .5
    if name in ("e3", "e5"):
        if len(a) < EXP_HISTORY_DAYS:
            raise ContractError("incomplete 168-day mature exponential history")
        h = a[-EXP_HISTORY_DAYS:]
        age = np.arange(len(h)-1, -1, -1, dtype=float)
        weights = np.exp(-np.log(2)*age/int(name[1:]))
        return (weights[:, None]*h).sum(0)/weights.sum() >= .5
    raise ContractError("unknown frozen history expert")


def history_context(store, target):
    """Read-only CurveStore/spy-provider API; no target evaluation reader."""
    d = allowed_day(target)
    cutoff = d-timedelta(days=2)
    start = cutoff-timedelta(days=REQUIRED_MATURE_DAYS-1)
    curves = []
    dates = [start+timedelta(days=i) for i in range(REQUIRED_MATURE_DAYS)]
    for historical in dates:
        rt = finite_curve(store.get(historical, "rt"), "direction mature RT")
        da = finite_curve(store.get(historical, "da"), "direction mature DA")
        curves.append(rt-da)
    spread = np.stack(curves)
    signs = spread > 0
    # These 84 full target truths are mature at D's origin. For historical
    # target row j, array [:j-1] ends j-2, not j-1.
    js = range(len(signs)-RELIABILITY_DAYS, len(signs))
    scores = {}
    for name in EXPERTS:
        correct = sum(int((expert_prediction(signs[:j-1], name) == signs[j]).sum()) for j in js)
        scores[name] = (float(correct)+2)/(float(RELIABILITY_DAYS*96)+4)
    pick = int(np.argmax([scores[name] for name in EXPERTS]))
    selected = EXPERTS[pick]
    reliability = scores[selected]
    predicted = expert_prediction(signs, selected)
    previous = d-timedelta(days=1)
    prefix = finite_curve(store.get(previous, "rt", 56), "direction D-1 RT prefix", 56) - finite_curve(
        store.get(previous, "da", 56), "direction D-1 DA prefix", 56)
    permission = np.zeros(96, dtype=bool)
    permission[:56] = ((prefix > 0) == predicted[:56]) & (reliability >= .52)
    permission[56:] = reliability >= .57
    short = np.stack([expert_prediction(signs, name) for name in SHORT_EXPERTS])
    permission &= (short == predicted[None]).sum(0) >= 3
    provenance = dict(max_mature_history_truth_day=cutoff.isoformat(),
        mature_history_start=start.isoformat(), mature_history_days=len(dates),
        reliability_target_start=dates[-RELIABILITY_DAYS].isoformat(),
        reliability_target_end=cutoff.isoformat(), reliability_days=RELIABILITY_DAYS,
        historical_expert_cutoff="j-2", exp_history_max_days=EXP_HISTORY_DAYS,
        d1_observed_day=previous.isoformat(), d1_observed_prefix_end=56,
        mature_spread_sha256=hashlib.sha256(spread.tobytes()).hexdigest(),
        prefix_spread_sha256=hashlib.sha256(prefix.tobytes()).hexdigest(),
        source_identity=getattr(store, "source_identity", None), target_truth_opened=False)
    for a in (prefix, predicted, permission):
        a.setflags(write=False)
    return DirectionHistoryContext(d.isoformat(), prefix, predicted, permission,
                                   selected, reliability, scores, provenance)
