"""Pure frozen Phase19 -> 41 -> 43 -> 44 policy; no filesystem or truth reader.

Specialists replace the WHOLE day with a DDV1-based candidate, not only the
active segments of the selected base. Negative competence compares to each
historical day's post-positive decision. These details are frozen semantics.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np
from .contracts import ContractError, allowed_day, finite_curve, label_cutoff

EXPERTS = ("BASE", "V1", "MEDIAN", "MEAN", "NEG_EWMA", "NEG_ENERGY")
SEGMENTS = ((0, 32), (32, 64), (64, 96))


@dataclass(frozen=True)
class FrontierFConfig:
    version: str = "F_PHASE44_V1"
    window: int = 84
    minimum_history: int = 70
    dimensions: tuple = (0, 2, 3)
    neighbors: int = 15
    fallback: str = "V1"
    positive_high: float = .625
    negative_low: float = .40
    negative_pool_low: float = .375
    half_life: int = 5
    energy_gate: float = .10
    prefix: int = 56

    def __post_init__(self):
        if asdict(self) != {k: v.default for k, v in self.__dataclass_fields__.items()}:
            raise ContractError("F constants are frozen, not a tuning interface")

    @property
    def config_hash(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class FrontierCandidates:
    target_day: str
    pool: np.ndarray
    features: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    positive_active: bool
    negative_active: bool
    provenance: dict


@dataclass(frozen=True)
class FrontierHistoryRow:
    target_day: str
    features: np.ndarray
    candidate_accuracy: np.ndarray
    positive_accuracy: float
    negative_accuracy: float
    post_positive_accuracy: float
    artifact_hashes: dict


@dataclass(frozen=True)
class FrontierFResult:
    final_direction: np.ndarray
    post_positive: np.ndarray
    candidates: FrontierCandidates
    trace: dict

    def arrays(self):
        c = self.candidates
        return dict(final_direction=self.final_direction, post_positive=self.post_positive,
                    candidate_pool=c.pool, features=c.features, positive_candidate=c.positive,
                    negative_candidate=c.negative)


def make_candidates(target, p_positive, median, scenario_mean, ddv1, prefix, mature28,
                    provenance=None):
    cfg = FrontierFConfig()
    target = allowed_day(target).isoformat()
    p = finite_curve(p_positive, "F p_positive")
    med = finite_curve(median, "F median")
    mean = finite_curve(scenario_mean, "F scenario mean")
    v1 = np.asarray(ddv1)
    if v1.shape != (96,) or v1.dtype != np.bool_ or ((p < 0) | (p > 1)).any():
        raise ContractError("F requires Boolean DDV1 and legal probabilities")
    prev = finite_curve(prefix, "F D-1 prefix", 56)
    mature = np.asarray(mature28, float)
    # Newest day first; exactly 28 consecutive full days ending D-2.
    if mature.shape != (28, 96) or not np.isfinite(mature).all():
        raise ContractError("F requires dense 28 mature full curves")
    pc = float((prev > 0).mean())
    pe = float(np.sum(prev)/(np.sum(np.abs(prev))+1e-9))
    weights = np.exp(-np.log(2)*np.arange(28)/cfg.half_life)
    pool = np.stack((p > .5, v1, med > 0, mean > 0, v1, v1)).copy()
    pos, neg = v1.copy(), v1.copy()
    pa = na = False
    for lo, hi in SEGMENTS:
        rates = (mature[:, lo:hi] > 0).mean(1)
        rr = float(np.dot(rates, weights)/weights.sum())
        if rr < cfg.negative_pool_low and pc < cfg.negative_pool_low:
            pool[4, lo:hi] = False
        if rr < cfg.negative_pool_low and pe < -cfg.energy_gate:
            pool[5, lo:hi] = False
        if rr > cfg.positive_high and pe > cfg.energy_gate:
            pa = True
            pos[lo:hi] = True
        if rr < cfg.negative_low and pe < -cfg.energy_gate:
            na = True
            neg[lo:hi] = False
    # Keep the original mean-of-daily-means arithmetic, not a pooled sum.
    r7 = float(np.mean([(x > 0).mean() for x in mature[:7]]))
    r14 = float(np.mean([(x > 0).mean() for x in mature[:14]]))
    features = np.array([pc, r7, r14], float)
    for a in (pool, pos, neg, features):
        a.setflags(write=False)
    return FrontierCandidates(target, pool, features, pos, neg, pa, na, provenance or {})


def decide_frontier(candidates, history, dense_start, history_complete=True):
    """History is ordered, dense, and scored only through current target D-2."""
    from datetime import timedelta
    cfg = FrontierFConfig()
    c = candidates
    d = allowed_day(c.target_day)
    cutoff = label_cutoff(d)
    start = max(allowed_day(dense_start), cutoff-timedelta(days=cfg.window-1))
    expected = [(start+timedelta(days=i)).isoformat()
                for i in range(max(0, (cutoff-start).days+1))]
    dates = [r.target_day for r in history]
    if any(allowed_day(x) > cutoff for x in dates):
        raise ContractError("F candidate-performance truth exceeds D-2")
    if dates != sorted(set(dates)):
        raise ContractError("F history must be unique chronological")
    complete = history_complete and dates == expected
    h = np.arange(len(history)) if complete else np.array([], dtype=int)
    best, score = 1, 0.0
    if len(h):
        if len(h) >= cfg.minimum_history:
            X = np.stack([r.features for r in history])
            sd = X.std(0)+1e-6
            dist = np.mean(((X-c.features)/sd)**2, axis=1)
            h = h[np.argsort(dist)[:cfg.neighbors]]  # frozen default quicksort/ties
            scores = np.stack([history[i].candidate_accuracy for i in h]).mean(0)
            best, score = int(np.argmax(scores)), float(np.max(scores))
        else:
            score = float(np.mean([r.candidate_accuracy[1] for r in history]))
    post = c.pool[best].copy()
    pu = bool(c.positive_active and len(h) >= 15 and
              float(np.mean([history[i].positive_accuracy for i in h])) >= score)
    if pu:
        post = c.positive.copy()
    nu = bool(c.negative_active and len(h) >= 15 and
              float(np.mean([history[i].negative_accuracy for i in h])) >=
              float(np.mean([history[i].post_positive_accuracy for i in h])))
    final = c.negative.copy() if nu else post.copy()
    trace = dict(target_day=c.target_day, policy=cfg.version, config=asdict(cfg),
                 config_sha256=cfg.config_hash, selected_base_expert=EXPERTS[best],
                 selected_base_competence=score, dense_history_complete=complete,
                 fallback_reason=None if complete else "missing dense calendar history; V1 fail-safe",
                 history_days=dates, neighbor_days=[history[i].target_day for i in h],
                 max_performance_label_day=max(dates) if dates else None,
                 positive_specialist_active=c.positive_active, positive_specialist_applied=pu,
                 negative_specialist_active=c.negative_active, negative_specialist_applied=nu,
                 current_context=c.provenance,
                 history_artifact_hashes={r.target_day:r.artifact_hashes for r in history},
                 final_direction_sha256=hashlib.sha256(final.tobytes()).hexdigest(),
                 target_truth_opened=False)
    for a in (final, post):
        a.setflags(write=False)
    return FrontierFResult(final, post, c, trace)
