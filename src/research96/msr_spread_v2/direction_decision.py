"""DirectionDecisionV1: frozen Boolean business policy; numeric arrays untouched."""
from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np
from .contracts import ContractError, finite_curve
from .direction_history import EXPERTS, history_context

SEGMENTS = (("1-8", 0, 32), ("9-16", 32, 64), ("17-24", 64, 96))


@dataclass(frozen=True)
class DirectionDecisionConfig:
    name: str = "engineering-v1"
    version: str = "DirectionDecisionV1"
    segment_multipliers: tuple = (1., 1., .5)
    threshold_cap: float = .2
    experts: tuple = EXPERTS
    reliability_days: int = 84
    exp_history_max_days: int = 168
    prefix_reliability: float = .52
    suffix_reliability: float = .57
    short_consensus: int = 3
    fuzzy_band: tuple = (.2, .9)
    segment_overlays: tuple = ("meta_pband", "safe50", "safe40")
    safety_medians: tuple = (None, 50., 40.)

    def __post_init__(self):
        # The dataclass documents a contract, not a tuning interface.
        if asdict(self) != {k: v.default for k, v in self.__dataclass_fields__.items()}:
            raise ContractError("DirectionDecisionV1 constants are frozen")

    def to_dict(self):
        return asdict(self)

    @property
    def config_hash(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class DirectionDecisionResult:
    base_direction: np.ndarray
    biased_direction: np.ndarray
    final_direction: np.ndarray
    threshold: np.ndarray
    history_direction: np.ndarray
    override_permission: np.ndarray
    override_applied: np.ndarray
    safety_blocked: np.ndarray
    trace: dict

    def arrays(self):
        return {name: getattr(self, name) for name in (
            "base_direction", "biased_direction", "final_direction", "threshold",
            "history_direction", "override_permission", "override_applied", "safety_blocked")}


def day_bias(prefix):
    x = finite_curve(prefix, "D-1 spread prefix", 56)
    count = float((x > 0).mean()-.5)
    energy = .5*float(x.sum()/(np.abs(x).sum()+1e-9))
    energy32 = .5*float(x[:32].sum()/(np.abs(x[:32]).sum()+1e-9))
    agree = bool(np.sign(count) == np.sign(energy))
    states = [energy32, energy, energy] if agree else [count]*3
    return states, dict(count56=count, energy56=energy, energy32=energy32, sign_agreement=agree)


def decide_direction(p_positive, spread_median, context, config=None):
    config = config or DirectionDecisionConfig()
    p = finite_curve(p_positive, "predictor p_positive")
    median = finite_curve(spread_median, "predictor spread median")
    if ((p < 0) | (p > 1)).any():
        raise ContractError("predictor probabilities outside [0,1]")
    for value in (context.history_direction, context.override_permission):
        if np.asarray(value).shape != (96,) or np.asarray(value).dtype != np.bool_:
            raise ContractError("history direction/permission must be Boolean [96]")
    states, bias = day_bias(context.prefix_spread)
    threshold = np.empty(96, dtype=float)
    for (_, lo, hi), multiplier, state in zip(SEGMENTS, config.segment_multipliers, states):
        threshold[lo:hi] = .5 + np.clip(-multiplier*state, -config.threshold_cap, config.threshold_cap)
    base = p > .5
    biased = p > threshold
    history = context.history_direction.copy()
    permission = context.override_permission.copy()
    eligible = permission & (p >= config.fuzzy_band[0]) & (p <= config.fuzzy_band[1])
    blocked = np.zeros(96, dtype=bool)
    for (_, lo, hi), safety in zip(SEGMENTS, config.safety_medians):
        if safety is not None:
            blocked[lo:hi] = eligible[lo:hi] & base[lo:hi] & ~history[lo:hi] & (median[lo:hi] >= safety)
    applied = eligible & ~blocked
    final = biased.copy()
    final[applied] = history[applied]
    trace = dict(target_day=context.target_day, policy_name=config.name, policy_version=config.version,
        policy_config=config.to_dict(), policy_config_hash=config.config_hash,
        selected_history_expert=context.selected_expert, selected_expert_reliability=context.reliability,
        expert_reliabilities=context.expert_reliabilities, bias=bias,
        segment_bias_states=states, provenance=context.provenance, numeric_output_unchanged=True)
    result = DirectionDecisionResult(base, biased, final, threshold, history, permission, applied, blocked, trace)
    for a in result.arrays().values():
        a.setflags(write=False)
    return result


def direction_decision(p_positive, spread_median, target, store):
    """Predictor-agnostic adapter; no Torch or model import."""
    return decide_direction(p_positive, spread_median, history_context(store, target))
