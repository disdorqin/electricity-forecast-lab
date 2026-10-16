"""Explicit single-family schema; unknown fields never enter the tensor."""
from .config import FAMILIES, BYPASS
from .contracts import ContractError

# Source naming/semantics copied locally from V1/data_contract.py (read-only).
GRID = {
    "load": ("DEMAND","直调负荷实际","直调负荷预测"),
    "load_total": ("DEMAND","全网负荷实际","全网负荷预测"),
    "wind": ("RENEWABLE","风电实际","风电预测"),
    "solar": ("RENEWABLE","光伏实际","光伏预测"),
    "local": ("SUPPLY_SYSTEM","地方电厂出力实际","地方电厂出力预测"),
    "import": ("SUPPLY_SYSTEM","外电实际","外电预测"),
    "nuclear": ("SUPPLY_SYSTEM","核电实际","核电预测"),
    "self": ("SUPPLY_SYSTEM","自备电厂实际","自备电厂预测"),
    "test": ("SUPPLY_SYSTEM","试验机组实际","试验机组预测"),
}
CORE_GRID = ("load","wind","solar","local","import")
RAMPS = ("ramp15","ramp30","ramp60","curvature")
CALENDAR = ("slot","quarter","hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos")
PRICE_CONTEXT = ("d1_da","d1_rt_proxy","d1_spread_proxy")
MASKS = ("price_observed_mask","price_proxy_mask","price_boundary_distance")
PRICE_STATS = ("prefix_mean","prefix_std","prefix_last","prefix_ramp","target_da_mean","target_da_std")
DERIVED = {"renewable": "RENEWABLE", "renewable_share": "RENEWABLE",
           "residual_load": "DEMAND", "load_gap": "DEMAND",
           "bidding_space": "MARKET_TIGHTNESS", "space_ratio": "MARKET_TIGHTNESS"}

def registry():
    out = {}
    def add(name, family, role, lag=0, source=None):
        if name in out:
            raise ContractError("duplicate field registration")
        out[name] = dict(family=family, role=role, lag_days=lag, source=source)
    for lag in range(2,9):
        for field in ("da","rt","spread","low","high"):
            role = {"low":"low_anchor","high":"high_anchor"}.get(field,"spectral")
            add(f"hist_{field}_{lag}","PRICE",role,lag)
    for n in PRICE_CONTEXT + ("target_da",) + PRICE_STATS:
        add(n,"PRICE","spectral",1 if n.startswith(("d1","prefix")) else 0)
    for n in MASKS + CALENDAR:
        add(n,BYPASS,"bypass")
    for r in RAMPS:
        add(f"target_da_{r}","PRICE","spectral")
    for key, (fam, actual, forecast) in GRID.items():
        for prefix,src,lag in (("d1_actual",actual,1),("target_forecast",forecast,0),
                               ("d1_error",actual+" - "+forecast,1),("history_mean",actual,2)):
            name = f"{prefix}_{key}"
            add(name,fam,"spectral",lag,src)
            add("mask_"+name,BYPASS,"bypass",lag,src)
        for ramp in RAMPS:
            add(f"target_forecast_{key}_{ramp}",fam,"spectral",0,forecast)
    add("d1_actual_psh","SUPPLY_SYSTEM","spectral",1,"抽蓄实际")
    add("mask_d1_actual_psh",BYPASS,"bypass",1,"抽蓄实际")
    for prefix in ("target","d1"):
        for key,fam in DERIVED.items():
            add(f"{prefix}_{key}",fam,"spectral",int(prefix=="d1"))
            add(f"mask_{prefix}_{key}",BYPASS,"bypass")
        for ramp in RAMPS:
            add(f"{prefix}_bidding_space_{ramp}","MARKET_TIGHTNESS","spectral",int(prefix=="d1"))
    return out

def family_of(name):
    try:
        return registry()[name]["family"]
    except KeyError as e:
        raise ContractError(f"unregistered field {name}") from e

def indices(family, names, role=None):
    reg = registry()
    return [i for i,n in enumerate(names) if reg[n]["family"]==family and
            (reg[n]["role"]==role if role else reg[n]["role"] not in ("low_anchor","high_anchor"))]
