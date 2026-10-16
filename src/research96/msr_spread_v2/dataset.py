"""Native-96 deterministic tensorization. Labels never enter feature API.
No V1/production runtime imports. Reader projects and date-filters before materialization.
"""
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from .config import SOURCE_PATH, SOURCE_SHA256
from .contracts import allowed_day, day, label_cutoff, assert_mature, ContractError, finite_curve
from .feature_registry import GRID, CORE_GRID, CALENDAR, MASKS, RAMPS, registry
from .price_context import build_price_context
from .t24 import decompose

def slot_numbers(values):
    # Canonical business-ending clocks 00:15..24:00; never infer slot from row count.
    numbers = []
    for value in values:
        text = str(value)
        if ":" in text:
            h,m = map(int,text.split(":")[:2])
            if m not in (0,15,30,45):
                raise ContractError("non-quarter source clock")
            numbers.append((h*60+m)//15)
        else:
            numbers.append(int(text))
    return np.asarray(numbers,dtype=int)

def ramp(x, steps):
    # Natural line boundary: no circular predecessor; unavailable predecessors -> zero CHANGE.
    out = np.zeros_like(x)
    out[steps:] = x[steps:] - x[:-steps]
    return out

def add_ramps(out, name, x):
    r = ramp(x,1)
    for tag, n in (("ramp15",1),("ramp30",2),("ramp60",4)):
        out[name+"_"+tag] = ramp(x,n)
    out[name+"_curvature"] = ramp(r,1)

class CurveStore:
    """Canonical day/field curves with legal access tracing; RT prefix sliced first."""
    def __init__(self, values, source_identity):
        self.values = values
        self.source_identity = source_identity
        self.access_log = []

    def get(self, d, field, prefix=None):
        d = day(d)
        self.access_log.append((d.isoformat(),field,prefix))
        a = self.values.get((d,field), np.full(96,np.nan))
        # Slice before conversion, finiteness/eligibility checks and summaries.
        return np.asarray(a[:prefix] if prefix else a,dtype=np.float64).copy()

    @classmethod
    def canonical(cls, max_target, path=SOURCE_PATH):
        target = allowed_day(max_target)
        path = __import__("pathlib").Path(path).resolve()
        from .config import LAB_ROOT
        if not path.is_relative_to((LAB_ROOT/"data").resolve()):
            raise ContractError("source must be an immutable lab-local data snapshot")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != SOURCE_SHA256:
            raise ContractError("canonical source hash drift")
        table = ds.dataset(path,format="parquet")
        values = {}
        import datetime as dt
        def projection(columns, end, prefix=None):
            filt = ((ds.field("market_date") >= dt.date(2022,12,24)) &
                    (ds.field("market_date") <= end))
            if prefix:
                filt = filt & (ds.field("时段") <= "14:00")
            frame = table.to_table(columns=["market_date","时段",*columns.values()],filter=filt).to_pandas()
            if frame.duplicated(["market_date","时段"]).any():
                raise ContractError("duplicate source day/slot")
            for date,group in frame.groupby("market_date",sort=True):
                slots = slot_numbers(group["时段"])-1
                if (slots<0).any() or (slots>=96).any():
                    raise ContractError("invalid native source slots")
                for key,col in columns.items():
                    a = values.setdefault((day(date),key),np.full(96,np.nan))
                    a[slots] = pd.to_numeric(group[col],errors="coerce").to_numpy(float)
        projection({"da":"日前出清价格",**{"f_"+k:v[2] for k,v in GRID.items()}},target)
        projection({**{"a_"+k:v[1] for k,v in GRID.items()},"a_psh":"抽蓄实际"},target-timedelta(days=1))
        projection({"rt":"实时出清价格"},target-timedelta(days=2))
        projection({"rt":"实时出清价格"},target-timedelta(days=1),56)
        return cls(values,dict(path=str(path),sha256=sha,scope_end=target.isoformat(),
                              price_full_end=(target-timedelta(days=2)).isoformat(),
                              vintage="LEGACY-UNVERIFIED; PROXY/EXPERIMENTAL V2 assumptions"))

@dataclass
class Sample:
    target_day: str
    x: np.ndarray
    names: tuple
    da: np.ndarray
    provenance: dict

def build_sample(store, target):
    d = allowed_day(target)
    out = {}
    history = {}
    for lag in range(2,9):
        ld = d-timedelta(days=lag)
        da = finite_curve(store.get(ld,"da"),f"DA lag{lag}")
        rt = finite_curve(store.get(ld,"rt"),f"RT lag{lag}")
        s = rt-da
        low,high = decompose(s)
        history[lag] = s
        for name,curve in (("da",da),("rt",rt),("spread",s),("low",low),("high",high)):
            out[f"hist_{name}_{lag}"] = curve
    previous = d-timedelta(days=1)
    ctx = build_price_context(store.get(previous,"da"),store.get(previous,"rt",56),history[2])
    out.update(ctx)
    da = finite_curve(store.get(d,"da"),"full target DA")
    out["target_da"] = da
    add_ramps(out,"target_da",da)
    prefix = ctx["d1_spread_proxy"][:56]
    for name,v in (("mean",prefix.mean()),("std",prefix.std()),("last",prefix[-1]),
                   ("ramp",prefix[-1]-prefix[-2])):
        out["prefix_"+name] = np.full(96,v)
    out["target_da_mean"] = np.full(96,da.mean())
    out["target_da_std"] = np.full(96,da.std())
    physical = {"target":{},"d1":{}}
    def put(name,x):
        out[name] = x
        out["mask_"+name] = np.isfinite(x).astype(float)
    for key,(_,_,_) in GRID.items():
        a1 = store.get(previous,"a_"+key)
        fd = store.get(d,"f_"+key)
        f1 = store.get(previous,"f_"+key)
        if key in CORE_GRID:
            finite_curve(a1,"D1 actual "+key)
            finite_curve(fd,"D forecast "+key)
        physical["d1"][key],physical["target"][key] = a1,fd
        put("d1_actual_"+key,a1)
        put("target_forecast_"+key,fd)
        put("d1_error_"+key,a1-f1)
        hist = np.stack([store.get(d-timedelta(days=l),"a_"+key) for l in range(2,9)])
        count = np.isfinite(hist).sum(0)
        avg = np.divide(np.nansum(hist,0),count,out=np.full(96,np.nan),where=count>0)
        put("history_mean_"+key,avg)
        add_ramps(out,"target_forecast_"+key,fd)
    put("d1_actual_psh",store.get(previous,"a_psh"))
    for tag,p in physical.items():
        renewable = p["wind"]+p["solar"]
        space = p["load"]-renewable-p["import"]-p["nuclear"]
        for key,arr in (("renewable",renewable),
                        ("renewable_share",renewable/(np.abs(p["load"])+1)),
                        ("residual_load",p["load"]-renewable),
                        ("load_gap",p["load_total"]-p["load"]),
                        ("bidding_space",space),
                        ("space_ratio",space/(np.abs(p["load"])+1))):
            put(tag+"_"+key,arr)
        add_ramps(out,tag+"_bidding_space",space)
    t = np.arange(96)
    cal = (t/95,(t%4)/3,np.sin(2*np.pi*t/96),np.cos(2*np.pi*t/96),
           np.full(96,np.sin(2*np.pi*d.weekday()/7)),np.full(96,np.cos(2*np.pi*d.weekday()/7)),
           np.full(96,np.sin(2*np.pi*(d.month-1)/12)),np.full(96,np.cos(2*np.pi*(d.month-1)/12)))
    out.update(dict(zip(CALENDAR,cal)))
    names = tuple(registry())
    if set(out) != set(names):
        raise ContractError(f"tensor registry mismatch: missing={set(names)-set(out)}, extra={set(out)-set(names)}")
    return Sample(d.isoformat(),np.stack([out[n] for n in names],axis=-1),names,da,
                  dict(origin=(d-timedelta(days=1)).isoformat()+" 14:00 Asia/Shanghai",
                       label_cutoff=label_cutoff(d).isoformat(),anchor_lags=list(range(2,9)),
                       target_da_full96=True,d1_price_proxy_suffix=True,
                       grid_full_d1_actual_research_assumption=True))

def true_label(store, target):
    # Separate supervisory/evaluation API: never called by build_sample/eligibility.
    d = allowed_day(target)
    s = finite_curve(store.get(d,"rt"),"RT label")-finite_curve(store.get(d,"da"),"DA label")
    l,h = decompose(s)
    return np.stack((s,l,h),axis=-1)

def evaluation_label(target, path=SOURCE_PATH):
    # Must be called after prediction; exact date/price projection, no physical target actuals.
    import datetime as dt
    d = allowed_day(target)
    frame = ds.dataset(path,format="parquet").to_table(
        columns=["时段","日前出清价格","实时出清价格"],
        filter=ds.field("market_date")==d).to_pandas().sort_values("时段")
    if list(slot_numbers(frame["时段"])) != list(range(1,97)):
        raise ContractError("evaluation requires exactly 96 source slots")
    s = finite_curve(frame["实时出清价格"].to_numpy(),"RT evaluation")-finite_curve(
        frame["日前出清价格"].to_numpy(),"DA evaluation")
    l,h = decompose(s)
    return np.stack((s,l,h),axis=-1)

def training_panel(store, target, smoke=False):
    end = label_cutoff(target)
    start = __import__("datetime").date(2023,1,1)
    # Explicit debugging-only truncation; scientific profile always expanding.
    if smoke:
        start = max(start,end-timedelta(days=45))
    samples,labels,rejected = [],[],[]
    current = start
    while current <= end:
        try:
            sample = build_sample(store,current)
            y = true_label(store,current)
            samples.append(sample); labels.append(y)
        except ContractError as e:
            rejected.append(dict(day=current.isoformat(),reason=str(e)))
        current += timedelta(days=1)
    assert_mature([s.target_day for s in samples],target)
    return samples,np.stack(labels),rejected
