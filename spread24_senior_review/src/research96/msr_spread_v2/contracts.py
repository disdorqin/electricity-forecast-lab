"""Forecast timeline: D-8..D-2 mature; D-1 RT<=p56; D DA/forecast.
D-1 full physical actuals and D DA are docs/84 research assumptions, not live proof.
"""
from datetime import date, timedelta
import numpy as np
from .config import PACKAGE_ROOT, OUTPUT_ROOT

class ContractError(ValueError):
    pass

def day(value):
    return date.fromisoformat(str(value)[:10])

def allowed_day(value):
    d = day(value)
    if not date(2023, 1, 1) <= d <= date(2026, 8, 14):
        raise ContractError("outside consumed implementation scope; fresh/October sealed")
    return d

def label_cutoff(target):
    return allowed_day(target) - timedelta(days=2)

def assert_mature(labels, target):
    if not labels or any(day(d) > label_cutoff(target) or day(d) < date(2023,1,1) for d in labels):
        raise ContractError("fit labels must lie in 2023-01-01..D-2")

def assert_write_path(path):
    from pathlib import Path
    import os
    raw = os.fspath(path)
    # Normalize Windows extended spelling BEFORE containment checks; never bypass guard.
    if os.name=="nt" and raw.startswith("\\\\?\\"):
        raw = raw[4:]
    p = Path(raw).resolve()
    if not any(p.is_relative_to(r.resolve()) for r in (PACKAGE_ROOT, OUTPUT_ROOT)):
        raise ContractError("write outside V2 roots")
    # The deep Chinese workspace exceeds legacy MAX_PATH for actual smoke artifacts.
    return Path("\\\\?\\"+str(p)) if os.name=="nt" and len(str(p))>=240 else p

def finite_curve(x, name, length=96):
    a = np.asarray(x, dtype=np.float64)
    if a.shape != (length,) or not np.isfinite(a).all():
        raise ContractError(f"{name}: expected finite [{length}]")
    return a

def assert_feature_names(names):
    from .feature_registry import registry
    reg = registry()
    if len(names) != len(set(names)) or any(n not in reg for n in names):
        raise ContractError("unknown/duplicate/forbidden feature; fail closed")
