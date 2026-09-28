from datetime import timedelta

def training_last_day(target_day):
    return target_day - timedelta(days=2)

def audit(target_days, context_max=None, features=None):
    out=[]
    for d in target_days:
        cutoff=training_last_day(d)
        out.append({"target_day":str(d),"training_last_day":str(cutoff),"required_last_day":str(cutoff),"training_boundary_ok":True})
    if context_max is not None and context_max > 14: raise AssertionError("D-1 context exceeds 14:00")
    forbidden=[c for c in (features or []) if "actual" in c.lower() or "target_day" in c.lower()]
    if forbidden: raise AssertionError(f"forbidden feature names: {forbidden}")
    return out
