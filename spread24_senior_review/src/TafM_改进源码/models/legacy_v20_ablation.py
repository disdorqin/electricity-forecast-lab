"""Explicit opt-in V2.0 Gate-A compatibility runner; never used by default pipeline."""
from ..train import train_target_day


def run_legacy_v20_target_day(target_day, *, mode="A2",profile="smoke",**kwargs):
    return train_target_day(target_day,mode=mode,profile=profile,train_mode="stage_a",legacy_v20=True,**kwargs)
