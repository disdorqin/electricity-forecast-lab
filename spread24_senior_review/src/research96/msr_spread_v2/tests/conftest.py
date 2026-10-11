"""Consumed/synthetic-only fixtures; no production or sealed truth."""
from datetime import date,timedelta
import numpy as np
import pytest
from ..dataset import CurveStore
from ..feature_registry import GRID

@pytest.fixture
def store():
    rng = np.random.default_rng(20261009)
    values = {}
    start = date(2022,12,24)
    for i in range(85):
        d = start+timedelta(days=i)
        t = np.arange(96)
        da = 300+40*np.sin(t/13)+i*.2+rng.normal(0,3,96)
        spread = 20*np.cos(t/11+i/5)+rng.normal(0,10,96)
        values[d,"da"],values[d,"rt"] = da,da+spread
        for j,k in enumerate(GRID):
            curve = 1000+500*j+100*np.sin(t/10+j)+rng.normal(0,15,96)
            values[d,"f_"+k],values[d,"a_"+k] = curve,curve+rng.normal(0,35,96)
        values[d,"a_psh"] = rng.normal(0,50,96)
    return CurveStore(values,dict(kind="SYNTHETIC_ENGINEERING_ONLY",seed=20261009))
