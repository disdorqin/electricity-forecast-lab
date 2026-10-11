from datetime import date,timedelta
from pathlib import Path
import hashlib
import numpy as np
import pytest
from ..config import PACKAGE_ROOT, OUTPUT_ROOT, T24_SHA256
from ..contracts import ContractError, assert_write_path, assert_mature, allowed_day, assert_feature_names
from ..dataset import build_sample, true_label
from ..price_context import build_price_context
from ..feature_registry import registry, family_of, GRID
from ..t24 import operator, decompose

def test_T01_write_guard():
    assert assert_write_path(PACKAGE_ROOT/"tests/.tmp") == (PACKAGE_ROOT/"tests/.tmp").resolve()
    assert_write_path(OUTPUT_ROOT/"test")
    for p in (PACKAGE_ROOT.parent/"msr_spread_v1/x",PACKAGE_ROOT.parents[2]/"src/TafM_改进源码/x",
              PACKAGE_ROOT.parents[4]/"electricity_forecast_model2.5/x",OUTPUT_ROOT/"../../escape"):
        with pytest.raises(ContractError):
            assert_write_path(p)

def test_T01_windows_long_artifact_guard(tmp_path):
    from ..artifacts import write_json
    long_path = tmp_path/("a"*80)/("b"*60)/"GUARD_ROUNDTRIP.json"
    guarded = assert_write_path(long_path)
    assert assert_write_path(guarded)==guarded
    write_json(guarded,dict(status="PASS"))
    assert guarded.is_file()
    with pytest.raises(ContractError):
        assert_write_path(guarded.parent/("/".join([".."]*12))/"escape")


def test_T02_maturity_and_sealed_scope():
    assert_mature(["2026-01-01","2026-01-03"],"2026-01-05")
    with pytest.raises(ContractError):
        assert_mature(["2026-01-04"],"2026-01-05")
    for d in ("2026-09-16","2026-10-01"):
        with pytest.raises(ContractError):
            allowed_day(d)

def test_T03_target_da_and_truth_mutation(store):
    d = date(2023,2,15)
    s = build_sample(store,d)
    assert_feature_names(s.names)
    assert np.array_equal(s.x[:,s.names.index("target_da")],store.values[d,"da"])
    for k in ("rt",)+tuple("a_"+k for k in GRID):
        store.values[d,k] *= 1000
    after = build_sample(store,d)
    np.testing.assert_equal(s.x,after.x)
    for name in ("RT_D","Spread_D","target_actual_load","H_true_D"):
        with pytest.raises(ContractError):
            assert_feature_names([name])

def test_T04_suffix_never_read_or_eligible(store):
    d = date(2023,2,15)
    s = build_sample(store,d)
    store.values[d-timedelta(days=1),"rt"][56:] = np.nan
    a = build_sample(store,d)
    np.testing.assert_equal(s.x,a.x)
    store.values[d-timedelta(days=1),"rt"][56:] = 1e20
    np.testing.assert_equal(s.x,build_sample(store,d).x)
    accesses = [v for v in store.access_log if v[:2]==((d-timedelta(days=1)).isoformat(),"rt")]
    assert all(v[2]==56 for v in accesses)
    ctx = build_price_context(np.arange(96.),np.arange(56.)+3,np.full(96,7.))
    assert np.array_equal(ctx["d1_spread_proxy"][:56],np.full(56,3.))
    assert np.array_equal(ctx["d1_spread_proxy"][56:],np.full(40,7.))
    assert np.array_equal(ctx["d1_rt_proxy"],ctx["d1_da"]+ctx["d1_spread_proxy"])
    assert ctx["price_observed_mask"].sum()==56
    with pytest.raises(ContractError):
        build_price_context(np.zeros(96),np.zeros(96),np.zeros(96))

def test_T05_grid_full96(store):
    d = date(2023,2,15)
    s = build_sample(store,d)
    for key in GRID:
        np.testing.assert_equal(s.x[:,s.names.index("d1_actual_"+key)],store.values[d-timedelta(days=1),"a_"+key])
        np.testing.assert_equal(s.x[:,s.names.index("target_forecast_"+key)],store.values[d,"f_"+key])
    store.values[d-timedelta(days=1),"a_load"][90] += 100
    changed = build_sample(store,d)
    assert changed.x[90,changed.names.index("d1_actual_load")] != s.x[90,s.names.index("d1_actual_load")]

def test_T06_frozen_t24():
    a = operator()
    assert hashlib.sha256(a.tobytes()).hexdigest()==T24_SHA256
    d2 = np.diff(np.eye(96),n=2,axis=0)
    assert not d2[:,[0,95]].prod(axis=1).any()
    np.testing.assert_allclose((np.eye(96)+215.32246474604284*(d2.T@d2))@a,np.eye(96),atol=1e-12)
    s = np.random.default_rng(7).normal(0,200,(5,96))
    l,h = decompose(s)
    assert np.max(np.abs(s-l-h))<=1e-10

def test_T07_registry_fail_closed(store):
    reg = registry()
    assert set(v["family"] for v in reg.values())=={"PRICE","DEMAND","RENEWABLE","SUPPLY_SYSTEM","MARKET_TIGHTNESS","CALENDAR_MASK"}
    assert len(reg)==len(set(reg))
    assert family_of("price_proxy_mask")=="CALENDAR_MASK"
    with pytest.raises(ContractError):
        family_of("new_unverified_field")
    assert set(build_sample(store,"2023-02-15").names)==set(reg)

def test_T10_only_mature_residual_history(store):
    s = build_sample(store,"2023-02-15")
    assert [n for n in s.names if n.startswith("hist_high_")]==[f"hist_high_{i}" for i in range(2,9)]
    for i in range(2,9):
        d = date(2023,2,15)-timedelta(days=i)
        h = true_label(store,d)[:,2]
        np.testing.assert_equal(s.x[:,s.names.index(f"hist_high_{i}")],h)
    assert "d1_high_true" not in s.names and "H_true_D" not in s.names
