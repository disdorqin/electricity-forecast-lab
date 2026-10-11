from src.research96.msr_spread_v2.config import Config
from src.research96.msr_spread_v2.experiments.component_adjudication.profile import (
    ARMS, BOUNDS, FloorSelector, scientific_config)

def test_no_pre_floor_checkpoint_or_patience():
    s=FloorSelector(15)
    for e in range(1,15):
        assert not s.consider(e,1/e)
        assert s.epoch==0 and s.bad==0 and not s.stop
    assert s.consider(15,100)
    assert s.epoch==15

def test_patience_starts_after_floor():
    s=FloorSelector(5)
    for e in range(1,13):
        s.consider(e,1)
        assert s.stop == (e==12)
    assert s.epoch==5

def test_later_improvement_resets_patience():
    s=FloorSelector(10)
    for e in range(1,30):
        s.consider(e,1/e)
        assert not s.stop
    assert s.epoch==29

def test_guard_rejects_checkpoint_but_keeps_floor():
    s=FloorSelector(10)
    assert s.consider(10,10)
    assert not s.consider(11,1,guard=False)
    assert s.epoch==10 and s.bad==1

def test_profile_identical_for_every_scientific_arm():
    baseline=scientific_config(64).to_dict()
    assert all(scientific_config(64,arm).to_dict()==baseline for arm in ARMS)
    assert BOUNDS==(("A",15,40),("B",5,12),("C",10,30))
    assert baseline['seed']==20261009 and baseline['patience']==7
    assert baseline['validation_days']==28 and baseline['m_train']==32 and baseline['m_eval']==256

def test_core_defaults_untouched():
    cfg=Config()
    assert (cfg.epochs_a,cfg.epochs_b,cfg.epochs_c,cfg.patience)==(30,8,20,5)
