import numpy as np
import torch
from src.research96.msr_spread_v2.experiments.component_adjudication.low_diagnosis import (
    scenario_record,error_anatomy,context_hook,structure)
from src.research96.msr_spread_v2.experiments.component_adjudication.metrics import pooled

def test_oracle_identity_perfect_but_not_deployable():
    s=np.r_[np.ones(48),-np.ones(48)]
    y=np.stack((s,s*.4,s*.6),axis=-1)
    r=scenario_record('2026-01-05',np.broadcast_to(y[:,1]+y[:,2],(256,96)),y)
    m=pooled([r]);assert m['spread_mae']==0 and m['raw_direction']==1 and m['brier0']==0

def test_daily_level_shape_error_identity():
    y=np.arange(96,dtype=float);pred=y+7+np.sin(y)
    e=error_anatomy(pred,y)
    assert np.isclose(e['mse'],e['bias']**2+e['centered_mse'])

def inputs():
    x=torch.randn(1,96,12);x[...,7]=2 # Low median at last7 offset2
    return x,torch.ones(1,96,1),torch.ones(1,96,1),torch.zeros(1,96,1)

def test_oracle_location_retains_width_and_noncontext():
    args=inputs();old=args[0].clone();target=torch.full((1,96),9.)
    new=context_hook('ORACLE_LOCATION_CONTEXT',target)(None,args)
    assert torch.equal(args[0],old) and torch.equal(new[0][...,:5],old[...,:5])
    assert torch.equal(new[0][...,-2:],old[...,-2:]) and torch.equal(new[1],args[1])
    assert torch.equal(new[0][...,7],target)

def test_zero_context_only_seven_columns_and_fusion_width():
    args=inputs();new=context_hook('ZERO_CONTEXT',torch.zeros(1,96))(None,args)
    assert torch.equal(new[0][...,:5],args[0][...,:5])
    assert torch.count_nonzero(new[0][...,-7:])==0 and torch.count_nonzero(new[1])==0

def test_degenerate_true_summary_no_oracle_latent():
    args=inputs();target=torch.full((1,96),-4.)
    new=context_hook('ORACLE_DEGENERATE_CONTEXT',target)(None,args)
    assert torch.equal(new[0][...,-7:-2],target[...,None].expand(-1,-1,5))
    assert torch.count_nonzero(new[0][...,-2:])==0 and torch.count_nonzero(new[1])==0

def test_predicted_location_without_uncertainty():
    args=inputs();new=context_hook('PRED_LOCATION_ZERO_WIDTH',torch.zeros(1,96))(None,args)
    assert torch.equal(new[0][...,-7:-2],args[0][...,7:8].expand(-1,-1,5))
    assert torch.count_nonzero(new[0][...,-2:])==0

def test_constant_curve_has_no_ac_energy():
    assert structure(np.zeros(96))['ac_energy_bin8_plus']==0
