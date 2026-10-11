import numpy as np
from src.research96.msr_spread_v2.experiments.component_adjudication.metrics import pooled,promotion

def record(truth):
    n=len(truth)
    return dict(truth=np.stack((truth,np.zeros(n),truth),axis=-1),
        out=dict(spread_scenarios=np.zeros((1,256,n)),spread_median=np.zeros((1,n)),
            spread_quantiles=np.zeros((1,n,9)),p_positive=np.zeros((1,n))))

def test_pooled_balanced_not_average_of_daily_balanced():
    a=record(-np.ones(96));b=record(np.r_[np.ones(48),-np.ones(48)])
    result=pooled([a,b])
    assert result['balanced_accuracy']==.5
    assert (pooled([a])['balanced_accuracy']+pooled([b])['balanced_accuracy'])/2==.75
    assert result['raw_direction']==.75 and result['all_negative_baseline']==.75
    assert result['positive_recall']==0 and result['negative_recall']==1
    assert result['brier0']==.25

def test_promotions_symmetric():
    baseline=dict(balanced_accuracy=.5,raw_direction=.5,brier0=.2,spread_mae=100.,crps=80.)
    for change in (-.04,.04):
        alternative=dict(baseline,spread_mae=100*(1+change))
        assert 'spread_mae' in promotion(baseline,alternative)
    assert not promotion(baseline,baseline)

def test_empty_panel_not_fabricated():
    assert pooled([]) is None and promotion(None,None)==[]
