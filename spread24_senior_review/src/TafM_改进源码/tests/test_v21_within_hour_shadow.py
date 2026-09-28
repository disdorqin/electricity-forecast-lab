import numpy as np
from src.TafM_改进源码.selector import within_hour_shadow


def test_shadow_permutation_preserves_each_hour_marginal_and_does_not_mix_hours():
    hours=np.repeat(np.arange(1,25),5)
    day=np.tile(np.arange(5),24)
    x=np.column_stack([hours*100+day,day*1000+hours]).astype(np.int64)
    shadow=within_hour_shadow(x,hours,np.random.default_rng(44))
    for h in range(1,25):
        idx=hours==h
        for j in range(x.shape[1]):
            assert sorted(shadow[idx,j].tolist())==sorted(x[idx,j].tolist())
        assert np.all(shadow[idx,0]//100==h)
