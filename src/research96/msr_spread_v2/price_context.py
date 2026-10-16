"""D-1 price suffix is context/proxy only; suffix truth is never accepted."""
import numpy as np
from .contracts import finite_curve

def build_price_context(da_d1, rt_prefix56, spread_d2):
    da = finite_curve(da_d1, "DA_D1")
    rt = finite_curve(rt_prefix56, "observed RT prefix", 56)
    previous = finite_curve(spread_d2, "mature Spread D2")
    spread = np.concatenate((rt - da[:56], previous[56:]))
    rt_proxy = np.concatenate((rt, da[56:] + spread[56:]))
    observed = (np.arange(96) < 56).astype(float)
    return dict(d1_da=da.copy(), d1_rt_proxy=rt_proxy, d1_spread_proxy=spread,
                price_observed_mask=observed, price_proxy_mask=1-observed,
                price_boundary_distance=(np.arange(1,97)-56)/96)
