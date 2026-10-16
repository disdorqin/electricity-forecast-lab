"""Local immutable T24 adaptation of V1/decomposition.py natural_trend.
Authority docs/69,71,84; exact epf-2 NumPy 2.4.6 matrix frozen in t24_operator.npy.
No runtime V1 import, inversion/platform-dependent byte drift, or reselection.
"""
from functools import lru_cache
import hashlib
import numpy as np
from .config import PACKAGE_ROOT, T24_SHA256, T24_LAMBDA
from .contracts import ContractError

@lru_cache(maxsize=1)
def operator():
    a = np.load(PACKAGE_ROOT / "t24_operator.npy", allow_pickle=False)
    if a.shape != (96,96) or a.dtype != np.float64 or hashlib.sha256(a.tobytes()).hexdigest() != T24_SHA256:
        raise ContractError("frozen T24 matrix identity mismatch")
    a.setflags(write=False)
    return a

def decompose(spread):
    s = np.asarray(spread, dtype=np.float64)
    if s.shape[-1] != 96 or not np.isfinite(s).all():
        raise ContractError("T24 accepts complete finite true mature labels only")
    low = s @ operator().T
    return low, s-low
