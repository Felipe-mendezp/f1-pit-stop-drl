# (A)+(C solo offsets) SIN monotonia en las pendientes (se mantiene positividad y monotonia de offsets)
from mc_ACf import *
import mc_ACf as _base
_orig = _base.check_compatibility
def check_compatibility(d, c, v, fixed=True):
    if c in d:
        raise ValueError(f"Compound {c} is already in the dictionary")
    return _orig(d, c, v, fixed) if fixed else True
_base.check_compatibility = check_compatibility
fun_real_data_v2 = _base.fun_real_data_v2
