from mc_ACf import *
import mc_ACf as _base
_orig = _base.check_compatibility
def check_compatibility(d, c, v, fixed=True):
    if c in d: raise ValueError('dup')
    return True if fixed else _orig(d, c, v, fixed)
_base.check_compatibility = check_compatibility
fun_real_data_v2 = _base.fun_real_data_v2
