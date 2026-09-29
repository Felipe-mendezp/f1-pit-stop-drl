"""
Compatibility helpers for loading pickled models across numpy / Python versions.
Import this module before any pickle.load() call on trained models.
"""
import sys
import types
import numpy as np


def setup_numpy_compat():
    """Numpy 2.x / 1.x compatibility shim for unpickling models.

    FIX 2026-08-11: el guard `if not hasattr(np, "_core")` era inseguro bajo
    numpy 1.x — numpy 1.26 trae un `numpy/_core/` PARCIAL (sin `numeric`) que
    torch/pandas pueblan al importarse, así que con orden de imports
    torch-primero el shim se convertía en no-op silencioso y los pickles
    creados con numpy 2 fallaban con `No module named 'numpy._core.numeric'`.
    Ahora se registran los alias que falten de forma incondicional (bajo
    numpy 2 los módulos reales ya existen en sys.modules y no se tocan).
    """
    if hasattr(np, "core") and not hasattr(getattr(np, "_core", None), "numeric"):
        _core = sys.modules.get("numpy._core") or types.ModuleType("numpy._core")
        if not hasattr(_core, "numeric"):
            _core.numeric = np.core.numeric
        if not hasattr(_core, "multiarray"):
            _core.multiarray = np.core.multiarray
        sys.modules["numpy._core"] = _core
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)


def setup_pickle_compat():
    """Ensure FastLinearPredictor is discoverable by pickle.load()."""
    from reg import FastLinearPredictor

    if "__main__" in sys.modules:
        sys.modules["__main__"].FastLinearPredictor = FastLinearPredictor


def setup_sklearn_compat():
    """
    scikit-learn >= 1.6 renamed check_array(force_all_finite=...) to
    ensure_all_finite; fancyimpute (SoftImpute) still uses the old name.
    Patch check_array BEFORE fancyimpute is imported so the old keyword
    keeps working.
    """
    try:
        import inspect
        import sklearn.utils
        import sklearn.utils.validation as _val

        if 'force_all_finite' in inspect.signature(_val.check_array).parameters:
            return  # old sklearn, nothing to patch
        if getattr(_val.check_array, '_compat_patched', False):
            return

        _orig_check_array = _val.check_array

        def check_array(*args, **kwargs):
            if 'force_all_finite' in kwargs:
                kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
            return _orig_check_array(*args, **kwargs)

        check_array._compat_patched = True
        _val.check_array = check_array
        sklearn.utils.check_array = check_array
    except ImportError:
        pass


def setup_all():
    """Run all compatibility setups."""
    setup_numpy_compat()
    setup_pickle_compat()
    setup_sklearn_compat()
