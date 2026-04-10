"""
Compatibility helpers for loading pickled models across numpy / Python versions.
Import this module before any pickle.load() call on trained models.
"""
import sys
import types
import numpy as np


def setup_numpy_compat():
    """Numpy 2.x / 1.x compatibility shim for unpickling models."""
    if not hasattr(np, "_core"):
        _core = types.ModuleType("numpy._core")
        _core.numeric = np.core.numeric
        _core.multiarray = np.core.multiarray
        sys.modules["numpy._core"] = _core
        sys.modules["numpy._core.numeric"] = np.core.numeric
        sys.modules["numpy._core.multiarray"] = np.core.multiarray


def setup_pickle_compat():
    """Ensure FastLinearPredictor is discoverable by pickle.load()."""
    from reg import FastLinearPredictor

    if "__main__" in sys.modules:
        sys.modules["__main__"].FastLinearPredictor = FastLinearPredictor


def setup_all():
    """Run all compatibility setups."""
    setup_numpy_compat()
    setup_pickle_compat()
