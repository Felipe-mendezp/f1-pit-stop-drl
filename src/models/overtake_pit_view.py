"""
Picklable per-GP view of the pooled pit-aware overtake logit (2026-08-07).

The pooled fit (src/models/logit_overtakes_pit.py) estimates one logistic
regression over all GPs with GP-specific intercepts and GP-specific
delta_total slopes, on the RELABELED sample where position gains over a
pitting rival count as overtakes and delta_total extends to [-30, 5]. This
class exposes the single-GP restriction with the exact interface the
simulator already consumes:

  - .params: pd.Series indexed by ['const', 'delta_total'] (plus optional
    'pitout_prox' if the pooled term is ever re-enabled)
  - .predict(X): sigmoid(X[params.index] @ params)

Because the model stays linear in delta_total, the environment's 2-parameter
fast path (_calibrate_overtake_logit / _quick_tau_estimate) reproduces it
exactly, so no simulator code changes are required.

Defined at module level (NOT inside a function) so the per-GP pickles saved
to trained_models/simulation_pitlogit_*/<GP>/overtake_logit.pkl can be
unpickled from any process that has src/ on sys.path (model_loader does).
"""
import numpy as np


class PerGPLogitView:
    """Single-GP restriction of the pooled overtake logit."""

    def __init__(self, params, pvalues):
        self.params = params
        self.pvalues = pvalues

    def predict(self, X):
        X = X[self.params.index]
        linear = X.to_numpy() @ self.params.to_numpy()
        return 1 / (1 + np.exp(-linear))
