"""
Picklable conditional logit for the rivals' compound choice (K1, 2026-10-15).

Estimated with statsmodels ConditionalLogit on 2021-23 pit stops (one group per stop, alternatives = the
compounds available at that Year-GP), as in results/rival_model_eval_20261014/rivals_common_20261014.py (K1):
    U_c = nu_c + nu_{c,LapsLeft} * LapsLeft + nu_NoChange * not_change_compound,   reference C3 (nu_C3 = 0).
GP effects common to all alternatives cancel in a conditional logit, so there is none.

Interface consumed by the simulator (src/environment/f1_env_all_drivers.py, _predict_compound_choice):
  - .params: pd.Series with 'C(alt)[Cx]', 'C(alt)[Cx]:LapsLeft' and 'not_change_compound';
  - .predict(df): rows = the alternatives of ONE decision (columns alt, LapsLeft, not_change_compound);
    returns the softmax probabilities (they already sum to 1).
The environment's fast path assumes a normalized binary logit; its probe check fails for this model and the
environment keeps the slow path, which calls predict() and normalizes (a no-op here). No simulator change needed.
Defined at module level so the pickles can be loaded from any process with src/ on sys.path.
"""
import numpy as np
import pandas as pd


class CondLogitCompoundView:
    choice_rule = 'softmax'

    def __init__(self, params, bse=None):
        self.params = params
        self.bse = bse

    def utility(self, df):
        p = self.params
        alt = df['alt'].astype(str).values
        base = np.array([p.get(f'C(alt)[{a}]', np.nan) for a in alt], dtype=float)
        ll = np.array([p.get(f'C(alt)[{a}]:LapsLeft', 0.0) for a in alt], dtype=float)
        return (base + ll * np.asarray(df['LapsLeft'], dtype=float)
                + p.get('not_change_compound', 0.0) * np.asarray(df['not_change_compound'], dtype=float))

    def predict(self, df):
        v = self.utility(df)
        e = np.exp(v - np.nanmax(v))
        return e / np.nansum(e)
