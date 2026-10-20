"""Medicion 2026-10-20 (temporal, no modifica el repo): simulador con TyreLife = convencion de los datos
(vuelta de salida = 1 en vez de 0). Suma F1_TL_OFFSET (def. 1) vueltas de vida a:
  - el tiempo de vuelta en pista libre de todos los pilotos (+ offset * coef TyreLife:C del compuesto),
  - el logit de parada de los rivales (d.tyrelife + offset durante el calculo de probabilidades).
La observacion del agente no cambia. Uso: python tl_plus1.py <script> <args...>"""
import os, sys, runpy
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts')]
import config, compat; compat.setup_all()
from environment import f1_env_all_drivers as M
OFF = float(os.environ.get('F1_TL_OFFSET', '1'))
Env = [v for v in vars(M).values() if isinstance(v, type) and hasattr(v, '_simulate_lap_times')][0]
_lt = Env._simulate_lap_times
def _lap_times(self, is_yf):
    lt = _lt(self, is_yf)
    if OFF and not (is_yf and self.fast_predictor_yf is not None):
        p = self.fast_predictor_clear
        idx = getattr(self, '_tl_idx', None)
        if idx is None:
            idx = self._tl_idx = {c: p.feature_names_.index(col) for c, col in self._tyre_life_col_map.items()}
        for i, d in enumerate(self._ordered_drivers):
            if not d.is_dnf:
                lt[i] += OFF * float(p.coef_[idx[d.compound]])
                d.predicted_laptime = lt[i]
    return lt
Env._simulate_lap_times = _lap_times
def _wrap(name):
    f = getattr(Env, name)
    def g(self, *a, **k):
        for d in self.rival_drivers: d.tyrelife += OFF
        try: return f(self, *a, **k)
        finally:
            for d in self.rival_drivers: d.tyrelife -= OFF
    setattr(Env, name, g)
for n in ('_get_rival_pit_probs_fast', '_get_rival_pit_probs_batch', '_should_rival_pit'):
    if hasattr(Env, n): _wrap(n)
print(f'[tl_plus1] offset={OFF} env={Env.__name__} wrapped:',
      [n for n in ('_get_rival_pit_probs_fast', '_get_rival_pit_probs_batch', '_should_rival_pit') if hasattr(Env, n)], flush=True)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
