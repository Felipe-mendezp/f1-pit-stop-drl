"""2026-10-14 — Ec. 2 porcentual con respaldos (eq2_fallbacks_20261010.py) + regla nueva: se excluyen las carreras
en que el piloto de referencia (VER) completa menos de 15 vueltas limpias, porque ahi los coeficientes de la Ec. 1
de esa carrera (relativos a VER) no estan definidos. En 2021-2024 solo cae Australia 2024 (VER: 2 vueltas).
La exclusion vale como observacion de entrenamiento y como carrera previa (P1-P3) de cada piloto, tanto al predecir
como en las filas de entrenamiento: P1-P3 se recalculan en todas las filas como las 3 carreras validas previas del
piloto (las covariables de eq2_features.csv se calcularon antes de la regla e incluian Australia 2024 en 50 filas
de Japon, China, Miami y Emilia Romagna 2024).

    from eq2_fallbacks_20261014 import Eq2   # misma interfaz: Eq2().predict(year, gp, driver)
"""
import os, importlib.util
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location('eq2_20261010', os.path.join(HERE, 'eq2_fallbacks_20261010.py'))
_M = importlib.util.module_from_spec(_s); _s.loader.exec_module(_M)
MIN_LAPS, spec = _M.MIN_LAPS, _M.spec


def bad_races():
    rc = pd.read_csv(os.path.join(HERE, 'race_coefs.csv'))
    v = rc[rc.Driver == 'VER'].set_index(['Year', 'GP']).n_laps
    return set(v[v < MIN_LAPS].index)


class Eq2(_M.Eq2):
    def __init__(self):
        super().__init__()
        self.excluded = bad_races()
        keep = [(y, g) not in self.excluded for y, g in zip(self.sample.Year, self.sample.GP)]
        S = self.sample[keep].sort_values(['Driver', 't']).copy()
        for k in (1, 2, 3):
            S[f'P{k}_pct'] = S.groupby('Driver').z_pct.shift(k)
        self.sample = S
