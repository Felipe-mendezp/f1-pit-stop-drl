"""2026-10-28: copia en results/sim_v10_20261028/ que ademas importa data_fix_20261028 (compuestos de Belgica 2024 y
Canada 2023 corregidos con OpenF1). Original: results/sim_v10_20261027/.
2026-10-27 — Piezas comunes del simulador v10 (docs/PROPUESTA_V10_20261027.md, raiz del proyecto).

PitIn/PitOut de data/session_*_V2.csv son pasos por el pit lane (FastF1 PitInTime/PitOutTime), no paradas: incluyen el
paso del peloton por el pit lane detras del SC y, en verde, penalizaciones (results/pitin_flag_check_20261027/).
Paso sin parada:  in_pass  = PitIn = 1 y la vuelta siguiente del piloto tiene el mismo compuesto y TyreLife + 1;
                  out_pass = PitOut = 1 y la vuelta anterior tiene el mismo compuesto y TyreLife - 1.
Se calcula sobre los CSV sin reescalar TyreLife (2021). Parada real = PitIn = 1 y no in_pass.
  pass_flags()            -> DataFrame Year, GP, Driver, LapNumber, in_pass, out_pass (2021-2024)
  patch_regressions(R)    -> R.load_data devuelve las vueltas SIN pasos (para la Ec. 1 y los neumaticos)
  rival_laps(RM)          -> RM.load_lap_data() con PitIn = parada real (para LogitStop y K1)
"""
import os
import numpy as np
import pandas as pd
import data_fix_20261028  # noqa: F401  2026-10-28 (copia en sim_v10_20261028): compuestos corregidos al leer los CSV

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data')
KEY = ['Year', 'GP', 'Driver', 'LapNumber']


def pass_flags(years=(2021, 2022, 2023, 2024)):
    out = []
    for y in years:
        s = pd.read_csv(os.path.join(DATA, f'session_{y}_V2.csv'))
        s['Year'] = y
        s = s.sort_values(['GP', 'Driver', 'LapNumber'])
        g = s.groupby(['GP', 'Driver'])
        n, p = g.shift(-1), g.shift(1)
        s['in_pass'] = ((s.PitIn == 1) & (n.LapNumber == s.LapNumber + 1)
                        & (n.Compound_Detail == s.Compound_Detail) & (n.TyreLife == s.TyreLife + 1))
        s['out_pass'] = ((s.PitOut == 1) & (p.LapNumber == s.LapNumber - 1)
                         & (p.Compound_Detail == s.Compound_Detail) & (s.TyreLife == p.TyreLife + 1))
        out.append(s[KEY + ['in_pass', 'out_pass']])
    return pd.concat(out, ignore_index=True)


def _merge(df, F):
    m = df.merge(F, on=KEY, how='left', validate='many_to_one')
    m[['in_pass', 'out_pass']] = m[['in_pass', 'out_pass']].fillna(False).astype(bool)
    return m


def patch_regressions(R):
    """R = models.regressions. Reemplaza R.load_data para que quite las vueltas de paso sin parada (in o out)."""
    if getattr(R, '_v10_patched', False):
        return
    orig, F = R.load_data, pass_flags()

    def load_data():
        res = []
        for y, d in zip((2021, 2022, 2023, 2024), orig()):
            d = d.copy(); had = 'Year' in d.columns
            d['Year'] = y
            m = _merge(d, F)
            keep = ~(m.in_pass | m.out_pass).values
            d = d[keep]
            if not had:
                d = d.drop(columns='Year')
            res.append(d)
        return tuple(res)

    R.load_data, R._v10_patched = load_data, True


def rival_laps(RM):
    """RM = models.rival_models. Vueltas 2021-23 como RM.load_lap_data(), con PitIn = parada real."""
    L = RM.load_lap_data()
    m = _merge(L, pass_flags((2021, 2022, 2023)))
    m['PitIn_raw'] = m.PitIn
    m['PitIn'] = ((m.PitIn == 1) & ~m.in_pass).astype(int)
    return m.drop(columns=['in_pass', 'out_pass'])
