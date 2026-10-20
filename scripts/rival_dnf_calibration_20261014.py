"""Calibracion de abandonos de rivales para el simulador multi-piloto (opt-in F1_RIVAL_DNF) — 2026-10-14.

Motivo: en la Fig. 7 los rivales siguen su estrategia real 2024, que incluye abandonos; en el entrenamiento nunca
abandonaban, y un rival retirado (time_diff_to_agent = +500) gatillaba paradas sin sentido de RPPO/TRPO
(results/world2024_v5h_ood_20261014). Con F1_RIVAL_DNF los rivales estocasticos abandonan con esta calibracion.

Datos: session_2021..2023_V2.csv, todas las carreras salvo las ediciones con >10 % de vueltas WET/INTERMEDIATE
(filtro A) o acortadas (< 90 % de las vueltas habituales del GP). Abandono = piloto que completa menos de n - 3
vueltas (los doblados terminan a 1-2 vueltas). Se guarda la probabilidad por piloto y carrera y la distribucion
empirica de la fraccion de carrera completada al abandonar (agrupadas, no por GP: ~2.5 abandonos por carrera).
    python scripts/rival_dnf_calibration_20261014.py        # -> data/rival_dnf_2021_2023_20261014.json
"""
import os, json
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'rival_dnf_2021_2023_20261014.json')

races, fracs = [], []
for y in (2021, 2022, 2023):
    d = pd.read_csv(os.path.join(ROOT, 'data', f'session_{y}_V2.csv'), usecols=['GP', 'Driver', 'LapNumber', 'Compound'])
    for g, x in d.groupby('GP'):
        races.append(dict(year=y, gp=g, n=int(x.LapNumber.max()),
                          wet=float(x.Compound.isin(['WET', 'INTERMEDIATE']).mean()),
                          last=x.groupby('Driver').LapNumber.max().to_dict()))
r = pd.DataFrame(races)
r['n_typ'] = r.groupby('gp').n.transform('max')
keep = r[(r.wet <= 0.10) & (r.n >= 0.9 * r.n_typ)]
starts = dnf = 0
for e in keep.itertuples():
    starts += len(e.last)
    for drv, k in e.last.items():
        if k < e.n - 3:
            dnf += 1; fracs.append(round(float(k) / e.n, 4))
out = dict(p_dnf=dnf / starts, n_races=len(keep), n_starts=starts, n_dnf=dnf,
           dnf_per_race=dnf / len(keep), frac_completed=sorted(fracs),
           excluded=[f'{e.year} {e.gp}' for e in r.drop(keep.index).itertuples()],
           note='scripts/rival_dnf_calibration_20261014.py; abandono = < n-3 vueltas; frac = vueltas completadas / n')
json.dump(out, open(OUT, 'w'), indent=1)
print(f"p_dnf={out['p_dnf']:.3f}  abandonos/carrera={out['dnf_per_race']:.2f}  carreras={len(keep)}  excluidas={out['excluded']}")
print('fraccion completada: cuartiles', pd.Series(fracs).quantile([.1, .25, .5, .75, .9]).round(2).tolist(), '->', OUT)
