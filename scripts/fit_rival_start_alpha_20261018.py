"""2026-10-18 — Alpha simetrico de la Dirichlet-multinomial de los compuestos de partida de los rivales (v8).

Datos: vuelta 1 de data/session_{2021,2022,2023}_V2.csv en los 8 GP del paper (agrupados), conteos por etiqueta
relativa (Compound = SOFT/MEDIUM/HARD); se excluyen las carreras con otra etiqueta (mojado o sin dato: Hungarian 2021,
Belgian 2021, Singapore 2022). 2024 queda fuera (Fig. 7). Ajuste por maxima verosimilitud de
    counts_r ~ DirMult(n_r, (alpha, alpha, alpha)).
Motivo: el sorteo uniforme (alpha -> inf) deja ~6 rivales por compuesto; en las carreras reales casi todos parten con
el mismo (diagnostico: results/world2024_v7_20261015/bahrain_trpo_dsq/README.md).

    python scripts/fit_rival_start_alpha_20261018.py  ->  data/rival_start_alpha_2021_2023_20261018.json
"""
import json, os
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPS = ['Bahrain', 'Saudi Arabian', 'Miami', 'Hungarian', 'Belgian', 'Dutch', 'Singapore', 'United States']
LAB = ['SOFT', 'MEDIUM', 'HARD']
YEARS = [2021, 2022, 2023]


def counts(years):
    rows = []
    for y in years:
        d = pd.read_csv(os.path.join(ROOT, 'data', f'session_{y}_V2.csv'), usecols=['GP', 'Driver', 'LapNumber', 'Compound'])
        for g in GPS:
            x = d[(d.GP == g + ' Grand Prix') & (d.LapNumber == 1)].drop_duplicates('Driver')
            if len(x) == 0:
                continue
            c = x.Compound.value_counts()
            if set(c.index) - set(LAB):
                print(f'  excluida: {g} {y} {dict(c)}')
                continue
            rows.append(dict(year=y, gp=g, **{k: int(c.get(k, 0)) for k in LAB}))
    return pd.DataFrame(rows)


def loglik(a, X):
    n, A = X.sum(1), a.sum()
    return float((gammaln(A) - gammaln(n + A) + (gammaln(X + a) - gammaln(a)).sum(1)).sum())


def main():
    R = counts(YEARS)
    X = R[LAB].values
    s = minimize_scalar(lambda t: -loglik(np.full(3, np.exp(t)), X), bounds=(-6, 4), method='bounded')
    alpha = float(np.exp(s.x))
    r = minimize(lambda t: -loglik(np.exp(t), X), np.zeros(3))
    print(R.to_string(index=False))
    print(f'alpha simetrico = {alpha:.4f}  (ll {-s.fun:.2f});  asimetrico S,M,H = {np.exp(r.x).round(3)}  (ll {-r.fun:.2f})')
    out = dict(alpha=round(alpha, 4), years=YEARS, gps=GPS, n_races=len(R), loglik=-s.fun,
               alpha_asym_SMH=np.exp(r.x).round(4).tolist(), loglik_asym=-r.fun,
               mean_majority_share_real=float((X.max(1) / X.sum(1)).mean()), races=R.to_dict('records'))
    path = os.path.join(ROOT, 'data', 'rival_start_alpha_2021_2023_20261018.json')
    json.dump(out, open(path, 'w'), indent=1)
    print('->', path)


if __name__ == '__main__':
    main()
