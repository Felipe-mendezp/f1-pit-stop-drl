"""2026-10-27 — Copia de ../rival_overtake_tables_v7_20261020/make_tables_20261020.py para el simulador v10 (trained_models/simulation_v10_20261027). Original: 2026-10-20 — Tablas del suplementario (tab:logit_pitin, tab:logit_compound_choice, tab:overtake_logit) desde los
modelos del simulador v7 (trained_models/simulation_v7_20261015). Ítems 30 y 32 del plan de cambios.
  LogitStop: pickle rival_pitstop_logit.pkl (statsmodels; e.e. estándar).
  CondLogitCompound (K1): coeficientes = pickle rival_compound_logit.pkl; e.e. agrupados por carrera desde
    k1_cluster.csv (k1_cluster_20261020.py reestima K1, coincide con el pickle, y calcula el sandwich por carrera).
  Overtake: pickles overtake_logit.pkl (PerGPLogitView, p-valores) + N y adelantamientos de
    results/sim_v7_20261015/overtake_v7_v7_paper/params.csv.
    source reproduce/env_paper_v3_20261007.sh; F1_TYRELIFE_2021=v7 OUT=<dir>/k1_cluster.csv python <dir>/k1_cluster_20261020.py
    python <dir>/make_tables_20261020.py
"""
import sys, os, pickle
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, 'src')]
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(ROOT, 'trained_models', 'simulation_v10_20261027')
PAPER = ['Bahrain', 'Belgian', 'Dutch', 'Hungarian', 'Miami', 'Saudi Arabian', 'Singapore', 'United States']


def load(f):
    with open(f, 'rb') as fh:
        return pickle.load(fh)


def stars(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''


def row(name, b, se, p, comment=False):
    return ('% ' if comment else '') + f'{name} & {b:.4f} & {stars(p)} & ({se:.3f}) \\\\'


def stop_table():
    s = load(os.path.join(SIM, 'Bahrain Grand Prix', 'rival_pitstop_logit.pkl'))
    L = [row('Intercept', s.params['Intercept'], s.bse['Intercept'], s.pvalues['Intercept'])]
    for k in s.params.index:
        if k.startswith('GP[T.'):
            g = k[5:-1].replace(' Grand Prix', '')
            L.append(row(f'{g} GP', s.params[k], s.bse[k], s.pvalues[k], comment=g not in PAPER))
    names = [('SC', 'SC'), ('VSC', 'VSC'), ('used_two_compounds', 'Used two compounds'),
             ('EarlyLaps', 'Early laps'), ('LateLaps', 'Late laps')]
    names += [(f'TyreLife:C(Compound_Detail)[C{c}]', f'TyreLife $\\cdot$ C{c}') for c in range(6)]
    names += [('Position', 'Position')]
    for k, n in names:
        L.append(row(n, s.params[k], s.bse[k], s.pvalues[k], comment=n.endswith('C0')))
    L += ['\\midrule', f'Observations & \\multicolumn{{3}}{{c}}{{{int(s.nobs):,}}} \\\\',
          f'Pseudo $R^2$ & \\multicolumn{{3}}{{c}}{{{s.prsquared:.3f}}} \\\\']
    return '\n'.join(L)


def compound_table():
    v = load(os.path.join(SIM, 'Bahrain Grand Prix', 'rival_compound_logit.pkl'))
    K = pd.read_csv(os.path.join(HERE, 'k1_cluster.csv'), index_col=0)
    def key(c):  # 'C(alt, Treatment("C3"))[C0]:LapsLeft' -> 'C(alt)[C0]:LapsLeft'
        return c.replace('C(alt, Treatment("C3"))', 'C(alt)')
    K.index = [key(c) for c in K.index]
    assert np.allclose(K.b.values, v.params[K.index].values), 'K1 reestimado != pickle'
    L = []
    for c in (0, 1, 2, 4, 5):
        k = f'C(alt)[C{c}]'; L.append(row(f'Alt: C{c}', K.b[k], K.se_cl[k], K.p_cl[k], comment=c == 0))
    L.append('')
    for c in (0, 1, 2, 4, 5):
        k = f'C(alt)[C{c}]:LapsLeft'
        L.append(row(f'Alt: C{c} $\\cdot$ LapsLeft', K.b[k], K.se_cl[k], K.p_cl[k], comment=c == 0))
    L.append('')
    k = 'not_change_compound'
    L.append(row('Alt: Same \\& only one comp. used', K.b[k], K.se_cl[k], K.p_cl[k]))
    return '\n'.join(L)


def overtake_table():
    P = pd.read_csv(os.path.join(ROOT, 'results', 'sim_v10_20261027', 'overtake_v10', 'params.csv')).set_index('GP')
    L = []
    for gp in sorted(P.index):
        m = load(os.path.join(SIM, gp, 'overtake_logit.pkl'))
        assert abs(m.params['const'] - P.loc[gp, 'const']) < 1e-9
        n, o = int(P.loc[gp, 'n']), int(P.loc[gp, 'overtakes'])
        g = gp.replace(' Grand Prix', '')
        L.append(f'{g:<17} & {m.params["const"]:.4f} & {stars(m.pvalues["const"])} & {m.params["delta_total"]:.4f} & '
                 f'{stars(m.pvalues["delta_total"])} & {o} & {n} & {o / n:.3f} \\\\')
    return '\n'.join(L)


if __name__ == '__main__':
    for name, f in [('tab_logit_pitin_v10', stop_table), ('tab_logit_compound_choice_v10', compound_table),
                    ('tab_overtake_logit_v10', overtake_table)]:
        t = f(); open(os.path.join(HERE, name + '.tex'), 'w').write(t + '\n'); print(f'--- {name}\n{t}')
