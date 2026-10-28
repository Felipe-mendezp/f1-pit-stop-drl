"""2026-10-28 — Prueba preliminar: Ec. 1 por GP (src/models/regressions.filtrar_outliers_iqr, vueltas verdes, misma formula)
con la edad de FastF1 (raw) y con las edades efectivas de ages_20261028.py (G, Q, GQ).

Datos como en v10d: compuestos corregidos, sin pasos por el pit lane, factor 2021 = 0.74 (results/sim_v10d_20261028), y
muestras como en el estudio de neumaticos (filtro A: hist_laps_A 2021-23 y laps_2024_A). Para cada GP la muestra (filtro
IQR) se fija con raw y las variantes se ajustan sobre las MISMAS vueltas, cambiando solo TyreLife (el factor 2021 se aplica
igual a todas). Se compara la suma de cuadrados de residuos (mismo numero de parametros) y las pendientes.
    F1_TYRELIFE_2021=v10 ~/.venvs/f1-drl-v2/bin/python results/tyrelife_effective_20261028/compare_20261028.py
Salida: fit_by_gp.csv, slopes.csv, compare_summary.txt
"""
import io, os, sys, contextlib, importlib.util, warnings
warnings.filterwarnings('ignore')
ROOT = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'results', 'sim_v10d_20261028')]
assert os.environ.get('F1_TYRELIFE_2021') == 'v10'
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
import models.regressions as R
import common_v10_20261027 as C          # importa data_fix_v10d_20261028
C.patch_regressions(R)
spec = importlib.util.spec_from_file_location('pooled', os.path.join(ROOT, 'results', 'tyre_coef_study_20260930',
                                                                      'pooled_20261004.py'))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
assert P.R is R
VARS = ['TL_G', 'TL_Q', 'TL_GQ']
KEY = ['Year', 'GP', 'Driver', 'LapNumber']
out = []


def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)


def slopes(m):
    return {k.split('[')[-1].rstrip(']'): v for k, v in m.params.items() if k.startswith('TyreLife:')}


def run(d, label, A):
    d = d.copy(); d['Year'] = d.Year.astype(int)
    d['GP0'] = d.GP.replace('Austrian Grand Prix', 'Styrian Grand Prix')   # hist_laps_A junta Styrian con Austrian
    m_ = d.merge(A, on=KEY, how='left')
    m2 = d.merge(A, left_on=['Year', 'GP0', 'Driver', 'LapNumber'], right_on=KEY, how='left', suffixes=('', '_s'))
    for c in ['TL_raw'] + VARS:
        m_[c] = m_[c].fillna(m2[c])
    miss = m_.TL_raw.isna().mean()
    f = (m_.TyreLife / m_.TL_raw).where(m_.TL_raw > 0, 1.0)          # factor 2021 (0.74) u 1
    for v in VARS:
        d[v] = (m_[v] * f).values
    d['TL_base'] = d.TyreLife
    say(f'== {label}: {len(d)} vueltas, sin edad efectiva {miss:.2%}; |TyreLife - TL_raw x factor| max '
        f'{(m_.TyreLife - m_.TL_raw * f).abs().max():.2g}')
    fit, sl = [], []
    for gp in sorted(d.GP.unique()):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m0, dd = R.filtrar_outliers_iqr(d, gp)
        except Exception as e:
            continue
        r = dict(set=label, GP=gp, n=int(m0.nobs), sse_raw=m0.ssr, r2_raw=m0.rsquared)
        sl += [dict(set=label, GP=gp, var='raw', comp=c, slope=b) for c, b in slopes(m0).items()]
        for v in VARS:
            x = dd.copy(); x['TyreLife'] = x[v]
            if x.TyreLife.isna().any():
                r[f'sse_{v}'] = np.nan; continue
            with contextlib.redirect_stdout(io.StringIO()):
                m1, _ = R.calibrar_regresion_gp(x)
            r[f'sse_{v}'] = m1.ssr; r[f'r2_{v}'] = m1.rsquared
            sl += [dict(set=label, GP=gp, var=v, comp=c, slope=b) for c, b in slopes(m1).items()]
        fit.append(r)
    return pd.DataFrame(fit), pd.DataFrame(sl)


def main():
    A = pd.read_csv(os.path.join(HERE, 'ages_2021_2024.csv.gz'))[KEY + ['TL_raw'] + VARS]
    dh, d24 = P.hist_laps_A(), P.laps_2024_A()
    F, S = zip(run(dh, '2021-23', A), run(d24, '2024', A))
    F, S = pd.concat(F), pd.concat(S)
    F.to_csv(os.path.join(HERE, 'fit_by_gp.csv'), index=False); S.to_csv(os.path.join(HERE, 'slopes.csv'), index=False)
    pd.set_option('display.width', 200)
    for lab, x in F.groupby('set'):
        for sub, y in (('todos', x), ('8 GP paper', x[x.GP.isin(P.PAPER)])):
            say(f'\n{lab} ({sub}, {len(y)} GP, {y.n.sum()} vueltas): RMSE raw {np.sqrt(y.sse_raw.sum() / y.n.sum()):.4f} s')
            for v in VARS:
                dv = (y[f'sse_{v}'] - y.sse_raw) / y.sse_raw
                say(f'  {v:6s} SSE total {100 * (y[f"sse_{v}"].sum() / y.sse_raw.sum() - 1):+.2f}% | GP que mejoran '
                    f'{int((dv < 0).sum())}/{dv.notna().sum()} | cambio mediano por GP {100 * dv.median():+.2f}%')
        if lab == '2021-23' or True:
            y = x[x.GP.isin(P.PAPER)].set_index('GP')
            cols = ['n'] + [f'{a}' for a in ['r2_raw']] + [f'r2_{v}' for v in VARS]
            say(y[cols].round(4).to_string())
    T = S.pivot_table(index=['set', 'GP', 'comp'], columns='var', values='slope').reset_index()
    for lab, x in T.groupby('set'):
        say(f'\npendientes {lab}: mediana por GP-compuesto (s/vuelta)',
            {v: round(float(x[v].median()), 4) for v in ['raw'] + VARS},
            '| pendientes negativas', {v: int((x[v] < 0).sum()) for v in ['raw'] + VARS}, f'de {len(x)}')
    open(os.path.join(HERE, 'compare_summary.txt'), 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
