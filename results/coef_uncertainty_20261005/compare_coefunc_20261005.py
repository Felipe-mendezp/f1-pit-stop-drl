"""2026-10-05 — Resumen de la evaluacion con coeficientes de neumatico sorteados por carrera
(scripts/evaluate_coef_uncertainty_20261005.py) contra la misma evaluacion sin sorteo.

  sin sorteo : publicados -> results/fig06_stochastic, results/fig07_real_2024 (mismas semillas)
               control    -> caches en trained_models/rl_agents_tyre_control_20261004/
  con sorteo : results/coef_uncertainty_20261005/fig<6|7>_<tag>_f<factor>/qmip_vs_drl_coefunc_results.csv

Correr desde la raiz del repo:  python results/coef_uncertainty_20261005/compare_coefunc_20261005.py
Salida: resumen_coefunc.csv y brecha_qmip_coefunc.csv en esta carpeta.
"""
import glob, os
import pandas as pd

ROOT = os.getcwd()
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = {
    (6, 'publicado'): 'results/fig06_stochastic/qmip_vs_drl_stochastic_results.csv',
    (7, 'publicado'): 'results/fig07_real_2024/all_gps_2024_qmip_results.csv',
    (6, 'control'): 'trained_models/rl_agents_tyre_control_20261004/qmip_vs_drl_stochastic_results.csv',
    (7, 'control'): 'trained_models/rl_agents_tyre_control_20261004/all_gps_2024_qmip_results.csv',
}
ALGOS = ['PPO', 'RecurrentPPO', 'QMIP']
DEGENERATE = {('Miami', 'PPO'), ('Belgian', 'RecurrentPPO')}  # rechazados por el filtro de paradas en control


def load(path, fig, tag, factor, mundo='publicado'):
    d = pd.read_csv(path)
    d = d[d.algo.isin(ALGOS) & ~d.display_name.str.startswith('Emilia')].copy()
    d['dsq'] = ~d.changed_compound.astype(bool)
    return d.assign(fig=fig, agentes=tag, factor=factor, mundo=mundo)


frames = [load(os.path.join(ROOT, p), f, t, 0) for (f, t), p in BASE.items() if os.path.isfile(os.path.join(ROOT, p))]
for p in sorted(glob.glob(os.path.join(HERE, 'fig*_f*', 'qmip_vs_drl_coefunc_results.csv'))):
    name = os.path.basename(os.path.dirname(p))           # fig6_publicado_f3
    fig, tag, fac, *extra = name.split('_')            # ..._mundo-opcion2_sinruidoest (06-oct)
    mundo = next((e.split('-')[1] for e in extra if e.startswith('mundo-')), 'publicado')
    frames.append(load(p, int(fig[3:]), tag, float(fac[1:]), mundo))
d = pd.concat(frames, ignore_index=True)
d['gp'] = d.display_name.str.split(' ').str[0]
d['degenerado'] = [(g, a) in DEGENERATE for g, a in zip(d.gp, d.algo)]

keys = ['fig', 'agentes', 'mundo', 'factor']
tot = d.groupby(keys + ['algo']).agg(pos=('final_position', 'mean'), dsq_pct=('dsq', lambda x: 100 * x.mean()),
                                    n=('final_position', 'size')).round(2)
tot.to_csv(os.path.join(HERE, 'resumen_coefunc.csv'))

piv = d.groupby(keys + ['gp', 'algo']).final_position.mean().unstack('algo')
gap = pd.DataFrame({a: piv['QMIP'] - piv[a] for a in ['PPO', 'RecurrentPPO']}).round(2)
gap.to_csv(os.path.join(HERE, 'brecha_qmip_coefunc.csv'))
mean_all = gap.groupby(keys).mean().round(2)
mask = gap.reset_index()
for a, g in [('PPO', 'Miami'), ('RecurrentPPO', 'Belgian')]:
    mask.loc[mask.gp == g, a] = float('nan')
mean_filt = mask.groupby(keys)[['PPO', 'RecurrentPPO']].mean().round(2)
pd.set_option('display.width', 200)
print('Posicion media (8 GP) y % DSQ:\n', tot.to_string())
print('\nBrecha QMIP - DRL (positivo = DRL mejor), promedio 8 GP:\n', mean_all.to_string())
print('\nIdem sin Miami PPO ni Belgica RPPO:\n', mean_filt.to_string())
print('\nBrecha por GP:\n', gap.unstack(['fig', 'agentes', 'mundo', 'factor']).to_string())
