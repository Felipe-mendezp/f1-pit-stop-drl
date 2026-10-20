"""2026-10-04 — Comparacion de las 3 calibraciones de neumatico (actual, opcion2, opcion3) con
PPO, RecurrentPPO y QMIP, al estilo de las Figs. 6 (carreras estocasticas) y 7 (replay 2024).

  actual  : resultados publicados (results/fig06_stochastic, results/fig07_real_2024), 8 GP
  opcionX : caches de evaluate_*.py en trained_models/rl_agents_tyre_<opcionX>_20261004/

Correr desde la raiz del repo:  python results/tyre_coef_study_20260930/compare_options_20261004.py
Salida: results/tyre_coef_study_20260930/compare_20261004/
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.getcwd()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compare_20261004')
os.makedirs(OUT, exist_ok=True)
F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
ALGOS = ['PPO', 'RecurrentPPO', 'QMIP']
COLORS = {'PPO': '#009E73', 'RecurrentPPO': '#CC79A7', 'QMIP': '#000000'}
OPTS = ['actual', 'control', 'opcion2', 'opcion3']
OPT_LABEL = {'actual': 'Actual', 'control': 'Control (reentr.)', 'opcion2': 'Op. 2 (libre)', 'opcion3': 'Op. 3 (monótona)'}
EXPS = {
    'fig6_estocastico': ('results/fig06_stochastic/qmip_vs_drl_stochastic_results.csv',
                         'qmip_vs_drl_stochastic_results.csv'),
    'fig7_real2024': ('results/fig07_real_2024/all_gps_2024_qmip_results.csv',
                      'all_gps_2024_qmip_results.csv'),
}


def load(exp):
    pub, cache = EXPS[exp]
    frames = []
    for opt in OPTS:
        path = os.path.join(ROOT, pub) if opt == 'actual' else \
            os.path.join(ROOT, 'trained_models', f'rl_agents_tyre_{opt}_20261004', cache)
        if not os.path.isfile(path):
            print(f'[{exp}] falta {path}; se omite {opt}')
            continue
        d = pd.read_csv(path)
        d = d[d.algo.isin(ALGOS) & ~d.display_name.str.startswith('Emilia')].copy()
        d['opcion'] = opt
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d['points'] = d.final_position.map(lambda p: F1_POINTS.get(int(p), 0))
    d['dsq'] = ~d.changed_compound.astype(bool)
    return d


def summarize(d, exp):
    agg = dict(pos_media=('final_position', 'mean'), pos_sd=('final_position', 'std'),
               puntos_media=('points', 'mean'), dsq_pct=('dsq', lambda x: 100 * x.mean()),
               paradas_media=('pit_stops', 'mean'), n=('final_position', 'size'))
    tot = d.groupby(['opcion', 'algo']).agg(**agg).round(3)
    gp = d.groupby(['display_name', 'opcion', 'algo']).agg(**agg).round(3)
    tot.to_csv(os.path.join(OUT, f'{exp}_resumen_total.csv'))
    gp.to_csv(os.path.join(OUT, f'{exp}_resumen_por_gp.csv'))
    print(f'\n=== {exp}: promedio sobre los 8 GP ===')
    print(tot.to_string())
    # QMIP - DRL por opcion (positivo = DRL mejor en posicion)
    piv = d.groupby(['opcion', 'display_name', 'algo']).final_position.mean().unstack('algo')
    gap = pd.DataFrame({f'QMIP-{a}': piv['QMIP'] - piv[a] for a in ['PPO', 'RecurrentPPO'] if a in piv})
    gap.round(2).to_csv(os.path.join(OUT, f'{exp}_brecha_qmip_por_gp.csv'))


def plot(d, exp):
    gps = list(dict.fromkeys(d.display_name))
    fig, axes = plt.subplots(2, 4, figsize=(19, 7.5), sharey=True)
    for ax, gp in zip(axes.flat, gps):
        pos, data, cols = [], [], []
        for i, opt in enumerate(OPTS):
            for j, a in enumerate(ALGOS):
                x = d[(d.display_name == gp) & (d.opcion == opt) & (d.algo == a)].final_position
                if len(x):
                    pos.append(i * 4 + j); data.append(x.values); cols.append(COLORS[a])
        bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=False)
        for patch, c in zip(bp['boxes'], cols):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        for p, x in zip(pos, data):
            ax.plot(p, np.mean(x), 'D', color='white', mec='k', ms=4)
        ax.set_xticks([i * 4 + 1 for i in range(len(OPTS))]); ax.set_xticklabels([OPT_LABEL[o] for o in OPTS], fontsize=8)
        ax.set_title(gp, fontsize=10); ax.invert_yaxis() if ax is axes.flat[0] else None
        ax.grid(axis='y', alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel('Posición final')
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[a], alpha=0.55) for a in ALGOS]
    fig.legend(handles, ALGOS, loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{exp}_boxplot.{ext}'), dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    for exp in EXPS:
        d = load(exp)
        summarize(d, exp)
        plot(d, exp)
    print('->', OUT)
