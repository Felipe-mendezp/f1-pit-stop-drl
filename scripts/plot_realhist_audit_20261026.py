"""2026-10-26 — Analisis B (historia real 2024): recomendacion DRL frente a la decision real, vuelta a vuelta (propuesta 1).

Lee compare_laps_20261026.csv (results/decision_support_v9r2smrealhist_20261026/compare_real_20261026.py) y los meta de
los rollouts. Escribe en --out:
  realhist_audit.{pdf,png}  8 paneles (GP x piloto), PPO escenario (ii) (como las Figs. 8-10): diferencia pareada media de
                            puntos "seguir la recomendacion DRL - decision real" en la vuelta n, banda +-2 SE; 0 cuando
                            coinciden. Marcadores con el color de la accion recomendada en las vueltas en que difieren;
                            paradas reales en lineas discontinuas; SC/VSC reales sombreados.
  realhist_table.tex        tabla del apendice (26-oct: filas = las de la figura de tandas): paradas reales + todas las
                            vueltas en que la recomendacion PPO (ii) difiere de la decision real; PPO y RPPO, escenarios (i) y (ii); * = mas de
                            2 SE; daga = la mascara corrige mas del 5 % de los rollouts de la accion real o recomendada.
Uso: F1_PAPER_FONTS=1 python scripts/plot_realhist_audit_20261026.py --res results/decision_support_v9r2smrealhist_20261026
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_decision_evolution import ACTION_COLORS, ACTION_MARKERS, draw_yf_spans
import paper_fonts_20261021

GPS = [('bahrain_alo', 'Bahrain', 'Fernando Alonso'), ('saudi_arabian_nor', 'Saudi Arabia', 'Lando Norris'),
       ('miami_oco', 'Miami', 'Esteban Ocon'), ('hungarian_tsu', 'Hungary', 'Yuki Tsunoda'),
       ('belgian_gas', 'Belgium', 'Pierre Gasly'), ('dutch_str', 'Netherlands', 'Lance Stroll'),
       ('singapore_hul', 'Singapore', 'Nico H\u00fclkenberg'), ('united_states_mag', 'United States', 'Kevin Magnussen')]
NAME = {0: 'No pit', 1: 'Hard', 2: 'Medium', 3: 'Soft'}
PIT_COLOR = '0.25'
ABBR = {'No': 'N', 'Hard': 'H', 'Medium': 'M', 'Soft': 'S'}  # 27-oct: abreviaturas en la tabla
MASK_FLAG = 5.0  # % de rollouts (accion real o recomendada) corregidos por la mascara de legalidad -> daga en la tabla


def fmt_d(d):
    """Diferencia con signo; '0.0' sin signo si redondea a cero (misma regla en la tabla y en la figura de tandas)."""
    return '0.0' if abs(d) < 0.05 else f'{d:+.1f}'


def meta(res, key, algo='PPO', scen='real', lap=1):
    return json.load(open(os.path.join(res, algo, f'{key}_{scen}_lap{lap}_meta.json')))


def act(label):
    """'Hard (C1)' -> 1, 'No pit' -> 0 (orden de la accion = indice del compuesto disponible + 1)."""
    return {'No': 0, 'Hard': 1, 'Medium': 2, 'Soft': 3}[label.split()[0]]


def figure(L, res, out):
    fig, axes = plt.subplots(4, 2, figsize=(11, 10.5), sharey=True)
    for ax, (key, gp, drv) in zip(axes.flat, GPS):
        m = meta(res, key)
        draw_yf_spans(ax, m['laps_sc_2024'], m['laps_vsc_2024'], label=False)
        g = L[(L.gp == key) & (L.algo == 'PPO') & (L.scen == '(ii)')].sort_values('lap')
        d = -g.dpts_real_vs_rec  # rec - real
        ax.fill_between(g.lap, d - g.dpts_2se, d + g.dpts_2se, color='0.4', alpha=0.25, lw=0, zorder=1)
        ax.plot(g.lap, d, color='black', lw=1.0, zorder=2)
        ax.axhline(0, color='black', lw=0.6, zorder=1)
        for r in g[g.real != g.rec].itertuples():
            a = act(r.rec)
            ax.scatter(r.lap, -r.dpts_real_vs_rec, s=26, marker=ACTION_MARKERS[a], color=ACTION_COLORS[a],
                       edgecolor='black', linewidth=0.5, zorder=4)
        for r in g[g.real != 'No pit'].itertuples():
            ax.axvline(r.lap, color=PIT_COLOR, ls='--', lw=1.0, zorder=3)
            ax.text(r.lap + 0.6, 0.97, f'Pit: {r.real.split()[0]}', transform=ax.get_xaxis_transform(),
                    ha='left', va='top', rotation=90, fontsize=8, color=PIT_COLOR)
        ax.set_title(f'{gp} ({drv})', fontsize=10)
        ax.set_xlim(0.5, m['n_laps'] - 0.5)
    for ax in axes[-1]:
        ax.set_xlabel('Current lap', fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel('Points: DRL $-$ actual', fontsize=10)
    h = [Line2D([], [], color='black', lw=1.0, label='Mean paired difference'),
         Patch(facecolor='0.4', alpha=0.25, label='$\\pm 2$ SE'),
         Line2D([], [], color=PIT_COLOR, ls='--', label='Actual pit stop')]
    h += [Line2D([], [], ls='none', marker=ACTION_MARKERS[a], color=ACTION_COLORS[a], markeredgecolor='black',
                 markeredgewidth=0.5, label=f'DRL recommends {NAME[a].lower() if a else "no pit"}') for a in (1, 2, 3, 0)]
    h += [Patch(facecolor='#8e7cc3', alpha=0.5, label='Safety car'), Patch(facecolor='#c9bde8', alpha=0.5, label='Virtual SC')]
    fig.legend(handles=h, loc='lower center', ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.045))
    fig.tight_layout()
    paper_fonts_20261021.apply(fig)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'realhist_audit.{ext}'), dpi=300, bbox_inches='tight')
    print('->', os.path.join(out, 'realhist_audit.pdf'))


def table(L, res, out, rppo=False):
    p2 = L[(L.algo == 'PPO') & (L.scen == '(ii)')]
    keys = sorted(set(map(tuple, p2[(p2.real != 'No pit') | (p2.real != p2.rec)][['gp', 'lap']].values)),
                  key=lambda k: ([g[0] for g in GPS].index(k[0]), k[1]))
    # (ii) = banderas amarillas reales hasta n y sorteadas despues (como las Figs. 8-10); (i) = reales toda la carrera
    cols = [('PPO', '(ii)'), ('PPO', '(i)')] + ([('RecurrentPPO', '(ii)'), ('RecurrentPPO', '(i)')] if rppo else [])
    lines, mask_max = [], 0.0
    for key, n in keys:
        gp = dict((g[0], g[1]) for g in GPS)[key]
        m = meta(res, key, lap=n)
        st = m['agent_state_at_lap']
        comp = ABBR[NAME[m['available_compounds'].index(st['compound']) + 1]]
        r2 = p2[(p2.gp == key) & (p2.lap == n)].iloc[0]
        cells = []
        for algo, scen in cols:
            r = L[(L.gp == key) & (L.algo == algo) & (L.scen == scen) & (L.lap == n)].iloc[0]
            mask_max = max(mask_max, r.mask_real, r.mask_rec)
            if r.rec == r.real:
                cells.append('=')
            else:
                star = ('*' if r.signif else '') + ('\\dagger' if max(r.mask_real, r.mask_rec) > MASK_FLAG else '')
                star = f'$^{{{star}}}$' if star else ''
                cells.append(f'{ABBR[r.rec.split()[0]]}, ${fmt_d(-r.dpts_real_vs_rec)} \\pm {r.dpts_2se:.1f}${star}')
        lines.append(f'{gp} & {n} & {st["position"]} & {comp} ({st["tyrelife"]}) & {ABBR[r2.real.split()[0]]} & '
                     + ' & '.join(cells) + ' \\\\')
    if rppo:
        tex = ('\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lcccccccc}\n\\toprule\n'
               '& & & & & \\multicolumn{2}{c}{PPO} & \\multicolumn{2}{c}{Rec. PPO} \\\\\n'
               '\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n'
               'GP & Lap & Position & Tire (age) & Actual & Sampled YF & Actual YF & Sampled YF & Actual YF \\\\\n\\midrule\n'
               + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}}\n')
    else:  # 27-oct: solo PPO, sin resizebox
        tex = ('\\begin{tabular}{lccccll}\n\\toprule\n'
               '& & & & & \\multicolumn{2}{c}{DRL recommendation} \\\\\n'
               '\\cmidrule(lr){6-7}\n'
               'GP & Lap & Position & Tire (age) & Actual & Sampled YF & Actual YF \\\\\n\\midrule\n'
               + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')
    open(os.path.join(out, 'realhist_table.tex'), 'w').write(tex)
    print(tex)
    print(f'filas {len(keys)}; mascara max en las acciones de la tabla (real o rec): {mask_max:.1f} %')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--rppo', action='store_true', help='incluir Recurrent PPO (tabla de 4 columnas, version del 26-oct)')
    a = ap.parse_args()
    out = a.out or os.path.join(a.res, 'figs')
    os.makedirs(out, exist_ok=True)
    L = pd.read_csv(os.path.join(a.res, 'compare_laps_20261026.csv'))
    figure(L, a.res, out)
    table(L, a.res, out, rppo=a.rppo)


if __name__ == '__main__':
    main()
