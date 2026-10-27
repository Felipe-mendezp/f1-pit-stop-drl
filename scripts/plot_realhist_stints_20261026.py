"""2026-10-26 — Analisis B (historia real 2024): tandas reales del piloto y, debajo, una fila por cada vuelta en que la
recomendacion DRL difiere de la decision real (pedido del autor; reemplaza la figura de lineas realhist_audit).

Fila "Actual": estrategia real 2024 (data/session_2024_V2.csv, misma convencion que ds_rollouts_realhist_20261024.py:
la parada de la vuelta L abre la tanda nueva en L, como en la Fig. 9). Fila por vuelta n con rec != real (PPO, banderas
sorteadas desde n = archivos *_real_*): prefijo real hasta n-1 (atenuado) y, desde n, la continuacion representativa al
forzar la recomendacion y dejar decidir a la politica, obtenida con scripts/plot_decision_stints.py a traves de los
envoltorios del barrido E4 (mismo prefijo, fisica y semilla comun 1000 que la Fig. 9). A la derecha, diferencia pareada
media de puntos "recomendacion - real" +-2 SE (M = 500, compare_laps_20261026.csv). SC/VSC reales sombreados.
Requiere el entorno del barrido E4 (ver results/decision_support_v9r2smrealhist_20261026/run_20261026.sh) y ademas
F1_DS_FORCE=hist F1_DS_WORLD2024=1 F1_SIMULATION_DIR F1_RL_AGENTS_DIR:
    python scripts/plot_realhist_stints_20261026.py --res results/decision_support_v9r2smrealhist_20261026 \
        --gps saudi_arabian_nor
"""
import argparse, json, os, re, subprocess, sys
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_decision_evolution import _contiguous_runs
import matplotlib as _mpl
_mpl.rcParams['hatch.linewidth'] = 1.0
import paper_fonts_20261021
from plot_realhist_audit_20261026 import fmt_d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPS = {'bahrain_alo': ('Bahrain Grand Prix', 'Driver_ALO', 'Bahrain_ALO', 'Fernando Alonso'),
       'saudi_arabian_nor': ('Saudi Arabian Grand Prix', 'Driver_NOR', 'Saudi_Arabian_NOR', 'Lando Norris'),
       'miami_oco': ('Miami Grand Prix', 'Driver_OCO', 'Miami_OCO', 'Esteban Ocon'),
       'hungarian_tsu': ('Hungarian Grand Prix', 'Driver_TSU', 'Hungarian_TSU', 'Yuki Tsunoda'),
       'belgian_gas': ('Belgian Grand Prix', 'Driver_GAS', 'Belgian_GAS', 'Pierre Gasly'),
       'dutch_str': ('Dutch Grand Prix', 'Driver_STR', 'Dutch_STR', 'Lance Stroll'),
       'singapore_hul': ('Singapore Grand Prix', 'Driver_HUL', 'Singapore_HUL', 'Nico Hülkenberg'),
       'united_states_mag': ('United States Grand Prix', 'Driver_MAG', 'United_States_MAG', 'Kevin Magnussen')}
STINT_COLORS = ['white', 'gold', 'red']   # Fig. 5/9: hardest, middle, softest
WRAP = ['scripts/gp_drivers_v5hd_20261014.py', 'scripts/ds_rollouts_realhist_20261024.py', '--run',
        'scripts/plot_decision_stints.py']
SEED = 1000
GROUP = True   # agrupar vueltas consecutivas con la misma recomendacion
TRI = 1.0   # escala de los triangulos (la grilla del paper los agranda)
FS_BAR = 9   # tamano del numero de vueltas dentro de las barras (la grilla 2x4 lo reduce)
S24 = pd.read_csv(os.path.join(ROOT, 'data', 'session_2024_V2.csv'))


def stints_from_desc(desc):
    """'C3 (6) | C4 (44)' -> [(1, 7, 3), (7, 51, 4)] (inicio, fin exclusivo, compuesto)."""
    out, s = [], 1
    for c, n in re.findall(r'C(\d) \((\d+)\)', desc):
        out.append((s, s + int(n), int(c))); s += int(n)
    return out


def continuation(key, n, label, algo, rd, tmp):
    gp, drv, md, _ = GPS[key]
    cmd = [sys.executable] + WRAP + ['--gp', gp, '--driver', drv, '--model-dir', md, '--lap', str(n), '--real-2024',
                                     '--seed', str(SEED), '--algo', algo, '--results-dir', os.path.join(rd, algo),
                                     '--following-gap-scale', '1.0', '--sim-future',
                                     '--out-pdf', os.path.join(tmp, f'{md}_{n}.pdf'),
                                     '--out-png', os.path.join(tmp, f'{md}_{n}.png')]
    txt = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout
    line = next(l for l in txt.splitlines() if l.strip().startswith(label + ' ') or l.strip().startswith(label + ':'))
    return stints_from_desc(line.split(':', 1)[1])


def real_plan(gp, driver):
    """Copia de ds_rollouts_realhist_20261024.real_plan (modo hist); ese modulo no se importa porque lee sys.argv."""
    d = S24[(S24.GP == gp) & (S24.Driver == driver.replace('Driver_', ''))].sort_values('LapNumber')
    num = lambda c: int(str(c).lstrip('C'))
    plan, prev = {0: num(d.Compound_Detail.iloc[0])}, d.iloc[0]
    for _, r in d.iloc[1:].iterrows():
        if r.TyreLife < prev.TyreLife or r.Compound_Detail != prev.Compound_Detail:
            plan[int(prev.LapNumber)] = num(r.Compound_Detail)
        prev = r
    return plan


def real_stints(plan, n_laps):
    """{0: c0, L: c, ...} -> tandas con la convencion de la Fig. 9 (la parada de L abre la tanda en L)."""
    laps = sorted(plan)
    starts = [1] + [l for l in laps if l > 0]
    comps = [plan[l] for l in laps]
    ends = starts[1:] + [n_laps + 1]
    return list(zip(starts, ends, comps))


def draw_bar(ax, y, stints, color_of, h, alpha_until=None):
    """Tandas como barras. 27-oct (pedido del autor, legibilidad al imprimir): el prefijo comun (vueltas < n) ya no va
    atenuado; la vuelta de la decision la marca solo el triangulo. Una tanda que cruza n sin parada (el DRL no sugiere
    parar) se dibuja continua: sin borde en n y con un solo numero (largo total de la tanda)."""
    for s, e, c in stints:
        ax.barh(y, e - s, left=s - 0.5, height=h, color=color_of[c], lw=0, zorder=3)
        ax.barh(y, e - s, left=s - 0.5, height=h, fill=False, edgecolor='black', linewidth=0.8, zorder=3.5)
        col = 'white' if color_of[c] == 'red' else 'black'
        ax.text((s + e - 1) / 2, y, str(e - s), ha='center', va='center', fontsize=FS_BAR, color=col, zorder=5)


# 27-oct (pedido del autor): SC sombreado como en las Figs. 9-10 (consistencia); VSC con achurado diagonal sin relleno
SC_COLOR, SC_ALPHA = '#8e7cc3', 0.35
YF_STYLE = {'VSC': dict(hatch='////', edgecolor='#5b3f9e')}


def draw_yf_hatch(ax, sc_laps, vsc_laps):
    for s, e in _contiguous_runs(sc_laps or []):
        ax.axvspan(s - 0.5, e + 0.5, color=SC_COLOR, alpha=SC_ALPHA, lw=0, zorder=0)
    for s, e in _contiguous_runs(vsc_laps or []):
        ax.axvspan(s - 0.5, e + 0.5, facecolor='none', edgecolor=YF_STYLE['VSC']['edgecolor'],
                   hatch=YF_STYLE['VSC']['hatch'], lw=0, zorder=0)


TITLE = {'Saudi Arabian': 'Saudi Arabia', 'Hungarian': 'Hungary', 'Belgian': 'Belgium', 'Dutch': 'Netherlands'}
NAMES3 = ['Hard', 'Medium', 'Soft']
H = 0.6


def gp_rows(key, rd, out, algo='PPO', scen='(ii)'):
    """Datos de un panel: tandas reales + una fila por vuelta con rec != real (continuaciones en cache JSON)."""
    gp, drv, md, name = GPS[key]
    m = json.load(open(os.path.join(rd, algo, f'{key}_real_lap1_meta.json')))
    L = pd.read_csv(os.path.join(rd, 'compare_laps_20261026.csv'))
    g = L[(L.gp == key) & (L.algo == algo) & (L.scen == scen)].sort_values('lap')
    cache_p = os.path.join(out, 'stints_cache_20261026.json')
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    tmp = os.path.join(out, 'stints_tmp'); os.makedirs(tmp, exist_ok=True)
    rows = [dict(label='Actual', stints=real_stints(real_plan(gp, drv), m['n_laps']), lap=None)]
    for r in g[g.real != g.rec].itertuples():
        ck = f'{key}|{algo}|{r.lap}|{r.rec}|{SEED}'
        if ck not in cache:
            cache[ck] = continuation(key, r.lap, r.rec, algo, rd, tmp)
            json.dump(cache, open(cache_p, 'w'), indent=0)
        st = [tuple(x) for x in cache[ck]]
        rows.append(dict(label=f'DRL at lap {r.lap}', stints=st, lap=r.lap, rec=r.rec.split(' (')[0],
                         pts_rec=r.pts_rec, pts_real=r.pts_real, d=-r.dpts_real_vs_rec, se2=r.dpts_2se, sig=r.signif))
        print(f'{key} lap {r.lap}: rec {r.rec} -> ' + ' | '.join(f'C{c} ({e - s})' for s, e, c in st), flush=True)
    gpn = gp.replace(' Grand Prix', '')
    return dict(key=key, title=f'{TITLE.get(gpn, gpn)} ({name})', meta=m, rows=group_rows(rows) if GROUP else rows)


def group_rows(rows):
    """Agrupa vueltas consecutivas con la misma recomendacion en una fila (pedido del autor, 26-oct): la barra es la
    continuacion de la primera vuelta del grupo, un triangulo por vuelta, y a la derecha la diferencia de la primera vuelta."""
    out = [rows[0]]
    for r in rows[1:]:
        q = out[-1]
        if q['lap'] is not None and q['rec'] == r['rec'] and r['lap'] == q['laps'][-1] + 1:
            q['laps'].append(r['lap']); q['ds'].append(r['d']); q['sigs'].append(r['sig'])
            q['label'] = f"DRL at laps {q['laps'][0]}\u2013{q['laps'][-1]}"
        else:
            out.append(dict(r, laps=[r['lap']], ds=[r['d']], sigs=[r['sig']]))
    return out


def draw_panel(ax, P, header=True, nrows=None, short=False, fs=9):
    m, rows = P['meta'], P['rows']
    n_laps, comps = m['n_laps'], m['available_compounds']
    color_of = {c: STINT_COLORS[i] for i, c in enumerate(comps)}
    draw_yf_hatch(ax, m['laps_sc_2024'], m['laps_vsc_2024'])
    nr = nrows or len(rows)
    frac = (nr - 0.4 - (len(rows) - 0.5)) / (nr + 0.2)   # SC/VSC solo sobre las filas usadas (eje y invertido)
    for pa in ax.patches:   # SC/VSC por encima de las barras (semitransparente), visibles en todas las filas
        pa.set_zorder(4.5)
        if hasattr(pa, 'set_height') and frac > 0:
            pa.set_y(frac); pa.set_height(1 - frac)
    tr = ax.get_yaxis_transform()   # x en fraccion del eje, y en datos
    for i, r in enumerate(rows):
        draw_bar(ax, i, r['stints'], color_of, H, alpha_until=r['lap'])
        if r['lap'] is None:
            continue
        # triangulo negro invertido justo encima de la barra, al inicio de la vuelta de la decision (n - 0.5; pedido del autor)
        for lp in r.get('laps', [r['lap']]):
            ax.scatter(lp - 0.5, i - H / 2 - 0.1, marker='v', s=TRI * (40 if len(r.get('laps', [0])) == 1 else 22),
                       color='black', zorder=6, clip_on=False)
        # grupo: diferencia de la primera vuelta (la de la barra dibujada) con +-2 SE (pedido del autor, 26-oct)
        lap_txt = f"\n(lap {r['laps'][0]})" if len(r.get('laps', [0])) > 1 else ''
        txt = f"{r['rec']}: {fmt_d(r['d'])} $\\pm$ {r['se2']:.1f}" + ('*' if r['sig'] else '') + lap_txt
        ax.text(1.02, i, txt,
                transform=tr, va='center', ha='left', fontsize=fs, clip_on=False)
    if header:
        ax.text(1.02, -0.62, 'Points: DRL $-$ actual', transform=tr, va='bottom', ha='left', fontsize=fs, clip_on=False)
    labels = [r['label'].replace('DRL at laps', 'Laps').replace('DRL at lap', 'Lap') if short else r['label'] for r in rows]
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=fs + 1)
    ax.set_ylim((nrows or len(rows)) - 0.4, -0.6)
    ax.set_xlim(0.5, n_laps + 0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(P['title'], fontsize=10, loc='left')


def legend_handles(panels):
    hs = [Patch(facecolor=STINT_COLORS[i], edgecolor='black', label=NAMES3[i]) for i in range(3)]
    if any(P['meta']['laps_sc_2024'] for P in panels):
        hs.append(Patch(facecolor=SC_COLOR, alpha=0.5, label='Safety car'))
    if any(P['meta']['laps_vsc_2024'] for P in panels):
        hs.append(Patch(facecolor='white', edgecolor=YF_STYLE['VSC']['edgecolor'], hatch=YF_STYLE['VSC']['hatch'], label='Virtual SC'))
    return hs


def save(fig, out, name):
    paper_fonts_20261021.apply(fig)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('->', os.path.join(out, f'{name}.pdf'))


def figure(P, out):
    fig, ax = plt.subplots(figsize=(10, 0.75 * len(P['rows']) + 1.3))
    draw_panel(ax, P)
    ax.set_xlabel('Lap', fontsize=10)
    ax.legend(handles=legend_handles([P]), loc='upper center', bbox_to_anchor=(0.5, -0.2 if len(P['rows']) < 4 else -0.1),
              ncol=5, fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, out, f"realhist_stints_{P['key']}")


def figure_all(panels, out):
    hr = [len(P['rows']) + 0.6 for P in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 0.42 * sum(hr) + 1.2), gridspec_kw=dict(height_ratios=hr))
    for k, (ax, P) in enumerate(zip(axes, panels)):
        draw_panel(ax, P, header=(k == 0))
    axes[-1].set_xlabel('Lap', fontsize=10)
    fig.tight_layout(h_pad=0.6)
    fig.legend(handles=legend_handles(panels), loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=9,
               frameon=False)
    save(fig, out, 'realhist_stints_all')


def figure_grid(panels, out, ncols=4):
    """Grilla en el orden de las Figs. 5-7 (ncols=4: 2 x 4 como la Fig. 7; ncols=2: 4 x 2, paneles el doble de anchos).
    En cada fila de la grilla, todos los paneles con el mismo numero de filas (el del GP con mas filas), para que las
    barras tengan la misma altura. Tamanos pensados para \\includegraphics[width=\\linewidth] (letras grandes, como la
    Fig. 7); sin el encabezado "Points: DRL - actual" en cada panel (lo explica la leyenda del paper)."""
    global FS_BAR, TRI
    FS_BAR, TRI = 13, 3.0
    nr = len(panels) // ncols
    rows_per = [max(len(P['rows']) for P in panels[k:k + ncols]) for k in range(0, len(panels), ncols)]
    width = 17 if ncols == 4 else 15
    fig, axes = plt.subplots(nr, ncols, figsize=(width, 0.62 * sum(rows_per) + 0.9 * nr + 0.8),
                             gridspec_kw=dict(height_ratios=rows_per, wspace=0.95 if ncols == 4 else 0.62,
                                              hspace=0.45 if ncols == 4 else 0.75))
    for k, (ax, P) in enumerate(zip(axes.flat, panels)):
        draw_panel(ax, P, header=False, nrows=rows_per[k // ncols], short=True, fs=14)
        gp, drv = GPS[P['key']][0], GPS[P['key']][1]
        ax.set_title(f"{gp.replace(' Grand Prix', ' GP')} ({drv[7:]})", fontsize=16, fontweight='bold', loc='left')
        if k >= len(panels) - ncols:
            ax.set_xlabel('Lap', fontsize=15)
    fig.legend(handles=legend_handles(panels), loc='upper center', bbox_to_anchor=(0.5, 0.02 if ncols == 4 else 0.04),
               ncol=5, fontsize=15, frameon=False)
    save(fig, out, 'realhist_stints_grid' if ncols == 4 else 'realhist_stints_grid4x2')
    FS_BAR, TRI = 9, 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', required=True)
    ap.add_argument('--gps', nargs='+', default=list(GPS))
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-group', action='store_true', help='una fila por vuelta (sin agrupar)')
    a = ap.parse_args()
    assert os.environ.get('F1_DS_FORCE') == 'hist' and os.environ.get('F1_TYRE_REUSE') == '1', 'entorno del barrido E4'
    rd = os.path.abspath(a.res)
    out = a.out or os.path.join(rd, 'figs')
    os.makedirs(out, exist_ok=True)
    global GROUP
    GROUP = not a.no_group
    panels = [gp_rows(k, rd, out) for k in a.gps]
    for P in panels:
        figure(P, out)
    if len(panels) > 1:
        figure_all(panels, out)
    if len(panels) == 8:
        figure_grid(panels, out)
        figure_grid(panels, out, ncols=2)


if __name__ == '__main__':
    main()
