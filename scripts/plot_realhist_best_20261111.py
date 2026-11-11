"""2026-11-11 (propuesta del autor): Fig. 11 de un solo GP, sin agrupar, con una columna nueva.

Una fila por cada vuelta en que la recomendacion PPO difiere de la decision real (misma fila 'Actual' y mismas
continuaciones que scripts/plot_realhist_stints_20261026.py, que se importa). A la derecha, dos columnas:
  - DRL: accion recomendada y diferencia pareada media de puntos 'DRL - actual' +-2 SE (igual que la Fig. 11);
  - Best of the four actions: la accion con mas puntos medios entre las cuatro evaluadas con M = 500 rollouts en
    esa vuelta (columna 'best_pts' del CSV del barrido) y su diferencia con la decision real +-2 SE ('gain_best_vs_real',
    'gain_2se'). 'Same as DRL' si coincide con la recomendacion; si coincide con la decision real se escribe igual que las demas
    (p. ej. 'No pit: 0.0 +- 0.0'; pedido del autor, 11-nov).
Opcion (a) del autor (11-nov): con --vals <carpeta con PPO/*_real_lap*.csv> (repeticion independiente,
results/decision_support_v10gsmrealhist_20261102/verif_best_20261111), la accion 'mejor' se ELIGE con el barrido original
(compare_laps) pero las DOS columnas se MIDEN con la repeticion (diferencia pareada media +-2 SE), para que la columna del
mejor no arrastre el sesgo de seleccion. Sin --vals, todo sale del barrido original como antes.
No corre rollouts nuevos (salvo las continuaciones de las vueltas que no estuvieran en el cache, una por vuelta).
Mismo entorno que results/decision_support_v10gsmrealhist_20261102/fig11_20261103.sh, y ademas:
    python scripts/plot_realhist_best_20261111.py --res results/decision_support_v10gsmrealhist_20261102 \
        --laps compare_laps_20261103.csv --gp bahrain_alo --out <carpeta>
"""
import argparse, os, shutil, sys
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_realhist_stints_20261026 as B
from plot_realhist_audit_20261026 import fmt_d


def best_info(rd, key, algo='PPO', scen='(ii)'):
    L = pd.read_csv(os.path.join(rd, B.LAPS))
    g = L[(L.gp == key) & (L.algo == algo) & (L.scen == scen)]
    return {int(r.lap): r for r in g.itertuples()}


def paired(vals, key, lap, a, b):
    """Diferencia pareada media de puntos a - b y 2 SE en la repeticion independiente."""
    p = pd.read_csv(os.path.join(vals, 'PPO', f'{key}_real_lap{lap}.csv')).pivot(
        index='rollout', columns='action_label', values='points')
    d = p[a] - p[b]
    return d.mean(), 2 * d.std(ddof=1) / len(d) ** 0.5


def draw(P, info, out, fs=9, vals=None):
    rows = P['rows']
    if vals:
        for r in rows:
            if r['lap'] is None:
                continue
            x = info[r['lap']]
            r['d'], r['se2'] = paired(vals, P['key'], r['lap'], x.rec, x.real)
            r['sig'] = abs(r['d']) > r['se2']
    # 11-nov (tarde, pedido del autor): mas compacta en vertical (filas mas juntas, barras mas bajas, letra y triangulos
    # mas chicos, leyenda mas cerca del eje)
    fig, ax = plt.subplots(figsize=(8, 0.27 * len(rows) + 0.95))
    fig.subplots_adjust(left=0.15, right=0.60, top=0.90, bottom=0.21)   # 11-nov: eje fijo, columnas de texto a la derecha
    B.draw_panel(ax, dict(P, rows=[dict(r, rec=r.get('rec', ''), se2=r.get('se2', 0), sig=r.get('sig', False),
                                        d=r.get('d', 0)) for r in rows]), header=False, fs=fs, short=True)   # 11-nov: etiquetas "Lap n" (pedido del autor)
    # draw_panel escribe la columna DRL en x = 1.02; se agrega la del mejor en x = 1.30
    tr = ax.get_yaxis_transform()
    ax.text(1.02, -0.85, 'DRL $-$ actual', transform=tr, va='bottom', ha='left', fontsize=fs, fontweight='bold',
            clip_on=False)
    ax.text(1.40, -0.85, 'Best of the four $-$ actual', transform=tr, va='bottom', ha='left', fontsize=fs,
            fontweight='bold', clip_on=False)
    for i, r in enumerate(rows):
        if r['lap'] is None:
            continue
        x = info[r['lap']]
        if x.best_pts == x.rec:
            txt = 'Same as DRL'
        else:
            g, se = ((x.gain_best_vs_real, x.gain_2se) if not vals
                     else paired(vals, P['key'], r['lap'], x.best_pts, x.real))
            sig = x.best_signif if not vals else (se > 0 and abs(g) > se)
            txt = f"{x.best_pts.split(' (')[0]}: {fmt_d(g)} $\\pm$ {se:.1f}" + ('*' if sig else '')
        ax.text(1.40, i, txt, transform=tr, va='center', ha='left', fontsize=fs, clip_on=False)
    ax.set_xlabel('Lap', fontsize=fs + 1)
    ax.tick_params(axis='x', labelsize=fs)
    ax.legend(handles=B.legend_handles([P]), loc='upper center', handlelength=1.5, columnspacing=1.5, bbox_to_anchor=(0.5, -0.19), ncol=5, fontsize=fs,
              frameon=False)
    B.save(fig, out, f"realhist_best_{P['key']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', required=True)
    ap.add_argument('--laps', default='compare_laps_20261103.csv')
    ap.add_argument('--gp', default='bahrain_alo')
    ap.add_argument('--out', required=True)
    ap.add_argument('--vals', default=None, help='repeticion independiente (opcion a)')
    ap.add_argument('--no-title', action='store_true')
    a = ap.parse_args()
    assert os.environ.get('F1_DS_FORCE') == 'hist' and os.environ.get('F1_TYRE_REUSE') == '1', 'entorno del barrido E4'
    rd = os.path.abspath(a.res)
    os.makedirs(a.out, exist_ok=True)
    # cache de continuaciones: se copia el del barrido para no modificarlo
    src = os.path.join(rd, 'figs', 'stints_cache_20261026.json')
    dst = os.path.join(a.out, 'stints_cache_20261026.json')
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)
    B.LAPS, B.GROUP = a.laps, False
    P = B.gp_rows(a.gp, rd, a.out)
    if a.no_title:
        P['title'] = ''
    B.FS_BAR, B.TRI, B.H = 8.5, 0.45, 0.6   # 11-nov: tamanos para el cuerpo del paper (ancho completo)
    draw(P, best_info(rd, a.gp), a.out, fs=8.5, vals=os.path.abspath(a.vals) if a.vals else None)


if __name__ == '__main__':
    main()
