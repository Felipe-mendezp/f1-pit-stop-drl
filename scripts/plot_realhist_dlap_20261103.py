"""2026-11-03 - Diferencia de puntos por vuelta en una carrera del analisis B (pedido del autor): hace visible que,
en un grupo de vueltas con la misma recomendacion, el valor de seguirla cambia de vuelta en vuelta. Lee el CSV del
barrido (compare_laps_*.csv, M = 500 rollouts con banderas sorteadas), de modo que no corre nada nuevo: dibuja
d = pts_rec - pts_real con +-2 SE pareado. Las vueltas en que la recomendacion coincide con la decision real dan
cero por construccion y se marcan aparte.
    F1_PAPER_FONTS=1 python scripts/plot_realhist_dlap_20261103.py --res results/decision_support_v10gsmrealhist_20261102 \
        --laps compare_laps_20261103.csv --gp bahrain_alo --from-lap 33 --to-lap 42
"""
import argparse, os, sys
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_fonts_20261021

TITLE = {'bahrain_alo': 'Bahrain (Fernando Alonso)', 'saudi_arabian_nor': 'Saudi Arabia (Lando Norris)',
         'miami_oco': 'Miami (Esteban Ocon)', 'hungarian_tsu': 'Hungary (Yuki Tsunoda)',
         'belgian_gas': 'Belgium (Pierre Gasly)', 'dutch_str': 'Netherlands (Lance Stroll)',
         'singapore_hul': 'Singapore (Nico Hülkenberg)', 'united_states_mag': 'United States (Kevin Magnussen)'}

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--res', required=True)
ap.add_argument('--laps', default='compare_laps_20261103.csv')
ap.add_argument('--gp', default='bahrain_alo')
ap.add_argument('--algo', default='PPO')
ap.add_argument('--scen', default='(ii)')
ap.add_argument('--from-lap', type=int, default=33)
ap.add_argument('--to-lap', type=int, default=42)
ap.add_argument('--out', default=None, help='sin extension (default <res>/figs/realhist_dlap_<gp>)')
a = ap.parse_args()

L = pd.read_csv(os.path.join(a.res, a.laps))
g = L[(L.gp == a.gp) & (L.algo == a.algo) & (L.scen == a.scen) &
      (L.lap.between(a.from_lap, a.to_lap))].sort_values('lap')
assert len(g), (a.gp, a.algo, a.scen)
d = -g.dpts_real_vs_rec
dif = g.real != g.rec   # vueltas en que la recomendacion difiere de la decision real

fig, ax = plt.subplots(figsize=(5.4, 2.7))
ax.axhline(0, color='0.6', lw=0.8, zorder=1)
ax.errorbar(g.lap[dif], d[dif], yerr=g.dpts_2se[dif], fmt='o-', ms=4.5, lw=1.3, capsize=3,
            color='black', zorder=3, label='Recommendation differs')
ax.plot(g.lap[~dif], d[~dif], 'o', ms=4.5, mfc='white', mec='black', mew=1.0, zorder=3,
        label='Recommendation coincides')
ax.set_xlabel('Lap')
ax.set_ylabel('Points: DRL $-$ actual')
ax.set_xticks(range(a.from_lap, a.to_lap + 1))
ax.set_xlim(a.from_lap - 0.5, a.to_lap + 0.5)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper left', frameon=False, fontsize=9)
paper_fonts_20261021.apply(fig)
fig.tight_layout()
out = a.out or os.path.join(a.res, 'figs', f'realhist_dlap_{a.gp}')
os.makedirs(os.path.dirname(out), exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(f'{out}.{ext}', dpi=300, bbox_inches='tight')
print('->', out + '.pdf')
