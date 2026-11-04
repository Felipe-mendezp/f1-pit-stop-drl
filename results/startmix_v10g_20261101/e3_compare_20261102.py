"""2026-11-02 — E3: compara las politicas PPO post-entrenadas (startmix v10gsm) con las originales v10g en los dos
escenarios de las Figs. 6 y 7 (500 simulaciones por GP). Original: results/world2024_v10g_20261031; post-entrenado:
results/startmix_v10g_20261101/e3_world2024_20261102. Escribe e3_compare_20261102.csv e imprime el resumen con los
numeros que cita el apendice app:realhist: promedio de los 8 GP por escenario y DSQ maximo."""
import os, pandas as pd, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORIG = os.path.join(ROOT, 'results', 'world2024_v10g_20261031')
POST = os.path.join(ROOT, 'results', 'startmix_v10g_20261101', 'e3_world2024_20261102')
GPS = ['Bahrain_ALO', 'Saudi_Arabian_NOR', 'Miami_OCO', 'Hungarian_TSU', 'Belgian_GAS', 'Dutch_STR',
       'Singapore_HUL', 'United_States_MAG']
rows = []
for fig in ('fig6_2024_realdnf', 'fig7_2024'):
    for tag, base in (('orig', ORIG), ('post', POST)):
        for g in GPS:
            p = os.path.join(base, fig, g + '.csv')
            if not os.path.exists(p):
                print('falta', p); continue
            d = pd.read_csv(p)
            for algo, x in d.groupby('algo'):
                if algo not in ('PPO', 'QMIP'):
                    continue
                pos = np.where(x.changed_compound, x.final_position, 20)   # DSQ = P20, como evaluate_all_gps
                rows.append(dict(gp=g, algo=algo, pos=np.mean(pos), dsq=1 - x.changed_compound.mean(),
                                 n=len(x), fig=fig, tag=tag))
o = pd.DataFrame(rows)
o.round(4).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'e3_compare_20261102.csv'), index=False)
pd.set_option('display.width', 200)
for fig in ('fig6_2024_realdnf', 'fig7_2024'):
    f = o[o.fig == fig]
    t = f.pivot_table(index='gp', columns=['algo', 'tag'], values='pos').reindex(GPS)
    t['delta_PPO'] = t[('PPO', 'post')] - t[('PPO', 'orig')]
    print('\n===', fig, '===')
    print(t.round(3).to_string())
    print('promedio 8 GP  PPO orig %.3f  PPO post %.3f  delta %+.3f  QMIP %.3f' % (
        t[('PPO', 'orig')].mean(), t[('PPO', 'post')].mean(),
        t[('PPO', 'post')].mean() - t[('PPO', 'orig')].mean(), t[('QMIP', 'orig')].mean()))
    for tag in ('orig', 'post'):
        s = f[(f.algo == 'PPO') & (f.tag == tag)]
        print('  DSQ maximo PPO %s: %.1f%% (%s); GP con DSQ > 0: %d' % (
            tag, 100 * s.dsq.max(), s.loc[s.dsq.idxmax(), 'gp'], (s.dsq > 0).sum()))
    q = f[f.algo == 'QMIP'].pivot_table(index='gp', columns='tag', values='pos')
    print('  QMIP identica entre orig y post:', bool(np.allclose(q['orig'], q['post'])))
