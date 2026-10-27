"""2026-10-26 — E4: copia de ../decision_support_v9r2realhist_20261024/mask_check_20261025.py con el barrido B de los agentes post-entrenados con largada sorteada (Bsm, esta carpeta) además de v9r2, A y B. Original: 2026-10-25 — Control de fiabilidad de la política en los barridos A (realstart) y B (realhist) frente al barrido v9r2.

En cada rollout la política juega desde la vuelta n+1 hasta el final con la máscara de legalidad (legal_mask = True en
decision_support_rollouts.py): masked_hits > 0 indica que la política propuso al menos una acción ilegal (parar con el
mismo compuesto sin haber cumplido la regla de dos compuestos, o no parar en la última vuelta posible) y la máscara la
reemplazó. Es la señal directa de que el agente está en estados que no sabe manejar.
Por GP x algoritmo x escenario x barrido:
    rec=mejor %     vueltas en que la recomendación (argmax) es la acción de mayor puntaje medio;
    perdida pts     puntos medios de la mejor menos los de la recomendada, promedio sobre las vueltas;
    mascara rec %   % de rollouts de la acción recomendada con al menos una corrección (promedio sobre las vueltas);
    mascara todas % lo mismo para todas las acciones.
Salidas (esta carpeta): mask_check_20261026.csv (agregado) y mask_check_laps_20261026.csv (por vuelta y acción).
"""
import glob, json, os
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(D)
VERS = {'v9r2': 'decision_support_v9r2w24belief_20261023', 'A': 'decision_support_v9r2realstart_20261024',
        'B': 'decision_support_v9r2realhist_20261024', 'Bsm': 'decision_support_v9r2smrealhist_20261026'}
rows, laps = [], []
for v, d in VERS.items():
    for f in sorted(glob.glob(f'{R}/{d}/*/*_lap*.csv')):
        algo, b = f.split(os.sep)[-2], os.path.basename(f)[:-4]
        stem, lap = b.rsplit('_lap', 1)
        sc = 'realyf' if stem.endswith('_realyf') else 'real'
        gp = stem[:-len('_' + sc)]
        x = pd.read_csv(f, usecols=['action', 'action_label', 'masked_hits', 'points'])
        rec = json.load(open(f[:-4] + '_meta.json'))['recommended_action']
        g = x.groupby('action').agg(label=('action_label', 'first'), pts=('points', 'mean'),
                                    mask=('masked_hits', lambda s: (s > 0).mean()))
        for a, r in g.iterrows():
            laps.append(dict(ver=v, gp=gp, algo=algo, scen='(i)' if sc == 'realyf' else '(ii)', lap=int(lap),
                             action=int(a), label=r.label, pts=round(r.pts, 3), mask=round(r['mask'], 3), rec=int(a) == rec))
        rows.append(dict(ver=v, gp=gp, algo=algo, scen='(i)' if sc == 'realyf' else '(ii)', lap=int(lap),
                         rec_best=g.pts.idxmax() == rec, loss=g.pts.max() - g.pts[rec],
                         mask_rec=g['mask'][rec], mask_all=(x.masked_hits > 0).mean()))
o = pd.DataFrame(rows)
agg = (o.groupby(['gp', 'algo', 'scen', 'ver'])
       .agg(**{'rec=mejor %': ('rec_best', lambda s: round(100 * s.mean())), 'perdida pts': ('loss', 'mean'),
               'mascara rec %': ('mask_rec', lambda s: 100 * s.mean()), 'mascara todas %': ('mask_all', lambda s: 100 * s.mean())})
       .round(2).reset_index())
agg.to_csv(f'{D}/mask_check_20261026.csv', index=False)
pd.DataFrame(laps).to_csv(f'{D}/mask_check_laps_20261026.csv', index=False)
pd.set_option('display.width', 200)
print(o.groupby('ver').agg(rec_best=('rec_best', 'mean'), loss=('loss', 'mean'), mask_rec=('mask_rec', 'mean'),
                           mask_all=('mask_all', 'mean')).round(3))
print(agg.pivot_table(index=['gp', 'algo'], columns='ver', values='mascara rec %').round(1))
