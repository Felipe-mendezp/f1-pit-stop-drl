"""2026-11-03 — Candidatos para el caso ilustrativo de las Figs. 8-10: por GP x escenario, cada vuelta de parada del
agente: puntos por acción, diferencia pareada contra la recomendada (+-2 SE), top 10; y coincidencia por puntos en toda la carrera.
Salida: candidates_20261103.csv"""
import glob, json, os, re
import numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__))
L = lambda p: int(re.search(r'_lap(\d+)(_meta\.json|\.csv)$', p).group(1))
keys = sorted({re.sub(r'_real.*', '', os.path.basename(f)) for f in glob.glob(f'{D}/PPO/*_lap*.csv')})
rows = []
for k in keys:
    for sc, name in (('realyf', '(i)'), ('real', '(ii)')):
        ms = sorted(glob.glob(f'{D}/PPO/{k}_{sc}_lap*_meta.json'), key=L)
        last = json.load(open(ms[-1]))
        H = last['history']; stops = [h[0] for h in H if h[2] and h[0] >= 1]
        tt = []; pp = []
        for m in ms:
            x = pd.read_csv(m.replace('_meta.json', '.csv')); me = json.load(open(m)); r = me['recommended_action']
            tt.append(int(x.groupby('action').points.mean().idxmax()) == r); pp.append(int(x.groupby('action').final_position.mean().idxmin()) == r)
        for s in stops:
            m = f'{D}/PPO/{k}_{sc}_lap{s}_meta.json'; me = json.load(open(m)); x = pd.read_csv(m.replace('_meta.json', '.csv'))
            P = x.pivot(index='rollout', columns='action', values='points'); r = me['recommended_action']; st = me['agent_state_at_lap']
            d = dict(gp=k, scen=name, lap=s, n=me['n_laps'], pos=st['position'], comp=st['compound'], age=st['tyrelife'],
                     flag='SC' if s in me['laps_sc_2024'] else ('VSC' if s in me['laps_vsc_2024'] else ''), rec=me['action_labels'][str(r)],
                     pct_pts=round(100 * np.mean(tt)), pct_pos=round(100 * np.mean(pp)), n_stops_agent=len(stops), stops=' '.join(map(str, stops)))
            top = x.groupby('action').final_position.apply(lambda v: (v <= 10).mean())
            d['top10min'] = round(top.min(), 2); d['top10max'] = round(top.max(), 2)
            for a in P.columns:
                lab = {0: 'stay', 1: 'H', 2: 'M', 3: 'S'}[a]; d['pts_' + lab] = round(P[a].mean(), 2)
                if a != r:
                    z = P[a] - P[r]; d['d_' + lab] = f'{z.mean():+.2f}±{2 * z.std() / np.sqrt(len(z)):.2f}'
            rows.append(d)
T = pd.DataFrame(rows); T.to_csv(f'{D}/candidates_20261103.csv', index=False)
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
print(T[['gp', 'scen', 'lap', 'n', 'pos', 'comp', 'age', 'flag', 'rec', 'pts_stay', 'pts_H', 'pts_M', 'pts_S', 'd_stay', 'd_H', 'd_M', 'd_S', 'top10min', 'top10max', 'pct_pts', 'pct_pos']].to_string(index=False))
