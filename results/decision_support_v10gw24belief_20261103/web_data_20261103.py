"""2026-11-03 — Página web del barrido de apoyo a la decisión v10g, solo PPO (a partir de
../decision_support_v9r2w24belief_20261023/web_data_20261023.py y su plantilla). Agrega, por acción y vuelta, la diferencia
pareada de puntos contra la recomendada (+-2 SE) y la tabla de candidatos para las Figs. 8-10 (candidates_20261103.csv).
Requiere summary_20261103.py y candidates_20261103.py. Salida: web/decisiones_v10g_20261103.html"""
import glob, json, os, re
import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(D, '..', '..'))
GPS = {'bahrain_alo': ('Bahrain Grand Prix', 'ALO'), 'belgian_gas': ('Belgian Grand Prix', 'GAS'),
       'dutch_str': ('Dutch Grand Prix', 'STR'), 'hungarian_tsu': ('Hungarian Grand Prix', 'TSU'),
       'miami_oco': ('Miami Grand Prix', 'OCO'), 'saudi_arabian_nor': ('Saudi Arabian Grand Prix', 'NOR'),
       'singapore_hul': ('Singapore Grand Prix', 'HUL'), 'united_states_mag': ('United States Grand Prix', 'MAG')}
S24 = pd.read_csv(os.path.join(ROOT, 'data', 'session_2024_V2.csv'))
SUM = pd.read_csv(os.path.join(D, 'summary_20261103.csv'))
lapnum = lambda p: int(re.search(r'_lap(\d+)(_meta\.json|\.csv)$', p).group(1))

out = {'gps': {}, 'runs': {}}
for key, (gp, drv) in GPS.items():
    d = S24[(S24.GP == gp) & (S24.Driver == drv)].sort_values('LapNumber')
    m1 = json.load(open(f'{D}/PPO/{key}_real_lap1_meta.json'))
    out['gps'][key] = dict(name=gp.replace(' Grand Prix', ''), driver=drv, n_laps=m1['n_laps'],
                           start_pos=m1['agent_start_position'], sc=m1['laps_sc_2024'], vsc=m1['laps_vsc_2024'],
                           labels=m1['action_labels'],
                           real=dict(lap=d.LapNumber.astype(int).tolist(), pos=d.Position.fillna(0).astype(int).tolist(),
                                     comp=d.Compound_Detail.astype(str).tolist(), tl=d.TyreLife.fillna(0).astype(int).tolist()))
    for scen, sc in [('(i)', 'realyf'), ('(ii)', 'real')]:
        laps = []
        metas = sorted(glob.glob(f'{D}/PPO/{key}_{sc}_lap*_meta.json'), key=lapnum)
        for m in metas:
            meta = json.load(open(m))
            df = pd.read_csv(m.replace('_meta.json', '.csv'))
            P = df.pivot(index='rollout', columns='action', values='points'); rec = meta['recommended_action']
            acts = {}
            for a, x in df.groupby('action'):
                h = np.bincount(x.final_position.clip(1, 20).astype(int), minlength=21)[1:]
                z = P[a] - P[rec]
                acts[int(a)] = dict(pos=round(x.final_position.mean(), 3), pts=round(x.points.mean(), 2),
                                    top10=round((x.final_position <= 10).mean(), 3), stops=round(x.n_stops.mean(), 2),
                                    dp=round(z.mean(), 2), se=round(2 * z.std() / np.sqrt(len(z)), 2), hist=h.tolist())
            st = meta['agent_state_at_lap']
            laps.append(dict(lap=meta['lap'], pos=st['position'], comp=st['compound'], tl=st['tyrelife'], rec=rec, acts=acts))
        last = json.load(open(metas[-1]))
        row = SUM[(SUM.scen == scen) & (SUM.gp == key)].iloc[0]
        out['runs'][f'{key}|PPO|{sc}'] = dict(
            laps=laps, history=last['history'], M=last['M'],
            summary=dict(start=row['start'], pits=row['pits (vuelta:comp)'], rec_best=int(row['rec=mejor %']),
                         regret=float(row['regret pos medio']), miss=int(row['n rec=no pit, mejor=pit']),
                         bad=row['rec=pit, mejor=no pit (vuelta(+pos))'], real_start=row['real 2024 start'],
                         real_pits=row['real 2024 pits']))
C = pd.read_csv(os.path.join(D, 'candidates_20261103.csv'))
out['cands'] = [dict(gp=r.gp, scen='realyf' if r.scen == '(i)' else 'real', sl=r.scen, lap=int(r.lap), pos=int(r.pos), age=int(r.age),
                     flag=r.flag if isinstance(r.flag, str) else '', top=[float(r.top10min), float(r.top10max)], pct=int(r.pct_pts))
                for r in C.itertuples()]
tpl = open(f'{D}/web/template_20261103.html').read()
open(f'{D}/web/decisiones_v10g_20261103.html', 'w').write(tpl.replace('/*DATA*/null', json.dumps(out, separators=(',', ':'))))
print(os.path.getsize(f'{D}/web/decisiones_v10g_20261103.html') / 1e6, 'MB')
