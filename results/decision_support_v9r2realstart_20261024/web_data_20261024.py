"""2026-10-24 — Barrido A (realstart); copia de ../decision_support_v9r2w24belief_20261023/web_data_20261023.py con DSQ y tarjetas (keys_20261024.json); 25-oct: % de rollouts corregidos por la máscara (mask). Original: 2026-10-23 — Datos compactos (JSON) para la página web del barrido de apoyo a la decisión v9r2 (copia del script v8) (8 GP).

Por GP x algoritmo x escenario y por vuelta n: estado del agente (posición, compuesto, TyreLife), acción recomendada y,
por acción, posición final media, puntos medios, % top 10 y el histograma de la posición final en los M rollouts.
Agrega la estrategia y posición real 2024 del piloto (data/session_2024_V2.csv) y la fila de summary_20261024.csv.
Salida: web/ds_v9r2realstart_data_20261024.json
"""
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
SUM = pd.read_csv(os.path.join(D, 'summary_20261024.csv'))


def lapnum(p):
    return int(re.search(r'_lap(\d+)(_meta\.json|\.csv)$', p).group(1))


out = {'gps': {}, 'runs': {}}
for key, (gp, drv) in GPS.items():
    d = S24[(S24.GP == gp) & (S24.Driver == drv)].sort_values('LapNumber')
    m1 = json.load(open(f'{D}/PPO/{key}_real_lap1_meta.json'))
    out['gps'][key] = dict(name=gp.replace(' Grand Prix', ''), driver=drv, n_laps=m1['n_laps'],
                           start_pos=m1['agent_start_position'], sc=m1['laps_sc_2024'], vsc=m1['laps_vsc_2024'],
                           labels=m1['action_labels'],
                           real=dict(lap=d.LapNumber.astype(int).tolist(), pos=d.Position.fillna(0).astype(int).tolist(),
                                     comp=d.Compound_Detail.astype(str).tolist(), tl=d.TyreLife.fillna(0).astype(int).tolist()))
    for algo in ['PPO', 'RecurrentPPO']:
        for scen, sc in [('(i)', 'realyf'), ('(ii)', 'real')]:
            laps = []
            metas = sorted(glob.glob(f'{D}/{algo}/{key}_{sc}_lap*_meta.json'), key=lapnum)
            for m in metas:
                meta = json.load(open(m))
                df = pd.read_csv(m.replace('_meta.json', '.csv'))
                acts = {}
                for a, x in df.groupby('action'):
                    h = np.bincount(x.final_position.clip(1, 20).astype(int), minlength=21)[1:]
                    acts[int(a)] = dict(pos=round(x.final_position.mean(), 3), pts=round(x.points.mean(), 2),
                                        top10=round((x.final_position <= 10).mean(), 3),
                                        stops=round(x.n_stops.mean(), 2), hist=h.tolist(),
                                        mask=round((x.masked_hits > 0).mean(), 3))   # 25-oct: % rollouts corregidos por la mascara
                st = meta['agent_state_at_lap']
                laps.append(dict(lap=meta['lap'], pos=st['position'], comp=st['compound'], tl=st['tyrelife'],
                                 rec=meta['recommended_action'], acts=acts))
            mask_rec = sum(L['acts'][L['rec']]['mask'] for L in laps) / len(laps)   # 25-oct (mask_check_20261025.py)
            last = json.load(open(metas[-1]))
            row = SUM[(SUM.algo == algo) & (SUM.scen == scen) & (SUM.gp == key)].iloc[0]
            out['runs'][f'{key}|{algo}|{sc}'] = dict(
                laps=laps, history=last['history'], M=last['M'],
                summary=dict(start=row['start'], pits=row['pits (vuelta:comp)'], rec_best=int(row['rec=mejor %']),
                             regret=float(row['regret pos medio']), miss=int(row['n rec=no pit, mejor=pit']),
                             bad=row['rec=pit, mejor=no pit (vuelta(+pos))'], real_start=row['real 2024 start'],
                             real_pits=row['real 2024 pits'], dsq=bool(row['DSQ']), mask_rec=round(mask_rec, 3)))
out['keys'] = json.load(open(f'{D}/keys_20261024.json'))   # tarjetas del encabezado (texto del analisis)
os.makedirs(f'{D}/web', exist_ok=True)
json.dump(out, open(f'{D}/web/ds_v9r2realstart_data_20261024.json', 'w'), separators=(',', ':'))
print(os.path.getsize(f'{D}/web/ds_v9r2realstart_data_20261024.json') / 1e6, 'MB')
# Página: inyecta el JSON en la plantilla -> web/decisiones_v9r2realstart_20261024.html
tpl = open(f'{D}/web/template_20261024.html').read()
open(f'{D}/web/decisiones_v9r2realstart_20261024.html', 'w').write(tpl.replace('/*DATA*/null', json.dumps(out, separators=(',', ':'))))
