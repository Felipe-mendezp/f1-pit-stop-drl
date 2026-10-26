"""2026-10-24 — Resumen del barrido B (realhist: prefijo forzado, ver run_20261024.sh en ../decision_support_v9r2realhist_20261024/); copia de ../decision_support_v9r2w24belief_20261023/summary_20261023.py. Original: 2026-10-23 — Resumen del barrido de apoyo a la decisión v9r2 (copia de ../decision_support_v8w24belief_20261019/summary_20261023.py; solo cambian nombres de archivo) (8 GP, PPO y RPPO,

Por GP x algoritmo x escenario:
  - estrategia del agente (vueltas de parada:compuesto, del 'history' del meta de la última vuelta) y la real 2024;
  - rec = acción del agente en la vuelta n (argmax de la política); mejor = acción con menor posición final media
    en los M rollouts; regret = pos media(rec) - pos media(mejor);
  - cuántas vueltas con rec = no parar y mejor = parar, y las vueltas con rec = parar y mejor = no parar.
Salida: summary_20261024.csv (mismas columnas que v5h summary_diag.csv + real 2024) y singapore_lap18_20261024.csv.
"""
import glob, json, os, re
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(D, '..', '..'))
GPS = {'bahrain_alo': ('Bahrain Grand Prix', 'ALO'), 'belgian_gas': ('Belgian Grand Prix', 'GAS'),
       'dutch_str': ('Dutch Grand Prix', 'STR'), 'hungarian_tsu': ('Hungarian Grand Prix', 'TSU'),
       'miami_oco': ('Miami Grand Prix', 'OCO'), 'saudi_arabian_nor': ('Saudi Arabian Grand Prix', 'NOR'),
       'singapore_hul': ('Singapore Grand Prix', 'HUL'), 'united_states_mag': ('United States Grand Prix', 'MAG')}
S24 = pd.read_csv(os.path.join(ROOT, 'data', 'session_2024_V2.csv'))


def real_strategy(gp, drv):
    d = S24[(S24.GP == gp) & (S24.Driver == drv)].sort_values('LapNumber')
    out = [str(d.Compound_Detail.iloc[0])]
    prev = d.iloc[0]
    for _, r in d.iloc[1:].iterrows():
        if r.TyreLife < prev.TyreLife or r.Compound_Detail != prev.Compound_Detail:
            out.append(f'{int(prev.LapNumber)}:{r.Compound_Detail}')
        prev = r
    return out[0], ' '.join(out[1:])


def lapnum(p):
    return int(re.search(r'_lap(\d+)(_meta\.json|\.csv)$', p).group(1))


rows, sing = [], []
for algo in ['PPO', 'RecurrentPPO']:
    for scen, sc in [('(i)', 'realyf'), ('(ii)', 'real')]:
        for key, (gp, drv) in GPS.items():
            metas = sorted(glob.glob(f'{D}/{algo}/{key}_{sc}_lap*_meta.json'), key=lapnum)
            last = json.load(open(metas[-1]))
            hist = last['history']
            start = f'C{hist[0][1]}'
            pits = []
            for i, (l, c, p) in enumerate(hist):
                if p and l >= 1:
                    nc = hist[i + 1][1] if i + 1 < len(hist) else last['agent_state_at_lap']['compound']
                    pits.append(f'{l}:C{nc}')
            comps = {c for _, c, _ in hist} | {last['agent_state_at_lap']['compound']}
            nbest = n_miss = 0; reg = []; bad = []
            for m in metas:
                meta = json.load(open(m)); n = meta['lap']
                df = pd.read_csv(m.replace('_meta.json', '.csv'))
                g = df.groupby('action').final_position.mean()
                rec = meta['recommended_action']; best = g.idxmin()
                nbest += rec == best; reg.append(g[rec] - g[best])
                if rec == 0 and best != 0: n_miss += 1
                if rec != 0 and 0 in g.index and best == 0: bad.append(f'{n}(+{g[rec] - g[0]:.1f})')
                if key == 'singapore_hul' and n == 18:
                    for a, x in df.groupby('action'):
                        sing.append(dict(algo=algo, scen=scen, action=x.action_label.iloc[0], rec=a == rec,
                                         pos_media=round(x.final_position.mean(), 2), pts_media=round(x.points.mean(), 2),
                                         top10=round((x.final_position <= 10).mean(), 3), paradas=round(x.n_stops.mean(), 2)))
            rs, rp = real_strategy(gp, drv)
            rows.append({'algo': algo, 'scen': scen, 'gp': key, 'N': len(metas), 'start': start,
                         'pits (vuelta:comp)': ' '.join(pits), 'DSQ': len(comps) < 2,
                         'rec=mejor %': round(100 * nbest / len(metas)), 'regret pos medio': round(sum(reg) / len(reg), 2),
                         'n rec=no pit, mejor=pit': n_miss,
                         'rec=pit, mejor=no pit (vuelta(+pos))': ' '.join(bad) or '—',
                         'real 2024 start': rs, 'real 2024 pits': rp})
S = pd.DataFrame(rows); S.to_csv(f'{D}/summary_20261024.csv', index=False)
pd.DataFrame(sing).to_csv(f'{D}/singapore_lap18_20261024.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(S.to_string(index=False)); print(); print(pd.DataFrame(sing).to_string(index=False))
