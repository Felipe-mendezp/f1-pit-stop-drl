"""2026-10-24 — Barrido B (historia real): decision real 2024 del piloto frente a la recomendacion del DRL, vuelta a vuelta.

En cada vuelta n el prefijo reproduce la largada y las paradas reales hasta n-1 (scripts/ds_rollouts_realhist_20261024.py);
la accion real en n es parar al compuesto real si n es una vuelta de entrada a pits real, si no, no parar. Por vuelta:
rec (argmax de la politica), mejor (mayor puntaje medio en los M rollouts), diferencias pareadas de puntos y posicion de la
accion real frente a rec, y ganancia pareada de la mejor accion frente a la real (mismo rollout = mismas semillas; signif. =
mas de 2 errores estandar). Salidas:
  compare_laps_20261024.csv    una fila por GP x algoritmo x escenario x vuelta;
  compare_summary_20261024.csv una fila por GP x algoritmo x escenario.
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


def plan(gp, drv):
    d = S24[(S24.GP == gp) & (S24.Driver == drv)].sort_values('LapNumber')
    p, prev = {0: int(d.Compound_Detail.iloc[0][1:])}, d.iloc[0]
    for _, r in d.iloc[1:].iterrows():
        if r.TyreLife < prev.TyreLife or r.Compound_Detail != prev.Compound_Detail:
            p[int(prev.LapNumber)] = int(r.Compound_Detail[1:])
        prev = r
    return p


lapnum = lambda p: int(re.search(r'_lap(\d+)_meta\.json$', p).group(1))
rows = []
for key, (gp, drv) in GPS.items():
    P = plan(gp, drv)
    for algo in ['PPO', 'RecurrentPPO']:
        for scen, sc in [('(i)', 'realyf'), ('(ii)', 'real')]:
            for m in sorted(glob.glob(f'{D}/{algo}/{key}_{sc}_lap*_meta.json'), key=lapnum):
                meta = json.load(open(m)); n = meta['lap']; lab = {int(k): v for k, v in meta['action_labels'].items()}
                comp = meta['available_compounds']
                real = comp.index(P[n]) + 1 if n in P else 0
                df = pd.read_csv(m.replace('_meta.json', '.csv'))
                pts = df.pivot_table(index='rollout', columns='action', values='points')
                pos = df.pivot_table(index='rollout', columns='action', values='final_position')
                rec = meta['recommended_action']; best = int(pts.mean().idxmax())
                dp = pts[real] - pts[rec]; dq = pos[real] - pos[rec]; se = dp.std() / np.sqrt(len(dp))
                db = pts[best] - pts[real]; seb = db.std() / np.sqrt(len(db))
                st = meta['agent_state_at_lap']
                rows.append(dict(gp=key, algo=algo, scen=scen, lap=n, pos=st['position'], comp=f"C{st['compound']}",
                                 tyrelife=st['tyrelife'], real=lab[real], rec=lab[rec], best_pts=lab[best],
                                 pts_real=round(pts[real].mean(), 2), pts_rec=round(pts[rec].mean(), 2),
                                 dpts_real_vs_rec=round(dp.mean(), 2), dpts_2se=round(2 * se, 2),
                                 dpos_real_vs_rec=round(dq.mean(), 2),
                                 signif=bool(real != rec and abs(dp.mean()) > 2 * se),
                                 gain_best_vs_real=round(db.mean(), 2), gain_2se=round(2 * seb, 2),
                                 best_signif=bool(best != real and db.mean() > 2 * seb)))
L = pd.DataFrame(rows); L.to_csv(f'{D}/compare_laps_20261024.csv', index=False)
fmt = lambda x: ' '.join(f"{r.lap}:{r.rec}({r.dpts_real_vs_rec:+.1f}{'*' if r.signif else ''})" for r in x.itertuples()) or '—'
S = []
for (key, algo, scen), g in L.groupby(['gp', 'algo', 'scen'], sort=False):
    pit = g[g.real != 'No pit']; dif = g[g.real != g.rec]
    S.append({'gp': key, 'algo': algo, 'scen': scen, 'N': len(g), 'real 2024': ' '.join(f'{r.lap}:{r.real}' for r in pit.itertuples()),
              'rec = real %': round(100 * (g.real == g.rec).mean()),
              'en paradas reales: rec (dif. pts real-rec)': fmt(pit),
              'vueltas con rec = parar y real = no parar (signif.)': ' '.join(f"{r.lap}:{r.rec}{'*' if r.signif else ''}" for r in dif[dif.real == 'No pit'].itertuples()) or '—',
              'pts real - rec (suma en vueltas distintas)': round(dif.dpts_real_vs_rec.sum(), 2),
              'vueltas donde otra accion supera a la real (signif.): mejor(+pts)': ' '.join(
                  f'{r.lap}:{r.best_pts}(+{r.gain_best_vs_real:.1f})' for r in g[g.best_signif].itertuples()) or '—'})
S = pd.DataFrame(S); S.to_csv(f'{D}/compare_summary_20261024.csv', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 120)
print(S.to_string(index=False))
