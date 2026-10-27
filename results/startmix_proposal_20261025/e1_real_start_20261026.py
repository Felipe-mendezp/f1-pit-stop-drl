"""2026-10-26 — E1 restringido a la largada real 2024 (decision del autor, 26-oct): el analisis B (historia real) solo
pone al agente en la largada real del piloto, asi que el criterio E1 (DSQ <= 1 %) se aplica solo a la fila de la sonda
con ese compuesto. Lee e1_20261026/e1_all_20261026.csv, la largada real de data/session_2024_V2.csv (misma regla que
scripts/ds_rollouts_realhist_20261024.py: primer Compound_Detail del piloto) y la seleccion E2
(e2_selection_20261026/seed_summary_r578_dsq0.csv). Escribe e1_real_start_20261026.csv (por replica) y
e2_real_start_20261026.csv (eleccion: regla del paper + E1 en la largada real)."""
import os, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(H))
s = pd.read_csv(os.path.join(ROOT, 'data', 'session_2024_V2.csv'))
start = {}   # GP y piloto desde el nombre de carpeta (GP_CONFIGS sin el envoltorio v5hd tiene otros pilotos en 4 GP)
for md in ['Bahrain_ALO', 'Saudi_Arabian_NOR', 'Miami_OCO', 'Hungarian_TSU', 'Belgian_GAS', 'Dutch_STR', 'Singapore_HUL',
           'United_States_MAG']:
    gp, drv = md.rsplit('_', 1)[0].replace('_', ' ') + ' Grand Prix', md.rsplit('_', 1)[1]
    d = s[(s.GP == gp) & (s.Driver == drv)].sort_values('LapNumber')
    assert len(d), md
    start[md] = 'C' + str(d.Compound_Detail.iloc[0]).lstrip('C')
e1 = pd.read_csv(os.path.join(H, 'e1_20261026', 'e1_all_20261026.csv'))
own = e1[e1.modo == 'propia'][['gp', 'algo', 'seed', 'largadas']]
r = e1[e1.modo == e1.gp.map(start)].rename(columns={'modo': 'largada_real'})
r = r.merge(own, on=['gp', 'algo', 'seed'], suffixes=('', '_propia'))
r['misma_largada'] = [p.startswith(l + ':') for p, l in zip(r.largadas_propia, r.largada_real)]
r['pasa_e1_real'] = r.dsq_pct <= 1
r = r[['gp', 'algo', 'seed', 'largada_real', 'largadas_propia', 'misma_largada', 'dsq_pct', 'pos', 'pasa_e1_real']]
r.to_csv(os.path.join(H, 'e1_real_start_20261026.csv'), index=False)
sel = pd.read_csv(os.path.join(H, 'e2_selection_20261026', 'seed_summary_r578_dsq0.csv'))
sel = sel.merge(r.rename(columns={'gp': 'model_dir'}), on=['model_dir', 'algo', 'seed'])
out = []
for (g, a), d in sel.groupby(['model_dir', 'algo'], sort=False):
    ok = d[d.pasa & d.pasa_e1_real].sort_values('pos_media')
    regla = d[d.elegida].iloc[0]
    out.append(dict(model_dir=g, algo=a, largada_real=d.largada_real.iloc[0], seed_regla=int(regla.seed),
                    dsq_real_regla=regla.dsq_pct, seed_final=int(ok.seed.iloc[0]) if len(ok) else None,
                    pos_val_final=ok.pos_media.iloc[0] if len(ok) else None, n_ok=len(ok)))
o = pd.DataFrame(out); o.to_csv(os.path.join(H, 'e2_real_start_20261026.csv'), index=False)
pd.set_option('display.width', 200)
print(r.to_string(index=False)); print(); print(o.to_string(index=False))
print(f"\nreplicas que pasan: {r.pasa_e1_real.sum()}/{len(r)}; agentes con eleccion: {o.seed_final.notna().sum()}/{len(o)}")
