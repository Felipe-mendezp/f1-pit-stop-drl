"""2026-10-27 — Paradas reales bajo SC por vuelta del periodo, 2021-2024, carreras secas, sin bandera roja. Parada = cambio de
juego (baja TyreLife o cambia Compound_Detail), como real_plan; vuelta de SC = la mayoria de los pilotos tiene TrackStatus
con 4. La bandera PitIn de session_*_V2.csv no sirve aqui (bajo SC cuenta de mas, p. ej. Qatar 2024: 45 en 3 vueltas).
    ~/.venvs/f1-drl-v2/bin/python results/sc_second_lap_check_20261027/sc_stop_lap_share_20261027.py"""
import pandas as pd, numpy as np
rows=[]
for y in (2021,2022,2023,2024):
    s=pd.read_csv(f'data/session_{y}_V2.csv',usecols=['LapNumber','Driver','GP','TrackStatus','Compound','Compound_Detail','TyreLife'])
    s['ts']=s.TrackStatus.astype(str)
    for gp,g in s.groupby('GP'):
        if g.Compound.isin(['INTERMEDIATE','WET']).any(): continue
        st=g.groupby('LapNumber').ts
        sc=st.agg(lambda x: x.str.contains('4').mean()>0.5); red=st.agg(lambda x: x.str.contains('5').any())
        idx=0; m={}
        for L in sorted(sc.index):
            idx=idx+1 if sc[L] else 0; m[L]=idx
        for drv,h in g.sort_values('LapNumber').groupby('Driver'):
            h=h.reset_index(drop=True)
            for i in range(1,len(h)):
                p,r=h.iloc[i-1],h.iloc[i]
                if r.LapNumber!=p.LapNumber+1: continue
                if r.TyreLife<p.TyreLife or r.Compound_Detail!=p.Compound_Detail:
                    L=int(p.LapNumber)
                    if m.get(L,0)>0 and not red.get(L,False) and L>1: rows.append((y,gp,drv,L,m[L]))
d=pd.DataFrame(rows,columns=['y','gp','drv','lap','idx'])
print('stops under SC (dry, no red):',len(d))
print(d.idx.clip(upper=3).value_counts().sort_index().to_string()); print(d.idx.clip(upper=3).value_counts(normalize=True).sort_index().round(3).to_string())
print(d.groupby(['y','gp']).idx.apply(lambda x: x.clip(upper=3).value_counts().sort_index().to_dict()).to_string())
