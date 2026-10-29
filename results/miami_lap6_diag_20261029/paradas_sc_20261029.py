import sys, os
sys.argv=[sys.argv[0],'--time-only','--out','/dev/null','--n','200']
src=open('results/miami_lap6_diag_20261029/fixed_sweep_20261029.py').read().split("rows = []")[0]
exec(compile(src,'fs','exec'))
import pandas as pd
a=pd.read_csv('results/miami_lap6_diag_20261029/data/agents.csv'); a=a[~a.strat.str.contains('-')]
for s in a.strat.unique():
    x=a[a.strat==s]; n=sc=0; nsc_races=0; pos_sc=[];pos_no=[]
    for _,r in x.iterrows():
        lv,ls=scen[int(r.sim)]; neutral=set(lv)|set(ls)
        laps=[int(p.split(':')[0]) for p in str(r.pits).split() if int(p.split(':')[0])>=1]
        n+=len(laps); sc+=sum(l in neutral or l-1 in neutral or l+1 in neutral for l in laps)
        (pos_sc if any(l>=7 for l in neutral) else pos_no).append(r.pos)
    print(s, f'paradas {n}, en/junto a SC-VSC {sc/n:.0%} | pos con SC/VSC tras v6 {np.mean(pos_sc):.2f} (n={len(pos_sc)}) sin {np.mean(pos_no):.2f} (n={len(pos_no)})')
d=pd.concat([pd.read_csv(f) for f in __import__('glob').glob('results/miami_lap6_diag_20261029/data/part*.csv')])
for st in ['3-33-4','4-6-3']:
    x=d[d.strat==st]; m=[any(l>=7 for l in set(scen[i][0])|set(scen[i][1])) for i in x.sim]
    print(st, 'con SC/VSC', x.pos[m].mean().round(2), 'sin', x.pos[[not v for v in m]].mean().round(2))
