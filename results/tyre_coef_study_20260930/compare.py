import pandas as pd, numpy as np
S='/private/tmp/claude-501/-Users-charlesthraves-MIT-Dropbox-Charles-Thraves-Research-F1-DLR/b36a7349-f575-48a6-9b80-ceeace424387/scratchpad/ac'
t=pd.read_csv(f'{S}/V/race2024_slopes.csv').set_index(['GP','compound'])
hc=pd.read_excel(f'{S}/A2/compounds_coefs_V3.xlsx',index_col=0); rp=pd.read_csv(f'{S}/A2/race_params.csv').set_index('GP')
hist={}; hse={}
for _,r in hc[hc.Variable.str.startswith('TyreLife')].iterrows():
    c=r.Variable.split('[')[-1][:-1]; hist[(r.GP,c)]=r.Coef; hse[(r.GP,c)]=rp.at[r.GP,'se:'+r.Variable]
simn=pd.read_excel(f'{S}/mc_ACf/completed_coefs_V2_filled.xlsx',sheet_name='variable',index_col=0)
simo=pd.read_excel('data/completed_coefs_V2_filled.xlsx',sheet_name='variable',index_col=0)
pr=pd.read_csv(f'{S}/B/practice_slopes.csv').set_index(['GP','compound'])
elig=[g for g in simn.index if isinstance(simn.loc[g,'C_2024'],str) and simn.loc[g,'C_2024_m'].replace(',','').strip()=='']
rows=[]
for gp in elig:
    comps=simn.loc[gp,'C_2024'].split(', ')
    # ancla para (b): compuesto con menor se historico entre los que tienen historia y practica
    both=[c for c in comps if (gp,c) in hist and hist[(gp,c)]>0 and (gp,c) in pr.index]
    anc=min(both,key=lambda c:hse[(gp,c)]) if both else None
    for nm,c in zip('HMS',comps):
        if (gp,c) not in t.index: continue
        r={'GP':gp.replace(' Grand Prix',''),'c':f'{nm}={c}','verdad 2024':t.loc[(gp,c),'slope_2024'],'se24':t.loc[(gp,c),'se_2024'],'n24':t.loc[(gp,c),'n_laps'],
           'hist OLS':hist.get((gp,c),np.nan),'sim actual':simo.loc[gp,c] if gp in simo.index else np.nan,'sim A+C':simn.loc[gp,c],
           'práctica':pr.loc[(gp,c),'slope_prac'] if (gp,c) in pr.index else np.nan,
           'práctica cruda':pr.loc[(gp,c),'raw_slope'] if (gp,c) in pr.index else np.nan,
           'práct. dif.+ancla':(pr.loc[(gp,c),'slope_prac']-pr.loc[(gp,anc),'slope_prac']+hist[(gp,anc)]) if anc and (gp,c) in pr.index else np.nan}
        rows.append(r)
df=pd.DataFrame(rows); df.to_csv(f'{S}/V/comparison.csv',index=False)
pd.set_option('display.width',250)
C=['hist OLS','sim actual','sim A+C','práctica','práctica cruda','práct. dif.+ancla']
print(df.round(3).to_string(index=False))
d=df[df.n24>=30]
print(f'\nCeldas con >=30 vueltas en 2024: {len(d)}; GP: {d.GP.nunique()}')
for name,sub in [('todas las celdas con dato',None),('celdas comunes (todos los candidatos)',d.dropna(subset=C))]:
    print(f'\n== {name} ==')
    for c in C:
        x=(d if sub is None else sub).dropna(subset=[c]); e=x[c]-x['verdad 2024']
        print(f'{c:18s} n={len(x):3d}  MAE={e.abs().mean():.3f}  mediana|e|={e.abs().median():.3f}  sesgo={e.mean():+.3f}  corr={np.corrcoef(x[c],x["verdad 2024"])[0,1]:.2f}')
