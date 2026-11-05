import pandas as pd, numpy as np, re, glob, pickle, os, sys, json
P='../../Paper/ANOR/F1GTR_paper_ANOR/supplementary.tex'
txt=[l for l in open(P,encoding='utf-8') if not l.lstrip().startswith('%')]
cl=lambda s:re.sub(r'\\vten\{([^}]*)\}',r'\1',s)
rows=[];gp=None
for l in txt:
    x=cl(l); m=re.match(r'\s*([A-Za-z ]*?)\s*&\s*(C\d)\s*&\s*(\d+)\s*&\s*(-?[\d.]+)\s*&\s*([\d.]+)\s*\\\\',x)
    if m: gp=m[1].strip() or gp; rows.append((gp,m[2],int(m[3]),float(m[4]),float(m[5])))
T=pd.DataFrame(rows,columns='gp c st o s'.split())
for d in ['sim_v10_20261027','sim_v10eGQ_20261028']:
    c=pd.read_csv(f'results/{d}/tyre_v10/coefs_opcion3h.csv'); c['gp']=c.GP.str.replace(' Grand Prix','')
    m=T.merge(c,left_on=['gp','c'],right_on=['gp','compound'])
    m['ref0']=m.groupby('gp').offset_vs_ref.transform('first'); m['oo']=m.offset_vs_ref-m.ref0
    print(d,'offset rel hardest maxdiff',(m.o-m.oo).abs().max().round(4),'slope',(m.s-m.slope).abs().max().round(4))
    if 'eGQ' in d: print(m[['gp','c','o','oo','s','slope']].round(4).to_string())
print('Which sim is in tree?'); 
s=pd.read_csv('trained_models/simulation_v10eGQ_20261028/sources.csv'); print(s[s.GP.str.startswith('Hung')].to_string()[:800])
r=pd.read_csv('results/sim_v10eGQ_20261028/artifacts/rival_stop_params_vs_v7.csv'); print(list(r.columns)); print(r.round(4).to_string()[:2600])
print(pd.read_csv('results/sim_v10eGQ_20261028/artifacts/rival_compound_params_vs_v7.csv').round(4).to_string()[:1100])
print(open('results/sim_v10eGQ_20261028/rivals_v10.log').read()[-700:])
