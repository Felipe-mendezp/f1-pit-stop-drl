# Efecto analitico de la convencion TyreLife (sim: vuelta k de la tanda usa k-1; datos: k, y >k si el neumatico es usado)
import os, sys, itertools
ROOT=os.getcwd(); os.environ['F1_SIMULATION_DIR']=os.path.join(ROOT,'trained_models','simulation_v7_20261015')
sys.path[:0]=[ROOT,os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
from model_loader import ModelLoader
from evaluate_qmip_vs_drl import build_qmip_params
L=ModelLoader()
GPS=['Bahrain','Saudi Arabian','Miami','Hungarian','Belgian','Dutch','Singapore','United States']
D=pd.read_csv('data/session_2024_V2.csv')
def stint(b,n,off): return b[0]*n+b[1]*(n*(n-1)/2+off*n)   # vuelta k: b0+b1*(k-1+off)
def best(B,ts,N,off):
    out=(1e18,None)
    for ns in (1,2):
        for comps in itertools.product(['Hard','Medium','Soft'],repeat=ns+1):
            if len(set(comps))<2: continue
            for cuts in itertools.combinations(range(1,N),ns):
                ls=[b-a for a,b in zip((0,)+cuts,cuts+(N,))]
                t=sum(stint(B[c],l,off) for c,l in zip(comps,ls))+ns*ts
                if t<out[0]: out=(t,tuple(zip(comps,ls)))
    return out
cost=lambda B,ts,s,off: sum(stint(B[c],l,off) for c,l in s)+(len(s)-1)*ts
f=lambda s:' '.join(f"{c[0]}{l}" for c,l in s)
print(f"{'GP':14s} {'b1 H':>6s} {'b1 M':>6s} {'b1 S':>6s} | {'opt TL0 (sim)':16s} {'opt TL1 (datos)':16s} {'perdida':>7s} | {'sd desplaz.':>10s} {'gap adyac.':>10s} | {'usados: err medio/max (s)':>26s}")
for g in GPS:
    gp=g+' Grand Prix'; q=build_qmip_params(L,gp); B=q['betas_dict']; ts=q['time_stop']
    c2q={f'C{k}':v for k,v in q['compound_to_qmip'].items()}
    d=D[D.GP==gp].sort_values(['Driver','LapNumber']); N=int(d.LapNumber.max())
    t0,s0=best(B,ts,N,0); t1,s1=best(B,ts,N,1)
    loss=cost(B,ts,s0,1)-t1   # cuanto pierde la estrategia optima del sim si la verdad es TL1
    # estrategias reales 2024 de los que terminaron: desplazamiento total sum b1_C*laps_C (TL0->TL1) y error por neumaticos usados
    comp_col='Compound_Detail' if 'Compound_Detail' in d.columns else None
    sh=[];err=[]
    fin=d.groupby('Driver').LapNumber.max(); fin=fin[fin==N].index
    for drv,x in d[d.Driver.isin(fin)].groupby('Driver'):
        x=x.copy(); x['st']=((x.PitOut==1)|(x.LapNumber==1)).cumsum()
        s=0;e=0
        for _,y in x.groupby('st'):
            cd=y['Compound_Detail'].iloc[0] if comp_col else None
            q_=c2q.get(cd) if cd is not None else {'HARD':'Hard','MEDIUM':'Medium','SOFT':'Soft'}.get(y.Compound.iloc[0])
            if q_ is None: continue
            b1=B[q_][1]; n=len(y); s+=b1*n
            e+=b1*n*max(y.TyreLife.iloc[0]-1,0)      # TyreLife inicial extra (usado) no visto por el sim
        sh.append(s); err.append(e)
    # gap tipico entre posiciones adyacentes: tiempo total real
    tt=d[d.Driver.isin(fin)].groupby('Driver').LapTime.sum() if 'LapTime' in d.columns else None
    gap=np.median(np.diff(np.sort(tt.values))) if tt is not None else np.nan
    err=np.array(err)
    print(f"{g:14s} {B['Hard'][1]:6.3f} {B['Medium'][1]:6.3f} {B['Soft'][1]:6.3f} | {f(s0):16s} {f(s1):16s} {loss:7.3f} | {np.std(sh):10.3f} {gap:10.2f} | {err[err>0].mean() if (err>0).any() else 0:8.2f} / {err.max():6.2f} (n={int((err>0).sum())})")
