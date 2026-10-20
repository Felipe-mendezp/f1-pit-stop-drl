import os, sys, itertools, json
ROOT=os.getcwd(); os.environ['F1_SIMULATION_DIR']=os.path.join(ROOT,'trained_models',sys.argv[1])
sys.path[:0]=[ROOT,os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
import config, compat; compat.setup_all()
from model_loader import ModelLoader
from evaluate_qmip_vs_drl import build_qmip_params
from plot_fig5_stints_one_driver import LAPS_DD
L=ModelLoader()
GPS=['Bahrain','Saudi Arabian','Miami','Hungarian','Belgian','Dutch','Singapore','United States']
def stint(b,n): return b[0]*n+b[1]*n*(n+1)/2   # vuelta k de la tanda: b0 + b1*k
def best(q,N,allowed):
    B=q['betas_dict']; ts=q['time_stop']; out=(1e18,None)
    for ns in (1,2,3):
        for comps in itertools.product(allowed,repeat=ns+1):
            if len(set(comps))<2: continue
            # largos optimos por fuerza bruta (2-3 tandas)
            for cuts in itertools.combinations(range(1,N),ns):
                ls=[b-a for a,b in zip((0,)+cuts,cuts+(N,))]
                t=sum(stint(B[c],l) for c,l in zip(comps,ls))+ns*ts
                if t<out[0]: out=(t,list(zip(comps,ls)))
    return out
rec=[]
print(f"{'GP':14s} {'b0 H-M':>7s} {'b0 M-S':>7s} {'b1 H':>6s} {'b1 M':>6s} {'b1 S':>6s} {'pit':>5s}  opt               mejor con H        costo H")
for g in GPS:
    gp=g+' Grand Prix'; q=build_qmip_params(L,gp); B=q['betas_dict']; N=LAPS_DD[gp]
    t0,s0=best(q,N,['Hard','Medium','Soft'])
    # mejor estrategia que usa el duro al menos una vez
    BH=dict(B); 
    tH,sH=(1e18,None)
    for ns in (1,2,3):
        for comps in itertools.product(['Hard','Medium','Soft'],repeat=ns+1):
            if len(set(comps))<2 or 'Hard' not in comps: continue
            for cuts in itertools.combinations(range(1,N),ns):
                ls=[b-a for a,b in zip((0,)+cuts,cuts+(N,))]
                t=sum(stint(B[c],l) for c,l in zip(comps,ls))+ns*q['time_stop']
                if t<tH: tH,sH=t,list(zip(comps,ls))
    f=lambda s:' '.join(f"{c[0]}{l}" for c,l in s)
    rec.append(dict(gp=g,db0=B['Hard'][0]-B['Medium'][0],db1=B['Medium'][1]-B['Hard'][1],b1M=B['Medium'][1],opt=f(s0),costH=tH-t0))
    print(f"{g:14s} {B['Hard'][0]-B['Medium'][0]:7.3f} {B['Medium'][0]-B['Soft'][0]:7.3f} {B['Hard'][1]:6.3f} {B['Medium'][1]:6.3f} {B['Soft'][1]:6.3f} {q['time_stop']:5.1f}  {f(s0):17s} {f(sH):18s} {tH-t0:6.2f}")

json.dump(rec,open(sys.argv[2],'w'))
