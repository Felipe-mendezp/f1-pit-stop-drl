import pandas as pd, numpy as np, sys
B='results/world2024_v8_20261019'; T=sys.argv[1]
GPS=['Bahrain_ALO','Saudi_Arabian_NOR','Miami_OCO','Hungarian_TSU','Belgian_GAS','Dutch_STR','Singapore_HUL','United_States_MAG']
rng=np.random.default_rng(0)
for fig,sub in ((6,'fig6_2024_realdnf'),(7,'fig7_2024')):
    print(f'\n=== Fig {fig}: posicion final media (base v8 -> TyreLife+1), diferencia pareada [IC95 bootstrap] ; % carreras con pit_laps distinto')
    rows=[]
    for g in GPS:
        try: x=pd.read_csv(f'{T}/{sub}/{g}.csv')
        except FileNotFoundError: continue
        b=pd.read_csv(f'{B}/{sub}/{g}.csv')
        for a in ('PPO','RecurrentPPO','QMIP'):
            xa=x[x.algo==a].reset_index(drop=True); ba=b[b.algo==a].reset_index(drop=True)
            d=(xa.final_position-ba.final_position).values
            bs=[rng.choice(d,len(d)).mean() for _ in range(2000)]
            ch=(xa.pit_laps.astype(str)!=ba.pit_laps.astype(str)).mean()
            dsq=((~xa.changed_compound.astype(bool)).mean(),(~ba.changed_compound.astype(bool)).mean())
            rows.append((g,a,ba.final_position.mean(),xa.final_position.mean(),d.mean(),*np.percentile(bs,[2.5,97.5]),ch,*dsq))
    r=pd.DataFrame(rows,columns=['gp','algo','base','tl1','diff','lo','hi','pit_change','dsq_tl1','dsq_base'])
    pd.set_option('display.width',200); print(r.round(3).to_string(index=False))
    m=r.pivot(index='gp',columns='algo',values=['base','tl1']).mean()
    print('media sobre GP:'); print(m.round(3).to_string())
    for a in ('PPO','RecurrentPPO'):
        p=r.pivot(index='gp',columns='algo',values='tl1'); q=r.pivot(index='gp',columns='algo',values='base')
        print(f'{a} - QMIP: base {(q[a]-q.QMIP).mean():+.3f}  tl1 {(p[a]-p.QMIP).mean():+.3f}')
    p=r.pivot(index='gp',columns='algo',values='tl1'); q=r.pivot(index='gp',columns='algo',values='base')
    print(f'PPO - RPPO: base {(q.PPO-q.RecurrentPPO).mean():+.3f}  tl1 {(p.PPO-p.RecurrentPPO).mean():+.3f}')
