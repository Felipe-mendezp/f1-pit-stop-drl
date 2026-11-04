import pandas as pd, numpy as np, glob, os
S="results/pitstop_var_20261031/"
# 1) distribucion empirica del exceso de parada (2021-2024, dentro de GP)
exc=pd.read_csv(S+"pit_loss.csv")
tot=[]
for (y,gp,drv),g in exc.groupby(['Year','GP','Driver']):
    ins=g[g.kind=='in'].loss.tolist(); outs=g[g.kind=='out'].loss.tolist()
    for a,b in zip(ins,outs): tot.append((gp,a+b))
t=pd.DataFrame(tot,columns=['GP','loss']); t['exc']=t.loss-t.groupby('GP').loss.transform('median')
E=t.exc.dropna().to_numpy()

# 2) gaps 2024 por GP (segundos por posicion, empirico)
d24=pd.read_csv("data/session_2024_V2.csv")
d24=d24[(d24.PitIn==0)&(d24.PitOut==0)&(d24.TrackStatus.astype(str)=='1')]
rng=np.random.default_rng(0); N=20000
print(f"{'GP':16s} {'sd_pos actual':>13s} {'sd_pos +pit':>12s} {'Δmedia':>7s} {'P(pierde>=2)':>12s}")
res=[]
for f in sorted(glob.glob("results/world2024_v10g_20261031/fig7_2024/*.csv")):
    gp=os.path.basename(f)[:-4]; key=gp.rsplit('_',1)[0].replace('_',' ')
    g=d24[d24.GP.str.startswith(key)]
    last=g.LapNumber.max(); late=g[g.LapNumber>=last-10]
    iv=late.Interval_front.replace(100.0,np.nan).dropna(); iv=iv[(iv>0)&(iv<60)].to_numpy()
    if len(iv)<30: continue
    df=pd.read_csv(f); pos=df[df.algo=='PPO'].final_position.to_numpy(float)
    # cuantas posiciones pierde un exceso x: #autos cuyo gap acumulado < x
    draw=rng.choice(E,N)
    cum=np.sort(iv)  # muestra de gaps a autos sucesivos
    lost=np.array([np.searchsorted(np.cumsum(rng.choice(iv,8)),max(x,0)) for x in draw])
    newpos=rng.choice(pos,N)+lost
    res.append((gp.replace('_',' '),pos.std(),newpos.std(),lost.mean(),(lost>=2).mean()*100))
for a,b,c,dd,e in res: print(f"{a:16s} {b:13.2f} {c:12.2f} {dd:7.2f} {e:11.1f}%")
m=np.array([[b,c,dd,e] for _,b,c,dd,e in res]).mean(0)
print(f"{'MEDIA':16s} {m[0]:13.2f} {m[1]:12.2f} {m[2]:7.2f} {m[3]:11.1f}%")
