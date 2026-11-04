import pandas as pd, numpy as np
S="results/pitstop_var_20261031/"
fs=[f"data/session_{y}_V2.csv" for y in (2021,2022,2023,2024)]
df=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True).dropna(subset=['LapTime'])
df['TS']=df.TrackStatus.astype(str)
df=df.sort_values(['Year','GP','Driver','LapNumber'])

rec=[]
for (y,gp,drv),g in df.groupby(['Year','GP','Driver'],sort=False):
    g=g.reset_index(drop=True)
    clean=(g.PitIn==0)&(g.PitOut==0)&(g.TS=='1')
    lt=g.LapTime.to_numpy(float); last=g.LapNumber.max()
    pit_laps=set(g.LapNumber[g.PitIn==1].tolist())
    for i in range(len(g)):
        if g.PitIn.iat[i]!=1 or g.TS.iat[i]!='1': continue
        if i+1>=len(g) or g.PitOut.iat[i+1]!=1 or g.TS.iat[i+1]!='1': continue
        lo,hi=max(0,i-5),min(len(g),i+7)
        idx=[j for j in range(lo,hi) if j not in (i,i+1) and clean.iat[j]]
        if len(idx)<3: continue
        ref=np.median(lt[idx])
        loss=(lt[i]-ref)+(lt[i+1]-ref)
        ln=g.LapNumber.iat[i]
        # artefactos
        otra_parada_cerca = any(abs(p-ln)<=3 and p!=ln for p in pit_laps)
        murio_despues = (last-ln)<=3
        rec.append((gp,y,drv,ln,loss,otra_parada_cerca,murio_despues))
t=pd.DataFrame(rec,columns=['GP','Year','Driver','lap','loss','doble','fin_cerca'])
med=t.groupby('GP').loss.transform('median'); t['exc']=t.loss-med
tail=t[t.exc>5]
print(f"paradas n={len(t)} ; cola (exceso>5s) n={len(tail)} = {100*len(tail)/len(t):.1f}%")
print(f"  de la cola: {100*tail.doble.mean():.0f}% con otra parada a <=3 vueltas (reparacion/doble)")
print(f"              {100*tail.fin_cerca.mean():.0f}% a <=3 vueltas del final de su carrera (abandono)")
lim = tail[~tail.doble & ~tail.fin_cerca]
print(f"  cola 'limpia' (ni doble ni abandono): n={len(lim)} = {100*len(lim)/len(t):.1f}% de todas las paradas")
print(f"     exceso medio {lim.exc.mean():.1f} s | mediana {lim.exc.median():.1f} s | p90 {lim.exc.quantile(.9):.1f} s")
core=t[(t.exc<=5)&(~t.doble)&(~t.fin_cerca)]
print(f"  nucleo limpio: n={len(core)} sd={core.exc.std():.2f} s  IQR/1.349={(core.exc.quantile(.75)-core.exc.quantile(.25))/1.349:.2f} s")
