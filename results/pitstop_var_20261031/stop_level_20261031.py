import pandas as pd, numpy as np
S="results/pitstop_var_20261031/"
r=pd.read_csv(S+"pit_loss.csv")
# emparejar in-lap y out-lap de la misma parada: out-lap = in-lap + 1
piv=r.pivot_table(index=['Year','GP','Driver'],columns='kind',values='loss',aggfunc=list)
# mas simple: sumar por parada usando el orden
tot=[]
for (y,gp,drv),g in r.groupby(['Year','GP','Driver']):
    ins=g[g.kind=='in'].loss.tolist(); outs=g[g.kind=='out'].loss.tolist()
    for a,b in zip(ins,outs): tot.append((gp,a+b))
t=pd.DataFrame(tot,columns=['GP','loss'])
med=t.groupby('GP').loss.transform('median')
res=(t.loss-med).dropna()
print(f"paradas emparejadas n={len(res)}")
print(f"  perdida total mediana por GP: {t.groupby('GP').loss.median().min():.1f} a {t.groupby('GP').loss.median().max():.1f} s")
q=res.quantile([.05,.25,.5,.75,.9,.95,.99]).round(2)
print("  residuo dentro de GP:",{f"p{int(a*100)}":v for a,v in q.items()})
iqr=res.quantile(.75)-res.quantile(.25)
print(f"  NUCLEO sd robusta (IQR/1.349) = {iqr/1.349:.2f} s ; sd cruda = {res.std():.2f} s")
for x in (2,3,5,10,20):
    print(f"    P(exceso > {x}s) = {(res>x).mean()*100:5.2f}%")
print(f"  media del exceso condicional a >5s: {res[res>5].mean():.1f} s")
