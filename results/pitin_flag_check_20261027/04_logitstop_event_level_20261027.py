import sys, warnings; warnings.filterwarnings('ignore')
sys.path[:0]=['.', 'src']
import config, compat; compat.setup_all()
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from models import rival_models as RM
L=RM.load_lap_data().sort_values(['Year','GP','Driver','LapNumber'])
k=['Year','GP','Driver']; n=L.groupby(k).shift(-1)
L['stop']=(((n.TyreLife<L.TyreLife)|(n.Compound_Detail!=L.Compound_Detail))&(n.LapNumber==L.LapNumber+1)).astype(int)
X=RM.preprocess_laps(L).sort_values(['Year','GP','Driver','LapNumber'])
# evento YF por carrera-vuelta (estado mayoritario SC o VSC), consecutivo
st=X.assign(s=np.where(X.SC==1,'SC',np.where(X.VSC==1,'VSC','G'))).groupby(['Year','GP','LapNumber']).s.agg(lambda x:x.mode().iat[0]).reset_index()
ev=[];prev=None;e=0;last='G'
for r in st.itertuples():
    if (r.Year,r.GP)!=prev: prev=(r.Year,r.GP); last='G'
    if r.s!='G' and r.s!=last: e+=1
    ev.append(e if r.s!='G' else 0); last=r.s
st['ev']=ev
X=X.merge(st[['Year','GP','LapNumber','ev']],on=['Year','GP','LapNumber'])
f="PitIn ~ SC + VSC + used_two_compounds + EarlyLaps + LateLaps + TyreLife:C(Compound_Detail) + Position + GP"
g=X[X.ev==0].assign(PitIn=X.PitIn)            # vueltas verdes: como hoy
y=X[X.ev>0]
first=y.groupby(k+['ev']).head(1).copy()       # primera vuelta del evento para cada piloto
first['PitIn']=y.groupby(k+['ev']).stop.max().values  # 1 si cambio de juego en cualquier vuelta del evento
for name,D in [('actual (por vuelta, PitIn)',X),('evento -> 1a vuelta (cambio de juego)',pd.concat([g,first]))]:
    r=smf.logit(f,D).fit(disp=False)
    fs=first[first.LapNumber>5]
    for t in ('SC','VSC'):
        z=fs[fs[t]==1]
        print(f"{name:40s} {t}={r.params[t]:.3f}({r.bse[t]:.3f}) | 1a vuelta {t}: prob predicha {r.predict(z).mean():.3f} vs real parar en el evento {z.PitIn.mean():.3f} (n={len(z)})")
