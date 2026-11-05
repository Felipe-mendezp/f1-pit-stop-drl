import sys,os; ROOT=os.getcwd(); sys.path[:0]=[ROOT,'src',os.path.join(ROOT,'results','sim_v10eGQ_20261028'),os.path.join(ROOT,'results','rival_model_eval_20261014')]
os.environ['F1_TYRELIFE_2021']='v10eGQ'
import pandas as pd, numpy as np, json
import config, compat; compat.setup_all()
import common_v10_20261027 as C
# 70%
d=pd.read_csv('results/decision_support_v10gw24belief_20261103/PPO/saudi_arabian_nor_real_lap7.csv'); x=d[d.action==0]
pl=x.pit_laps.fillna('').astype(str).str.split(';').apply(lambda v:[int(float(i)) for i in v if i])
n=pl.apply(len); first=pl.apply(min); print('stay out: n stops dist',n.value_counts().to_dict(),'first>25',(first>25).mean(),' two+ stops & first<=25',((n>1)&(first<=25)).sum(),'upper bound share',((first>25)|((n>1)&(first<=25))).mean())
# D18
a=pd.read_csv('trained_models/simulation_v10eGQ_20261028/driver_coef_audit.csv'); print(list(a.columns)); print(a.head(3).to_string()[:400])
p=pd.read_csv('results/driver_coef_v10_20261028/v10eGQ/eq2_predictions.csv'); print(list(p.columns)[:12]); print(p[(p.Year==2024)&(p.GP=='Belgian Grand Prix')].head(3).to_string()[:500])
from pathlib import Path
import pickle
def load(f):
    with open(f,'rb') as h: return pickle.load(h)
m=load('trained_models/simulation_v10eGQ_20261028/Belgian Grand Prix/laptime_clear.pkl'); print(type(m), [k for k in dir(m) if not k.startswith('_')][:14])
# Stints
s=pd.read_csv('data/session_2024_V2.csv',low_memory=False).sort_values(['GP','Driver','LapNumber'])
F=C.pass_flags((2024,))
print(F.columns.tolist(), len(s))
rawpath=None
# wet edition presence
from models import rival_models as RM
raw=C.rival_laps(RM); print('rival laps',len(raw),'years',sorted(raw.Year.unique()))
for y,g in [(2021,'Emilia Romagna Grand Prix'),(2022,'Emilia Romagna Grand Prix'),(2023,'Dutch Grand Prix'),(2022,'Singapore Grand Prix'),(2022,'Monaco Grand Prix'),(2023,'Monaco Grand Prix'),(2022,'Japanese Grand Prix'),(2021,'Belgian Grand Prix')]:
    print(' rival',y,g,len(raw[(raw.Year==y)&(raw.GP==g)]))
y=pd.read_csv('results/sim_v10eGQ_20261028/yf_model_v10_20261027.csv'); print(y[y.n_sc_laps>0].GP.tolist()[:6], 'GP count',len(y))
