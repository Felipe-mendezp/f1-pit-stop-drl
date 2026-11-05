import sys,os; ROOT=os.getcwd(); sys.path[:0]=[ROOT,'src',os.path.join(ROOT,'results','sim_v10eGQ_20261028'),os.path.join(ROOT,'scripts')]
os.environ['F1_TYRELIFE_2021']='v10eGQ'
import pickle, numpy as np, pandas as pd
import config, compat; compat.setup_all()
m=pickle.load(open('trained_models/simulation_v10eGQ_20261028/Belgian Grand Prix/laptime_clear.pkl','rb'))
print(m.feature_names_[:6], m.categorical_maps_ and list(m.categorical_maps_)[:6], len(m.coef_)); 
n=[f for f in m.feature_names_ if 'ALB' in f or 'NOR' in f or 'GAS' in f]; print(n, [round(float(m.coef_[m.feature_names_.index(f)]),3) for f in n])
a=pd.read_csv('trained_models/simulation_v10eGQ_20261028/driver_coef_audit.csv'); print(a[(a.GP=='Belgian Grand Prix')&a.driver.isin(['ALB','NOR','GAS'])].to_string())
