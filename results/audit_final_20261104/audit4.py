import pandas as pd, numpy as np, glob, os, json, re, zipfile, io
SIM='trained_models/simulation_v10eGQ_20261028/'
GPS=['Bahrain','Saudi Arabian','Miami','Hungarian','Belgian','Dutch','Singapore','United States']
print('== coef tire'); c=pd.read_csv('results/sim_v10eGQ_20261028/tyre_v10/coefs_opcion3h.csv'); print(list(c.columns)); print(open('results/sim_v10eGQ_20261028/tyre_v10/tau.txt').read()[:300])
gpcol=[k for k in c.columns if k.lower() in('gp','grand_prix')][0]
print(c[c[gpcol].astype(str).str.contains('Hungar|Saudi')].round(4).to_string()[:1500])
print('== overtake'); print(pd.read_csv('results/sim_v10eGQ_20261028/overtake_v10f/params.csv').round(4).to_string()[:1400]); print(pd.read_csv('results/sim_v10eGQ_20261028/overtake_v10f/sample_counts.csv').to_string()[:900])
print('== yf'); y=pd.read_csv('results/sim_v10eGQ_20261028/yf_model_v10_20261027.csv'); print(y[y.iloc[:,0].astype(str).str.contains('Bahrain|Saudi|Dutch|United')].round(3).to_string()[:900]); print(open('results/sim_v10eGQ_20261028/yf_model_v10_20261027_fit.txt').read()[:900])
print('== alpha'); a=json.load(open('data/rival_start_alpha_2021_2023_20261018.json')); print({k:(v if k!='gps' else len(v)) for k,v in a.items() if k not in('counts',)})
print('== config'); 
for g in GPS:
    j=json.load(open(SIM+g+' Grand Prix/config.json')); print(g,j.get('sc_prob'),j.get('vsc_prob'),j.get('n_laps'),j.get('compounds') or j.get('available_compounds'), {k:v for k,v in j.get('initial_positions_2024',{}).items() if k in('Driver_ALO','Driver_GAS','Driver_STR','Driver_TSU','Driver_OCO','Driver_NOR','Driver_HUL','Driver_MAG')})
s=pd.read_csv('data/session_2024_V2.csv',low_memory=False)
AG={'Bahrain':'ALO','Saudi Arabian':'NOR','Miami':'OCO','Hungarian':'TSU','Belgian':'GAS','Dutch':'STR','Singapore':'HUL','United States':'MAG'}
print('== real 2024: laps, DNF, SC/VSC laps, grid')
for g,dr in AG.items():
    x=s[s.GP==g+' Grand Prix']; nl=x.LapNumber.max(); last=x.groupby('Driver').LapNumber.max(); dnf=last[last<nl-3]
    st=x.groupby('LapNumber').TrackStatus.first().astype(str)
    sc=[int(l) for l,v in st.items() if '4' in v]; vsc=[int(l) for l,v in st.items() if '6' in v or '7' in v]
    l1=x[x.LapNumber==1].set_index('Driver').Position
    print(g,'N',nl,'dnf',dnf.to_dict(),'SC',sc,'VSC',vsc,'posL1 ag',l1.get(dr))
yf=json.load(open('data/yf_laps_2024_20261007.json')); print('yf json keys',str(list(yf.items())[:3])[:300])
print('== SC laps Hung/Belg 2021-23')
for y in (2021,2022,2023):
    d=pd.read_csv(f'data/session_{y}_V2.csv',low_memory=False)
    for g in ('Hungarian Grand Prix','Belgian Grand Prix','Dutch Grand Prix'):
        x=d[d.GP==g]
        if len(x)==0: continue
        st=x.groupby('LapNumber').TrackStatus.agg(lambda v:'|'.join(sorted(set(map(str,v)))))
        print(y,g,'laps with SC',[int(l) for l,v in st.items() if '4' in v][:12],'N',x.LapNumber.max())
print('== sets')
t=pd.read_csv('data/tyre_sets_2024_reg_v10eGQ_20261028.csv'); o=pd.read_csv('data/tyre_sets_2024_reg_20261021.csv')
for n,tt in (('v10eGQ',t),('orig',o)):
    print(n,tt.GP.nunique(),'mean per driver',round(tt.groupby(['GP','Driver']).size().mean(),3),'new share',round((tt.Age==0).mean(),3))
    e=tt[tt.GP.str.contains('|'.join(GPS))]; print('  8GP',round(e.groupby(['GP','Driver']).size().mean(),3),round((e.Age==0).mean(),3))
r=pd.read_csv('data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv'); print('stints',len(r),'used',(r.Age>0).sum(),round((r.Age>0).mean(),3),'starts',round((r[r.PitLap==0].Age>0).mean(),3),'after',round((r[r.PitLap>0].Age>0).mean(),3), 'used>=? ',(r.Age>=3).sum())
print('== selection'); 
ss=pd.read_csv('results/seed_selection_v5_20261010/v10f_v10g/seed_summary_DQN_A2C_TRPO_PPO_RecurrentPPO_r578_dsq0.csv'); print(list(ss.columns))
print(ss[ss.get('selected',False)==True].round(3).to_string()[:3500] if 'selected' in ss else ss.head(3).to_string())
