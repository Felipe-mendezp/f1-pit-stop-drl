import pandas as pd, numpy as np, re, glob, os, json, zipfile
D='results/driver_coef_v10_20261028/v10eGQ/'
print(pd.read_csv(D+'eq2_mse_summary_20261014.csv').round(3).to_string()[:900])
print(pd.read_csv(D+'eq2_fallback_mse_20261014.csv').round(3).to_string()[:1100])
print(open(D+'fig4_tikz_coords_20261014.txt').read()[:900])
print(pd.read_csv(D+'fig4_mse_by_gp_20261014.csv').round(3).to_string()[:1100])
print(open('results/driver_coef_v10_20261028/belgian_table_20261028.txt').read()[:1000])
print(os.listdir('results/sim_v10eGQ_20261028/outliers_us_v10'))
for f in glob.glob('results/sim_v10eGQ_20261028/outliers_us_v10/*.csv')[:1]:
    print(pd.read_csv(f).head(10).round(4).to_string()[:900])
print(open('results/sim_v10eGQ_20261028/laptime_v10.log').read()[-500:])
# Opt gap
print(os.listdir('results/onedriver_v10eGQ_20261028/tableN1'))
for f in glob.glob('results/onedriver_v10eGQ_20261028/tableN1/*.csv')[:2]:
    print(f); print(pd.read_csv(f).round(3).to_string()[:1000])
# selection
S=pd.read_csv('results/seed_selection_v5_20261010/v10f_v10g/seed_summary_DQN_A2C_TRPO_PPO_RecurrentPPO_r578_dsq0.csv')
E=S[S.elegida==True]; print(E[['model_dir','algo','seed','pos_media','dsq','share_ok','pasa','nota']].sort_values(['algo','model_dir']).round(3).to_string())
print('not pasa:',E[~E.pasa.astype(bool)][['model_dir','algo','seed','dsq','share_ok']].to_string()); print('selected per seed files vs',S.seed.max())
# Yf
y=json.load(open('data/yf_laps_2024_20261007.json')); print({k:v for k,v in y.items() if 'Miami' in k or 'United' in k}) 
print(str(y.get('_doc'))[:400])
c=pd.read_csv('results/sim_v10eGQ_20261028/yf_model_v10_20261027.csv'); print(c[c.GP.str.contains('Hungar|Belgian')][['GP','n_sc_laps','u_SC']].to_string())
# used stints original
for f in ('data/tyre_race_stint_ages_2024_20261021.csv','data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv'):
    r=pd.read_csv(f); print(f,len(r),r.GP.nunique()); 
    for th in (1,3): print(' th',th,'used',(r.Age>=th).sum(),round((r.Age>=th).mean(),3),'starts',round((r[r.PitLap==0].Age>=th).mean(),3),'after',round((r[r.PitLap>0].Age>=th).mean(),3))
# hyperparams
for base,lbl in (('trained_models/retrained_v10g_20261030/selected/Saudi_Arabian_NOR','multi'),('results/onedriver_v10eGQ_20261028/selected','mono')):
    if lbl=='mono': continue
    for a in ['DQN','A2C','TRPO','PPO','RecurrentPPO']:
        z=zipfile.ZipFile(f'{base}/{a}/best_model.zip'); dd=json.loads(z.read('data'))
        pk=lambda k:(dd.get(k) if not isinstance(dd.get(k),dict) else {kk:vv for kk,vv in dd[k].items() if not kk.startswith(':')})
        print(a,{k:pk(k) for k in ['gamma','n_steps','batch_size','ent_coef','vf_coef','gae_lambda','n_epochs','target_kl','buffer_size','learning_starts','exploration_final_eps','target_update_interval'] if k in dd}, str(dd.get('learning_rate'))[:40], str(dd['policy_kwargs'])[:170], dd.get('clip_range',{}) if not isinstance(dd.get('clip_range'),dict) else '')
