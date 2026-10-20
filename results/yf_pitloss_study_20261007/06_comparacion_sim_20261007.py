from yf_common_20261007 import *
Y=pd.read_pickle(OUT+'strict.pkl'); G=pd.read_pickle(OUT+'green_comp.pkl')
k=Y.groupby('cls')['r'].median()
x=pd.read_excel(D+'yf_regression_2024.xlsx',sheet_name='Sheet2',index_col=0)
a=pd.read_csv(D+'../trained_models/simulation/yf2024_coef_audit.csv').pivot(index='GP',columns='term',values='old_2021_23')
rows=[]
for gp in PAPER:
    gi,go=G.loc[gp,'in'],G.loc[gp,'out']
    rows.append(dict(GP=gp.replace(' Grand Prix',''),
      verde=gi+go, sim_in=x.loc[gp,'yf_PitIn'], sim_out=x.loc[gp,'yf_PitOut'], sim_tot=x.loc[gp,'yf_PitIn']+x.loc[gp,'yf_PitOut'],
      old2123_tot=a.loc[gp,'PitIn']+a.loc[gp,'PitOut'],
      SC_in=k['in_SC']*gi, SC_out=k['out_SC']*go, SC_tot=k['in_SC']*gi+k['out_SC']*go,
      VSC_tot=k['in_VSC']*gi+k['out_VSC']*go))
T=pd.DataFrame(rows).set_index('GP'); T.loc['Media']=T.mean()
print('kappas:',k.round(2).to_dict())
print(T.round(1).to_string())
print('\nRazón sobre la pérdida verde: sim',round((T.sim_tot/T.verde).iloc[:-1].mean(),2),' SC limpio',round((T.SC_tot/T.verde).iloc[:-1].mean(),2),' VSC limpio',round((T.VSC_tot/T.verde).iloc[:-1].mean(),2))
# bootstrap por carrera de los kappas
rng=np.random.default_rng(0); races=Y.race.unique(); B=[]
for b in range(2000):
    s=rng.choice(races,len(races)); bb=pd.concat([Y[Y.race==r] for r in s])
    B.append(bb.groupby('cls')['r'].median())
B=pd.DataFrame(B); print('\nIC 90% bootstrap por carrera de kappa:'); print(B.quantile([.05,.5,.95]).round(2))
