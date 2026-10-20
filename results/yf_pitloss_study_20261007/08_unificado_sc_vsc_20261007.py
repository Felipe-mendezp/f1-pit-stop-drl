from yf_common_20261007 import *
from scipy.stats import mannwhitneyu
Y=pd.read_pickle(OUT+'strict.pkl'); G=pd.read_pickle(OUT+'green_comp.pkl')
Y['grp']=np.where(Y.pure=='VSC','VSC',np.where(Y.ev_idx==2,'SC_temprano','SC_tardio'))
for c in ['in','out']:
    a=Y[(Y.comp==c)&(Y.grp=='SC_temprano')].r; b=Y[(Y.comp==c)&(Y.grp=='VSC')].r
    print(c,'SC temprano n',len(a),'med',round(a.median(),2),' VSC n',len(b),'med',round(b.median(),2),' Mann-Whitney p',round(mannwhitneyu(a,b).pvalue,3))
U=Y[Y.grp!='SC_tardio']
rng=np.random.default_rng(1); races=U.race.unique(); B=[]
for _ in range(2000):
    s=rng.choice(races,len(races)); bb=pd.concat([U[U.race==r] for r in s]); B.append(bb.groupby('comp')['r'].median())
B=pd.DataFrame(B)
k=U.groupby('comp')['r'].median()
print('\nModelo unificado (SC temprano + VSC): kappa',k.round(2).to_dict(),' n',U.groupby('comp').size().to_dict(),' carreras',U.groupby('comp').race.nunique().to_dict())
print('IC90:',B.quantile([.05,.95]).round(2).to_dict())
x=pd.read_excel(D+'yf_regression_2024.xlsx',sheet_name='Sheet2',index_col=0)
rows=[]
for gp in PAPER:
    gi,go=G.loc[gp,'in'],G.loc[gp,'out']
    rows.append(dict(GP=gp.replace(' Grand Prix',''),verde=gi+go,sim=x.loc[gp,'yf_PitIn']+x.loc[gp,'yf_PitOut'],unif_in=k['in']*gi,unif_out=k['out']*go,unif=k['in']*gi+k['out']*go,
                     lo=B['in'].quantile(.05)*gi+B['out'].quantile(.05)*go, hi=B['in'].quantile(.95)*gi+B['out'].quantile(.95)*go))
T=pd.DataFrame(rows).set_index('GP'); T.loc['Media']=T.mean(); print(T.round(1).to_string())
print('ratio unificado/verde',round((T.unif/T.verde).iloc[:-1].mean(),2),' sim/verde',round((T.sim/T.verde).iloc[:-1].mean(),2))
