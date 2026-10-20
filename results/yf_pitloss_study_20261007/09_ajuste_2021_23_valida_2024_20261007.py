"""2026-10-07 — Modelo unificado (SC temprano + VSC) ajustado SOLO con 2021-23 y contrastado con 2024.
Componente verde por GP también con 2021-23. Usa clean_events.pkl (de 04_componentes_limpios)."""
from yf_common_20261007 import *
E=pd.read_pickle(OUT+'clean_events.pkl')
E=E[E.ref_src=='vuelta_pura'].copy(); E['race']=E.Year.astype(str)+' '+E.GP
def fit(S):
    G=S[S.pure=='G'].groupby(['GP','comp'])['exc'].median().unstack()
    Y=S[(S.pure!='G')&((S.pure=='VSC')|(S.ev_idx==2))].join(G.rename(columns={'in':'Gin','out':'Gout'}),on='GP')
    Y['Gc']=np.where(Y.comp=='in',Y.Gin,Y.Gout); Y['r']=Y.exc/Y.Gc
    return G,Y,Y.groupby('comp')['r'].median()
Gt,Yt,kt=fit(E[E.Year<=2023]); Ga,Ya,ka=fit(E)
print('kappa 2021-23:',kt.round(2).to_dict(),' n',Yt.groupby('comp').size().to_dict(),' carreras',Yt.groupby('comp').race.nunique().to_dict())
print('kappa 2021-24:',ka.round(2).to_dict())
rng=np.random.default_rng(1); races=Yt.race.unique(); B=[]
for _ in range(2000):
    s=rng.choice(races,len(races)); B.append(pd.concat([Yt[Yt.race==r] for r in s]).groupby('comp')['r'].median())
B=pd.DataFrame(B); print('IC90 2021-23:',B.quantile([.05,.95]).round(2).to_dict())
x=pd.read_excel(D+'yf_regression_2024.xlsx',sheet_name='Sheet2',index_col=0)
rows=[]
for gp in PAPER:
    gi,go=Gt.loc[gp,'in'],Gt.loc[gp,'out']
    rows.append(dict(GP=gp.replace(' Grand Prix',''),verde_2123=gi+go,sim=x.loc[gp,'yf_PitIn']+x.loc[gp,'yf_PitOut'],
        unif_2123=kt['in']*gi+kt['out']*go,lo=B['in'].quantile(.05)*gi+B['out'].quantile(.05)*go,hi=B['in'].quantile(.95)*gi+B['out'].quantile(.95)*go,
        unif_2124=ka['in']*Ga.loc[gp,'in']+ka['out']*Ga.loc[gp,'out']))
T=pd.DataFrame(rows).set_index('GP'); T.loc['Media']=T.mean(); print(T.round(1).to_string())
# contraste fuera de muestra con los eventos limpios 2024
V=Ya[Ya.Year==2024].copy()
V['pred_unif']=np.where(V.comp=='in',kt['in']*V.GP.map(Gt['in']),kt['out']*V.GP.map(Gt['out']))
V['pred_sim']=np.where(V.comp=='in',V.GP.map(x.yf_PitIn),V.GP.map(x.yf_PitOut))   # sim ajustado CON 2024 (dentro de muestra)
V['pred_const']=np.where(V.comp=='in',Yt[Yt.comp=='in'].exc.median(),Yt[Yt.comp=='out'].exc.median())
V=V.dropna(subset=['pred_unif','pred_sim'])
print('\nEventos limpios 2024:',len(V),' carreras',V.race.nunique(),V.groupby(['race','comp']).size().to_dict())
for c,s in list(V.groupby('comp'))+[('total',V)]:
    print(c,'n',len(s),' MAE unif(2021-23)',round((s.pred_unif-s.exc).abs().mean(),2),' MAE sim(2024)',round((s.pred_sim-s.exc).abs().mean(),2),
          ' MAE const(2021-23)',round((s.pred_const-s.exc).abs().mean(),2),' | sesgo unif',round((s.pred_unif-s.exc).median(),2),' sim',round((s.pred_sim-s.exc).median(),2))
print('\nPor carrera 2024 (mediana observada vs predicha):')
print(V.groupby(['race','comp'])[['exc','pred_unif','pred_sim']].median().round(1).to_string())
