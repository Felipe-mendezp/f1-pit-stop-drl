from yf_common_20261007 import *
E=pd.read_pickle(OUT+'clean_events.pkl')
E['race']=E.Year.astype(str)+' '+E.GP
S=E[(E.ref_src=='vuelta_pura')].copy()
G=S[S.pure=='G'].groupby(['GP','comp'])['exc'].median().unstack()   # componente verde por GP
S=S.join(G.rename(columns={'in':'Gin','out':'Gout'}),on='GP')
S['Gc']=np.where(S.comp=='in',S.Gin,S.Gout)
Y=S[S.pure!='G'].copy(); Y['r']=Y.exc/Y.Gc
print('Referencia estricta (autos sin pit con vuelta 100% en el mismo estado, >=3):')
t=Y.groupby('cls').agg(n=('exc','size'),races=('race','nunique'),GPs=('GP','nunique'),med=('exc','median'),
    q25=('exc',lambda s:s.quantile(.25)),q75=('exc',lambda s:s.quantile(.75)),ratio_verde=('r','median'))
print(t.round(2))
print('\nComponente verde (todas las carreras): in',round(S[S.pure=="G"].query("comp=='in'").exc.median(),2),' out',round(S[S.pure=="G"].query("comp=='out'").exc.median(),2))
# robustez: mediana por carrera y luego mediana de carreras (peso igual por carrera)
print('\nMediana de medianas por carrera:'); print(Y.groupby(['cls','race'])['exc'].median().groupby('cls').agg(['count','median']).round(2))
print('\nPor carrera (out_SC):'); print(Y[Y.cls=='out_SC'].groupby('race')['exc'].agg(['count','median']).round(1).to_string())
print('\nPor carrera (in_SC):'); print(Y[Y.cls=='in_SC'].groupby('race')['exc'].agg(['count','median']).round(1).to_string())
# CV leave-one-race-out por componente: constante global vs kappa*verde_GP vs mediana propia del GP
out=[]
for race,te in Y.groupby('race'):
    tr=Y[Y.race!=race]
    for r in te.itertuples():
        trc=tr[tr.cls==r.cls]
        if len(trc)<3: continue
        own=trc[trc.GP==r.GP]
        out.append(dict(cls=r.cls,y=r.exc,const=trc.exc.median(),kappa=trc.r.median()*r.Gc,
                        gp=own.exc.median() if len(own) else trc.exc.median()))
O=pd.DataFrame(out)
print('\nCV deja-una-carrera-fuera (MAE, s):')
print(O.groupby('cls').apply(lambda s: pd.Series({'n':len(s),**{m:(s[m]-s.y).abs().mean() for m in ['const','kappa','gp']}})).round(2))
Y.to_pickle(OUT+'strict.pkl'); G.to_pickle(OUT+'green_comp.pkl')
