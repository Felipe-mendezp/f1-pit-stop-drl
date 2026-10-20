from yf_common_20261007 import *
import statsmodels.formula.api as smf
d=load()
ws=d.groupby(['Year','GP'])['wet'].mean()
d=d.join(ws.rename('ws'),on=['Year','GP']); d=d[(d.ws<=0.10)&(~d.wet)]
nopit=d[(d.PitIn==0)&(d.PitOut==0)]
d=d.join(nopit.groupby(['Year','GP','LapNumber'])['LapTime'].median().rename('med'),on=['Year','GP','LapNumber'])
d['exc']=d.LapTime-d.med
d=d.sort_values(['Year','GP','Driver','LapNumber']); g=d.groupby(['Year','GP','Driver'])
for c in ['exc','PitOut','lap_st']: d['n_'+c]=g[c].shift(-1)
st=d[(d.PitIn==1)&(d.n_PitOut==1)&(d.LapNumber>1)&d.lap_st.isin(['G','SC','VSC'])].dropna(subset=['exc','n_exc']).copy()
st['loss']=st.exc+st.n_exc
st['cls']=np.where(st.lap_st=='G','G',st.lap_st+np.where(st.ev_idx==1,'1','2+'))
st['out']=np.where(st.n_lap_st=='G','g','y')
# pérdida verde por GP (todas las temporadas, robusta)
G=st[st.cls=='G'].groupby('GP')['loss'].median().rename('Gloss')
st=st.join(G,on='GP')
yf=st[st.cls!='G'].copy(); yf['key']=yf.cls+'_'+yf.out
yf=yf[yf.loss.abs()<60]
print('Muestra YF (filtro A, |loss|<60):'); print(yf.groupby('key')['loss'].agg(['count','median']).round(2))
print('ratio loss/Gloss por clave:'); print((yf.loss/yf.Gloss).groupby(yf.key).median().round(2))
# modelo agrupado: loss = kappa_key * Gloss + b_GP + e  (robusto vía Huber por simplicidad: RLM sin RE; y MixedLM)
yf['r']=yf.loss/yf.Gloss
m=smf.mixedlm('loss ~ 0 + C(key):Gloss', yf, groups=yf['GP']).fit()
print(m.summary().tables[1].round(3)); print('sd GP RE:',np.sqrt(m.cov_re.iloc[0,0]).round(2),' sd resid:',np.sqrt(m.scale).round(2))
# CV leave-one-GP-out: predecir pérdida de cada parada YF de un GP con (a) mediana bruta del GP (si existe, otros años), (b) kappa global*Gloss, (c) MixedLM sin ese GP
err={'kappa_pool':[], 'raw_GP_other_years':[], 'sim_actual':[]}
yf['race']=yf.Year.astype(str)+'_'+yf.GP
res=[]
for race,te in yf.groupby('race'):
    tr=yf[yf.race!=race]
    gm=tr.groupby('key')['loss'].median()
    kap=tr.groupby('key')['r'].median()
    try:
        mm=smf.mixedlm('loss ~ 0 + C(key):Gloss', tr, groups=tr['GP']).fit()
        re=mm.random_effects
    except Exception: mm=None
    for r in te.itertuples():
        a=gm.get(r.key,np.nan); b=kap.get(r.key,np.nan)*r.Gloss
        trg=tr[(tr.GP==r.GP)&(tr.key==r.key)]
        dd=trg.loss.median() if len(trg) else a
        c=np.nan
        if mm is not None and f'C(key)[{r.key}]:Gloss' in mm.fe_params:
            c=mm.fe_params[f'C(key)[{r.key}]:Gloss']*r.Gloss + (re[r.GP].iloc[0] if r.GP in re else 0)
        res.append(dict(key=r.key,GP=r.GP,y=r.loss,glob=a,kappa=b,raw_gp=dd,mixed=c))
R=pd.DataFrame(res).dropna()
print('\nCV deja-una-carrera-fuera, error absoluto medio por parada (s):')
for k,s in R.groupby('key'):
    print(k, len(s), {m: round((s[m]-s.y).abs().mean(),2) for m in ['glob','kappa','raw_gp','mixed']})
print('TOTAL',len(R), {m: round((R[m]-R.y).abs().mean(),2) for m in ['glob','kappa','raw_gp','mixed']})
# tabla GPs paper
x=pd.read_excel(D+'yf_regression_2024.xlsx',sheet_name='Sheet2',index_col=0)
simloss=(x.yf_PitIn+x.yf_PitOut).rename('sim_YF'); simG=(x.reg_PitIn+x.reg_PitOut).rename('sim_G2024')
re=m.random_effects
rows=[]
for gp in PAPER:
    Gl=G.get(gp); row=dict(GP=gp.replace(' Grand Prix',''),G_datos=Gl,G_sim2024=simG.get(gp),YF_sim=simloss.get(gp))
    for k in ['SC1_y','VSC1_y']:
        s=yf[(yf.GP==gp)&(yf.key==k)]
        row[k+'_n']=len(s); row[k+'_raw']=s.loss.median() if len(s) else np.nan
        row[k+'_pool']=m.fe_params[f'C(key)[{k}]:Gloss']*Gl+(re[gp].iloc[0] if gp in re else 0)
    rows.append(row)
print(pd.DataFrame(rows).round(1).to_string(index=False))
