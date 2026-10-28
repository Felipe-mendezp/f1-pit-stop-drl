"""Formas funcionales del logit de adelantamiento (muestra C: L+1 sin SC/VSC/roja, etiqueta por par, todos los gaps)."""
import os, numpy as np, pandas as pd, statsmodels.api as sm, warnings; warnings.filterwarnings('ignore')
T=os.path.expanduser('~/.claude/jobs/4608e3ae/tmp/')
exec(open(T+'ot_pair.py').read().split("GPS=['Bahrain'")[0].replace("print('muestra C:',","("))
s=df[df.nosc_L1].copy(); s['g']=s.GP.str.replace(' Grand Prix',''); s['race']=s.Year.astype(str)+s.g
x=s.delta_total.values
def X_of(d, form):
    cols={'lin':[d],'quad':[d,d**2],'cub':[d,d**2,d**3],'sqrt':[d,np.sign(d)*np.sqrt(np.abs(d))],
          'asinh':[np.arcsinh(d)],'lin+asinh':[d,np.arcsinh(d)],'pw':[d,np.minimum(d,0)]}[form]   # pw: pendiente distinta a cada lado de 0
    return np.column_stack(cols)
FORMS=['lin','quad','cub','sqrt','asinh','lin+asinh','pw']
GPS=sorted(s.g.unique())
def design(sub, form, pooled_extra):
    B=X_of(sub.delta_total.values, form); cols=[]; names=[]
    for g in GPS:
        m=(sub.g==g).values.astype(float)
        cols.append(m); names.append(g+':c')
        lim = 1 if pooled_extra and B.shape[1]>1 else B.shape[1]
        for j in range(lim): cols.append(m*B[:,j]); names.append(f'{g}:b{j}')
    if pooled_extra and B.shape[1]>1:
        for j in range(1,B.shape[1]): cols.append(B[:,j]); names.append(f'pool:b{j}')
    return np.column_stack(cols)
def fit(sub, form, pooled): return sm.Logit(sub.ot_pair.values, design(sub, form, pooled)).fit(disp=False, maxiter=200, method='bfgs' if form=='cub' else 'newton')
rows=[]
races=s.race.unique()
for form in FORMS:
    for pooled in ([False] if form in ('lin','asinh') else [False, True]):
        f=fit(s, form, pooled); k=len(f.params)
        # validacion cruzada: dejar una carrera fuera
        ll=0.0
        for r in races:
            tr=s[s.race!=r]; te=s[s.race==r]
            try:
                ft=fit(tr, form, pooled); p=np.clip(ft.predict(design(te, form, pooled)),1e-12,1-1e-12)
            except Exception: p=np.full(len(te),np.nan)
            ll+=np.sum(te.ot_pair*np.log(p)+(1-te.ot_pair)*np.log(1-p))
        # monotonia: probabilidad decreciente en delta en [-30, 30] para cada GP
        grid=np.linspace(-30,30,601); mono=True
        for g in GPS:
            G=pd.DataFrame({'delta_total':grid,'g':g}); p=f.predict(design(G, form, pooled))
            mono &= bool(np.all(np.diff(p)<=1e-12))
        rows.append(dict(forma=form, extra='comun' if pooled else 'por GP', k=k, LL=f.llf, AIC=2*k-2*f.llf, BIC=k*np.log(len(s))-2*f.llf, LL_cv=ll, monotona=mono))
R=pd.DataFrame(rows); R['dAIC']=R.AIC-R.AIC.min(); R['dBIC']=R.BIC-R.BIC.min(); R['dLLcv']=R.LL_cv-R.LL_cv.max()
pd.set_option('display.width',200); print('n',len(s),'adelantamientos',int(s.ot_pair.sum()),'carreras',len(races))
print(R.round(1).to_string(index=False))
# calibracion por tramos: lineal vs mejor alternativa
best=R.sort_values('LL_cv',ascending=False).iloc[0]; print('\nmejor en validacion cruzada:',best.forma,best.extra)
bins=[-30,-5,-2,-1,-0.5,0,0.5,1,2,5]
out=pd.DataFrame({'bin':pd.cut(s.delta_total,bins),'real':s.ot_pair})
out['lin']=fit(s,'lin',False).predict(design(s,'lin',False))
out['alt']=fit(s,best.forma,best.extra=='comun').predict(design(s,best.forma,best.extra=='comun'))
print(out.groupby('bin',observed=False).agg(n=('real','size'),real=('real','mean'),lineal=('lin','mean'),alternativa=('alt','mean')).round(3).T.to_string())
s.to_pickle(T+'otsim/sampleC.pkl')
