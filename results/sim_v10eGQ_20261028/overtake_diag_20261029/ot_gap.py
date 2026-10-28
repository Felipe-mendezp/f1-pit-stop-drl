"""Prueba del umbral de 1 s: probabilidad de adelantamiento real vs. lo que hace el simulador, por gap inicial."""
import os, sys; sys.path[:0]=[os.getcwd(), os.path.join(os.getcwd(),"src")]; import config, compat; compat.setup_all()
import pickle, numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import norm
T=os.path.expanduser('~/.claude/jobs/4608e3ae/tmp/')
exec(open(T+'ot_pair.py').read().split("GPS=['Bahrain'")[0])
GPS=['Bahrain','Saudi Arabian','Miami','Hungarian','Belgian','Dutch','Singapore','United States']
P=pd.read_csv(T+'ot_support.csv')
s=df[df.nosc_L1 & df.real_dt.notna()].copy(); s['g']=s.GP.str.replace(' Grand Prix','')
s['e']=s.real_dt-s.delta_t_lap            # residuo de la diferencia de tiempos de vuelta (real - predicho)
# sigma del ruido del simulador (OLS de vuelta en verde), por GP: diferencia de dos pilotos = sqrt(2)*sigma
CH='results/sim_v10eGQ_20261028/artifacts'
sig={}
for g in GPS:
    with open(f'{CH}/{g} Grand Prix/laptime_clear_hist.pkl','rb') as fh: m=pickle.load(fh)
    sig[g]=np.sqrt(m.sigma_square_)
print('sigma OLS por GP (s):',{k:round(v,2) for k,v in sig.items()})
print('sd empirica de e por GP (s):',s.groupby('g').e.std().round(2).to_dict(), '| sim: sqrt2*sigma', {k:round(np.sqrt(2)*v,2) for k,v in sig.items()})
rng=np.random.default_rng(0)
def lg(a,b,x): return 1/(1+np.exp(-np.clip(a+b*x,-500,500)))
for v in 'CE':
    Q=P[P.v==v].set_index('GP'); a=Q.const.reindex(s.g).values; b=Q.slope.reindex(s.g).values
    s['tau_'+v]=lg(a,b,s.delta_total.values)                                  # logit en el gap medio
    z=rng.standard_normal((len(s),400))*(np.sqrt(2)*s.g.map(sig).values)[:,None]
    s['sim_'+v]=lg(a[:,None],b[:,None],s.delta_total.values[:,None]+z).mean(1)  # logit en el gap con ruido (como el simulador)
s['det']=norm.cdf(-s.delta_total/(np.sqrt(2)*s.g.map(sig)))                 # regla determinista con ruido del simulador
s['real_det']=(s.delta_real<0).astype(int)                                    # adelanta <=> llega antes con tiempos reales
s['sim_actual']=np.where(s.interval_front_real<=1.0,s.sim_E,s.det)            # simulador hoy (umbral 1 s), con logit E
pd.set_option('display.width',250)
for lab,mask in [('gap inicial <= 1 s',s.interval_front_real<=1.0),('gap inicial 1-2 s',(s.interval_front_real>1)&(s.interval_front_real<=2)),('gap inicial > 2 s',s.interval_front_real>2)]:
    x=s[mask].copy(); x['bin']=pd.cut(x.delta_total,[-30,-5,-2,-1,-0.5,0,0.5,1,2,5])
    t=x.groupby('bin',observed=False).agg(n=('ot_pair','size'),real=('ot_pair','mean'),llega_antes_real=('real_det','mean'),
        logit_C=('tau_C','mean'),logit_C_ruido=('sim_C','mean'),logit_E_ruido=('sim_E','mean'),determ_ruido=('det','mean'))
    print(f'\n== {lab}: n={len(x)}, adelantamientos={int(x.ot_pair.sum())}  (promedios por tramo del delta medio predicho)')
    print(t.round(3).T.to_string())
    br={c:((x[c]-x.ot_pair)**2).mean() for c in ['tau_C','sim_C','sim_E','det']}
    print('Brier:',{k:round(v,4) for k,v in br.items()},'| esperado total vs real:',{c:round(x[c].sum(),1) for c in ['tau_C','sim_C','sim_E','det']},int(x.ot_pair.sum()))
    print('etiqueta vs llega-antes-real: coinciden', f'{(x.ot_pair==x.real_det).mean():.1%}', '| llega antes pero no adelanta:',int(((x.real_det==1)&(x.ot_pair==0)).sum()),'| adelanta sin llegar antes:',int(((x.real_det==0)&(x.ot_pair==1)).sum()))
s.to_pickle(T+'ot_gap.pkl')
