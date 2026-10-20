"""2026-10-07 — Regresión única del tiempo de vuelta bajo SC/VSC (2021-23, filtro A, vueltas 100% en el estado):
  LapTime/Lref_g = rho_VSC*VSC + rho_SCl*SC_libre + rho_SCf*SC_fila + u_g(SC)            [ritmo, con efecto GP]
                 + k_in*PitIn*Gin_g/Lref_g + k_out*PitOut*Gout_g/Lref_g                   [pérdida de pit]
                 + v_(carrera,vuelta) + e                                                 [ruido común de la vuelta]
SC_libre = 2a vuelta real del evento (antes de juntarse; equivale a la 1a vuelta SC del simulador),
SC_fila = vueltas >=3 (autos alineados). La vuelta real del despliegue se excluye (contaminada).
Paradas solo en VSC o SC_libre (paradas en SC_fila se excluyen: su tiempo mezcla el reagrupamiento)."""
from yf_common_20261007 import *
import statsmodels.formula.api as smf
d=load(); ws=d.groupby(['Year','GP'])['wet'].mean(); d=d.join(ws.rename('ws'),on=['Year','GP'])
d=d[(d.ws<=0.10)&(~d.wet)&(d.Year<=2023)&(d.LapNumber>1)&d.LapTime.notna()]
d['ts']=d.TrackStatus.astype(str).str.split('.').str[0]
E=pd.read_pickle(OUT+'clean_events.pkl'); E=E[(E.ref_src=='vuelta_pura')&(E.pure=='G')&(E.Year<=2023)]
G=E.groupby(['GP','comp'])['exc'].median().unstack()
Lg=d.groupby('GP')['Lref'].median()
y=d[d.ts.isin(['4','6'])].copy()
y['VSC']=(y.ts=='6').astype(int); y['SCl']=((y.ts=='4')&(y.ev_idx==2)).astype(int); y['SCf']=((y.ts=='4')&(y.ev_idx>=3)).astype(int)
y=y[(y.VSC+y.SCl+y.SCf)==1]
pit=(y.PitIn+y.PitOut)>0
y=y[~(pit&(y.SCf==1))]
y=y.join(G.rename(columns={'in':'Gin','out':'Gout'}),on='GP').dropna(subset=['Gin','Gout'])
y['ratio']=y.LapTime/y.Lref
y['PI']=y.PitIn*y.Gin/y.Lref; y['PO']=y.PitOut*y.Gout/y.Lref
y['SC']=y.SCl+y.SCf
y['rl']=y.Year.astype(str)+y.GP+y.LapNumber.astype(str)
y=y[(y.ratio>0.9)&(y.ratio<2.2)]
print('Muestra:',len(y),'vueltas;',y[['Year','GP']].drop_duplicates().shape[0],'carreras;',
      {'VSC':int(y.VSC.sum()),'SC_libre':int(y.SCl.sum()),'SC_fila':int(y.SCf.sum()),'PitIn':int(y.PitIn.sum()),'PitOut':int(y.PitOut.sum())})
m=smf.mixedlm('ratio ~ 0 + VSC + SCl + SCf + PI + PO', y, groups=y['GP'], re_formula='~0+SC',
              vc_formula={'rl':'0+C(rl)'}).fit(method='lbfgs')
print(m.summary().tables[1].iloc[:7].round(3))
fe=m.fe_params; re=m.random_effects
print('sd efecto GP en SC:',round(float(np.sqrt(m.cov_re.iloc[0,0])),3))
x=pd.read_excel(D+'yf_regression_2024.xlsx',sheet_name='Sheet2',index_col=0)
rows=[]
for gp in PAPER:
    L=Lg[gp]; u=float(re[gp].iloc[0]) if gp in re else 0.0
    pit=fe.PI*G.loc[gp,'in']+fe.PO*G.loc[gp,'out']
    rows.append(dict(GP=gp.replace(' Grand Prix',''),Lref=L,
        VSC_sin=fe.VSC*L, VSC_con=fe.VSC*L+pit, SC1_sin=(fe.SCl+u)*L, SC1_con=(fe.SCl+u)*L+pit, SCfila=(fe.SCf+u)*L,
        perdida_pit=pit, sim_VSC=x.loc[gp,'yf_VSC'], sim_SC=x.loc[gp,'yf_SC'], sim_pit=x.loc[gp,'yf_PitIn']+x.loc[gp,'yf_PitOut']))
T=pd.DataFrame(rows).set_index('GP')
print('\nTiempos predichos (s), GPs del paper. "con" = in-lap + out-lap del auto que para (suma del exceso de ambas vueltas):')
print(T.round(1).to_string())
print('\nComo razón sobre Lref:'); print((T[['VSC_sin','SC1_sin','SCfila']].div(T.Lref,axis=0)).round(3).to_string())
