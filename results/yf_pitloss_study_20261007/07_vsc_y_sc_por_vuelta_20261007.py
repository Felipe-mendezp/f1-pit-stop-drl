from yf_common_20261007 import *
pd.set_option('display.width',250)
Y=pd.read_pickle(OUT+'strict.pkl')
Y['rho_ref']=Y.med/Y.Lref
print('SC limpio según vuelta del evento:')
Y['k']=Y.ev_idx.clip(upper=4)
print(Y[Y.pure=='SC'].groupby(['comp','k']).agg(n=('exc','size'),races=('race','nunique'),med=('exc','median'),ratio=('r','median'),ritmo_ref=('rho_ref','median')).round(2))
print('\nVSC limpio, evento por evento:')
cols=['race','Driver','LapNumber','comp','ev_idx','TrackStatus','LapTime','med','n1','exc','Gc','rho_ref']
print(Y[Y.pure=='VSC'][cols].sort_values(['race','LapNumber']).round(2).to_string(index=False))
print('\nSC limpio, ritmo de la referencia (vuelta SC / vuelta verde) por vuelta del evento: arriba. VSC:',round(Y[Y.pure=="VSC"].rho_ref.median(),2))
