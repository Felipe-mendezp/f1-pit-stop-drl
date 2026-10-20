from yf_common_20261007 import *
d=load()
print(d.shape); print(d.st.value_counts())
# Wet share por carrera
wet=d.groupby(['Year','GP'])['wet'].mean()
ev=d.drop_duplicates(['Year','GP','LapNumber'])
ev=ev[ev.ev_idx==1].groupby(['Year','GP','lap_st']).size().unstack(fill_value=0)
print('\nEventos SC/VSC (inicios) 2021-24 por GP del paper:')
print(ev.reset_index().query('GP in @PAPER').groupby('GP')[['SC','VSC']].sum())
print('\nTotal carreras con datos:',d.groupby(['Year','GP']).ngroups, ' con algún SC:', (ev.get('SC',0)>0).sum(),' con algún VSC:',(ev.get('VSC',0)>0).sum())
# pits por índice de evento
yf=d[d.lap_st.isin(['SC','VSC'])&(~d.wet)]
print('\nPitIn bajo YF según vuelta del evento:')
print(yf[yf.PitIn==1].groupby(['lap_st','ev_idx']).size().unstack(fill_value=0))
print('\nPitIn bajo YF por GP del paper (2021-24), SC / VSC:')
print(yf[(yf.PitIn==1)&yf.GP.isin(PAPER)].groupby(['GP','lap_st']).size().unstack(fill_value=0))
