import os
import pandas as pd, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
D=os.path.join(HERE,'..','..','data')+'/'
OUT=HERE+'/'
PAPER=['Bahrain Grand Prix','Saudi Arabian Grand Prix','Miami Grand Prix','Hungarian Grand Prix','Belgian Grand Prix','Dutch Grand Prix','Singapore Grand Prix','United States Grand Prix']
def status(ts):
    s=str(ts).split('.')[0]
    if '5' in s: return 'RED'
    if '4' in s: return 'SC'
    if '6' in s: return 'VSC'   # incluye 67 (VSC ending)
    if '7' in s: return 'VSCend'
    if '2' in s: return 'Y'
    return 'G'
def load():
    L=[]
    for y in (2021,2022,2023,2024):
        d=pd.read_csv(D+f'session_{y}_V2.csv'); L.append(d)
    d=pd.concat(L,ignore_index=True)
    d['st']=d['TrackStatus'].map(status)
    d=d.sort_values(['Year','GP','Driver','LapNumber'])
    g=d.groupby(['Year','GP','Driver'])
    d['prev_st']=g['st'].shift(1).fillna('G')
    d['wet']=d['Compound'].isin(['WET','INTERMEDIATE'])
    # índice dentro del evento (por carrera y vuelta, con el estado del líder ~ moda)
    lapst=d.groupby(['Year','GP','LapNumber'])['st'].agg(lambda s:s.mode().iat[0]).rename('lap_st').reset_index()
    d=d.merge(lapst,on=['Year','GP','LapNumber'])
    lapst=lapst.sort_values(['Year','GP','LapNumber'])
    k=[];prev=None;cnt=0
    for r in lapst.itertuples():
        key=(r.Year,r.GP)
        if key!=prev: cnt=0; prev=key; last='G'
        if r.lap_st in('SC','VSC'):
            cnt = cnt+1 if last==r.lap_st else 1
        else: cnt=0
        last=r.lap_st; k.append(cnt)
    lapst['ev_idx']=k
    d=d.merge(lapst[['Year','GP','LapNumber','ev_idx']],on=['Year','GP','LapNumber'])
    # referencia verde: mediana de vueltas limpias (sin pit, sin vuelta 1, verde, seco)
    clean=d[(d.st=='G')&(d.lap_st=='G')&(d.PitIn==0)&(d.PitOut==0)&(d.LapNumber>1)&(~d.wet)]
    ref=clean.groupby(['Year','GP'])['LapTime'].median().rename('Lref')
    d=d.join(ref,on=['Year','GP'])
    return d
