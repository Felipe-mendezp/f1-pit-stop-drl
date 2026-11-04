import pandas as pd, numpy as np
df=pd.read_csv("data/session_2024_V2.csv")
GPS=['Bahrain','Saudi Arabian','Miami','Hungarian','Belgian','Dutch','Singapore','United States']
df=df[df.GP.str.replace(' Grand Prix','',regex=False).isin(GPS)]
df=df[(df.PitIn==0)&(df.PitOut==0)&(df.TrackStatus.astype(str)=='1')]
rows=[]
for gp,g in df.groupby('GP'):
    last=g.LapNumber.max()
    late=g[g.LapNumber>=last-10]
    iv=late.Interval_front.replace(100.0,np.nan).dropna()
    iv=iv[(iv>0)&(iv<60)]
    rows.append((gp.replace(' Grand Prix',''),iv.median(),iv.quantile(.25),
                 (iv<2).mean()*100,(iv<5).mean()*100))
d=pd.DataFrame(rows,columns=['GP','gap_mediano_s','p25_s','%pares<2s','%pares<5s'])
print(d.round(2).to_string(index=False))
print("\nmediana global del gap entre autos consecutivos: %.2f s"%d.gap_mediano_s.median())
