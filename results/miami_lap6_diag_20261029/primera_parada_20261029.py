import pandas as pd, glob
D='results/seed_selection_v5_20261010/v10f'
out=[]
for f in sorted(glob.glob(f'{D}/val_*.csv')):
    k=f.split('val_')[1][:-4]; al=k.split('_')[-1]; gp=k[:-len(al)-1]
    d=pd.read_csv(f)
    g=d.groupby('seed')
    out.append(dict(gp=gp,algo=al,pos=d.final_position.mean(),
        share_fp6=(d.first_pit==6).mean(), share_fp_le7=(d.first_pit.between(1,7)).mean(),
        med_fp=d.first_pit.median(), npits=d.n_pits.mean(),
        seeds_fp6=(g.first_pit.apply(lambda x:(x==6).mean())>0.5).sum()))
df=pd.DataFrame(out)
pd.set_option('display.width',200)
print(df.pivot(index='gp',columns='algo',values='share_fp6').round(2))
print(df.pivot(index='gp',columns='algo',values='med_fp'))
print(df.pivot(index='gp',columns='algo',values='seeds_fp6'))
print(df.pivot(index='gp',columns='algo',values='pos').round(2))
m=pd.concat([pd.read_csv(f'{D}/val_Miami_OCO_{a}.csv').assign(algo=a) for a in ['DQN','A2C','TRPO','PPO','RecurrentPPO']])
print(m.groupby('algo').pits.value_counts().groupby(level=0).head(4))
print(m.groupby('algo').a0.value_counts())
