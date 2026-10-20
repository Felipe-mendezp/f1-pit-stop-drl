import json, glob, os, re, sys
import pandas as pd, numpy as np
RD='.'
def load(algo, tag, n):
    m=json.load(open(f'{RD}/{algo}/{tag}_lap{n}_meta.json')); df=pd.read_csv(f'{RD}/{algo}/{tag}_lap{n}.csv'); return m,df
def laps(algo,tag):
    return sorted(int(re.search(r'_lap(\d+)_meta',p).group(1)) for p in glob.glob(f'{RD}/{algo}/{tag}_lap*_meta.json'))
def stints(algo, tag):
    L=laps(algo,tag); m,_=load(algo,tag,L[-1]); N=m['n_laps']; av=sorted(m['available_compounds'])
    name={av[0]:'H',av[1]:'M',av[2]:'S'}
    h=m['history']; starts=[(0,h[0][1])]+[(l,c) for l,c,p in h if p and l>=1]
    out=[]
    for i,(l,c) in enumerate(starts):
        e=starts[i+1][0] if i+1<len(starts) else N
        out.append(f'{name.get(c,c)}(C{c}) {e-l}v')
    return ' -> '.join(out)
def table(algo,tag,n):
    m,df=load(algo,tag,n)
    g=df.groupby('action_label').agg(pts=('points','mean'),pts_se=('points',lambda x:x.std()/np.sqrt(len(x))),pos=('final_position','mean'),stops=('n_stops','mean'),
        pitlaps=('pit_laps',lambda x:x.astype(str).value_counts().index[0]))
    # paired difference vs recommended
    rec=m['recommended_label']; st=m['agent_state_at_lap']
    piv=df.pivot(index='rollout',columns='action_label',values='points')
    g['d_vs_rec']=[ (piv[a]-piv[rec]).mean() for a in g.index]
    g['d_se']=[ (piv[a]-piv[rec]).std()/np.sqrt(len(piv)) for a in g.index]
    print(f'--- {algo} {tag} lap {n}: rec={rec}  estado={st}')
    print(g.round(2).to_string())
for algo in ['PPO','RecurrentPPO']:
    print('=====',algo)
    for t in sorted({re.sub(r'_lap\d+_meta\.json$','',os.path.basename(p)) for p in glob.glob(f'{RD}/{algo}/*_lap1_meta.json')}):
        print(f'{t:32s}', stints(algo,t))
for a,t,ns in eval(sys.argv[1]) if len(sys.argv)>1 else []:
    for n in ns: table(a,t,n)
