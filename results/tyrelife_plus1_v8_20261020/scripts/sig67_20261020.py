import pandas as pd,numpy as np,glob,itertools,sys
from scipy import stats
rng=np.random.default_rng(0)
for f in ['fig6_2024_realdnf','fig7_2024']:
  P={}
  for p in sorted(glob.glob(f'{f}/*.csv')):
    d=pd.read_csv(p); gp=p.split('/')[-1][:-4]
    for a,g in d.groupby('algo',sort=False): P[(gp,a)]=g.final_position.to_numpy()
  gps=sorted({k[0] for k in P}); algos=sorted({k[1] for k in P})
  assert all(len(P[(g,a)])==500 for g in gps for a in algos)
  print(f'== {f}  (diferencia = fila - columna; negativo = fila mejor)')
  rows=[]
  for a,b in itertools.combinations(algos,2):
    D=np.stack([P[(g,a)]-P[(g,b)] for g in gps])          # 8x500
    m=D.mean(); gpm=D.mean(1)
    # bootstrap estratificado por GP (carreras pareadas)
    bs=np.array([np.mean([x[rng.integers(0,500,500)].mean() for x in D]) for _ in range(2000)])
    lo,hi=np.percentile(bs,[2.5,97.5])
    w=stats.wilcoxon(D.ravel()).pvalue
    s=stats.wilcoxon(gpm).pvalue                            # nivel GP, n=8
    rows.append(dict(par=f'{a} vs {b}',dif=round(m,3),IC95=f'[{lo:.3f}, {hi:.3f}]',p_carreras=f'{w:.3g}',
                     GP_a_favor=f'{(gpm<0).sum()}/8',p_GP=f'{s:.3g}'))
  r=pd.DataFrame(rows); print(r.to_string(index=False)); print()
