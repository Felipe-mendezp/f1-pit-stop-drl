"""2026-10-29 §9: resumen por semilla de ent05/<etiqueta>_seedNN.csv (Miami, 200 carreras).
   python results/miami_lap6_diag_20261029/ent05_resumen_20261029.py <etiqueta> [<etiqueta> ...]"""
import sys, os, glob
# Los escenarios SC/VSC dependen del arbol de simulacion: correr tras `source .../env_seleccion_20261029.sh`.
assert 'v10eGQ' in os.environ.get('F1_SIMULATION_DIR', ''), 'falta source env_seleccion_20261029.sh'
sys_argv = sys.argv[1:]
sys.argv = [sys.argv[0], '--time-only', '--out', '/dev/null', '--n', '200']
src = open('results/miami_lap6_diag_20261029/fixed_sweep_20261029.py').read().split("rows = []")[0]
exec(compile(src, 'fs', 'exec'))   # define scen (vueltas SC/VSC por carrera)
import pandas as pd
for tag in sys_argv:
    rows = []
    for f in sorted(glob.glob(f'results/miami_lap6_diag_20261029/ent05/{tag}_seed*.csv')):
        a = pd.read_csv(f); a = a[~a.strat.str.contains('-')]
        n = sc = l6 = 0; psc = []
        for _, r in a.iterrows():
            lv, ls = scen[int(r.sim)]; neu = set(lv) | set(ls)
            laps = sorted(int(p.split(':')[0]) for p in str(r.pits).split() if p != 'nan' and int(p.split(':')[0]) >= 1)
            n += len(laps); sc += sum(l in neu or l - 1 in neu or l + 1 in neu for l in laps)
            l6 += bool(laps) and laps[0] == 6
            if any(l >= 7 for l in neu): psc.append(r.pos)
        rows.append(dict(seed=f[-6:-4], pos=a.pos.mean(), reward=a.reward.mean(), dsq=1 - a.ok.mean(),
                         first_lap6=l6 / len(a), stops=n / len(a), sc_adj=sc / max(n, 1), pos_sc_after6=np.mean(psc)))
    d = pd.DataFrame(rows)
    print(f'== {tag}'); print(d.round(3).to_string(index=False))
    print(f'media 10 semillas: pos {d.pos.mean():.2f} | mejor pos {d.pos.min():.2f} | semillas con v6 < 20 %: '
          f'{(d.first_lap6 < 0.2).sum()}/{len(d)} | semillas con pos < 12: {(d.pos < 12).sum()}')
