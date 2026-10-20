"""Resumen del barrido diagnóstico decision_support_diag_20260930.
Por (algo, GP, escenario): paradas en la realización, DSQ, y cómo se compara
la recomendación del DRL con la mejor acción según la media de las M simulaciones."""
import glob, json, os, re, sys
import pandas as pd

RD = sys.argv[1]
rows = []
for algo in ['PPO', 'RecurrentPPO']:
    for suffix, scen in [('_realyf', '(i)'), ('_real', '(ii)')]:
        metas = glob.glob(os.path.join(RD, algo, f'*{suffix}_lap*_meta.json'))
        tags = sorted({re.sub(r'_lap\d+_meta\.json$', '', os.path.basename(m)) for m in metas})
        for tag in tags:
            if not tag.endswith(suffix):
                continue
            laps = sorted(int(re.search(r'_lap(\d+)_meta', m).group(1))
                          for m in glob.glob(os.path.join(RD, algo, f'{tag}_lap*_meta.json')))
            last = json.load(open(os.path.join(RD, algo, f'{tag}_lap{laps[-1]}_meta.json')))
            N = last['n_laps']
            if laps[-1] < N - 1:
                continue  # corrida incompleta
            hist = last['history']
            c0 = hist[0][1]
            pits = [(l, c) for l, c, p in hist if p and l >= 1]
            changed = any(c != c0 for _, c in pits)
            n_best = n_rec_nopit = n_best_pit_rec_nopit = 0; bad_pits = []
            regret_pos, regret_pts = [], []
            for n in laps:
                m = json.load(open(os.path.join(RD, algo, f'{tag}_lap{n}_meta.json')))
                df = pd.read_csv(os.path.join(RD, algo, f'{tag}_lap{n}.csv'))
                g = df.groupby('action').agg(pos=('final_position', 'mean'), pts=('points', 'mean'))
                rec = m['recommended_action']
                best = g['pos'].idxmin()
                n_best += rec == best
                n_rec_nopit += rec == 0
                n_best_pit_rec_nopit += (rec == 0) and (best != 0)
                if rec != 0 and best == 0:
                    bad_pits.append(f"{n}(+{g.loc[rec,'pos']-g.loc[0,'pos']:.1f})")
                regret_pos.append(g.loc[rec, 'pos'] - g['pos'].min())
                regret_pts.append(g['pts'].max() - g.loc[rec, 'pts'])
            rows.append({
                'algo': algo, 'scen': scen, 'gp': tag.replace(suffix, ''), 'N': N,
                'start': f'C{c0}',
                'pits (vuelta:comp)': ' '.join(f'{l}:C{c}' for l, c in pits) or '—',
                'DSQ': not changed,
                'rec=mejor %': round(100 * n_best / len(laps)),
                'regret pos medio': round(sum(regret_pos) / len(laps), 2),
                
                'n rec=no pit, mejor=pit': n_best_pit_rec_nopit,
                'rec=pit, mejor=no pit (vuelta(+pos))': ' '.join(bad_pits) or '—',
            })

out = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(out.to_string(index=False))
out.to_csv(os.path.join(RD, 'summary_diag.csv'), index=False)
