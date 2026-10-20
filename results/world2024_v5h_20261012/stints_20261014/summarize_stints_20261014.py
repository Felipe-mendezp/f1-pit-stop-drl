"""Resumen de stints (trace_stints_20261014.py) — 2026-10-14. Verifica posiciones contra fig7_2024/<GP>.csv."""
import os, json, collections
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(os.path.dirname(HERE), 'fig7_2024')
ALGOS = ['RecurrentPPO', 'PPO', 'QMIP']
for g in ['Miami_OCO', 'Saudi_Arabian_NOR']:
    tr = json.load(open(os.path.join(HERE, f'fig7_{g}.json')))
    new = pd.read_csv(os.path.join(HERE, 'fig7_2024', f'{g}.csv'))
    old = pd.read_csv(os.path.join(ORIG, f'{g}.csv'))
    print(f'\n=== {g}  trazas {len(tr)}')
    rows = []
    for k, a in enumerate(ALGOS):
        pos_new = new[new.algo == a].final_position.tolist()
        assert pos_new == old[old.algo == a].final_position.tolist(), f'{a}: posiciones distintas al CSV original'
        for t, p in zip(tr[500 * k:500 * (k + 1)], pos_new):
            # t = [(lap_number antes del step, compuesto, edad del neumatico, accion)]; lap 0 = eleccion del compuesto
            # de partida. Parada en la vuelta L = accion != 0 en el step con lap_number L (incluye paradas al mismo
            # compuesto, que no cambian el compuesto y que el conteo de evaluate_all_gps no registra).
            stints, pits = [(t[0][1], None)], []
            for lap, c, life, act in t[1:]:
                if act != 0:
                    stints.append((c, None)); pits.append(lap)
            rows.append(dict(algo=a, pos=p, plan=' -> '.join(f'C{c}' for c, _ in stints), pits=pits))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, f'stints_fig7_{g}.csv'), index=False)
    for a in ALGOS:
        d = df[df.algo == a]
        print(f'-- {a}: pos media {d.pos.mean():.2f}')
        for plan, dd in d.groupby('plan'):
            laps = collections.Counter(p[0] for p in dd.pits)
            top = ', '.join(f'v{l}:{n}' for l, n in sorted(laps.items(), key=lambda x: -x[1])[:6])
            print(f'   {plan:14s} n={len(dd):3d}  pos media {dd.pos.mean():5.2f}  parada en [{top}]')
