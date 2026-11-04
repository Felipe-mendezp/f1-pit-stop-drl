"""2026-11-03 - Tabla de apendice (idea del autor): juegos disponibles y su vida de uso en cada punto de decision
marcado en la Fig. 11. Lee los volcados de scripts/tyre_sets_at_decisions_20261103.py y las filas de la figura
(agrupadas como group_rows: vuelta representativa = mayor |DRL - real|). Salida: fragmento LaTeX + tabla de texto.
    python scripts/table_tyre_sets_decisions_20261103.py --res results/decision_support_v10gsmrealhist_20261102 \
        --laps compare_laps_20261103.csv --out <tex>
"""
import argparse, json, os
import pandas as pd

GPS = [('bahrain_alo', 'Bahrain', 'Alonso'), ('saudi_arabian_nor', 'Saudi Arabia', 'Norris'),
       ('miami_oco', 'Miami', 'Ocon'), ('hungarian_tsu', 'Hungary', 'Tsunoda'),
       ('belgian_gas', 'Belgium', 'Gasly'), ('dutch_str', 'Netherlands', 'Stroll'),
       ('singapore_hul', 'Singapore', 'Hülkenberg'), ('united_states_mag', 'United States', 'Magnussen')]
NAMES3 = ['Hard', 'Medium', 'Soft']

ap = argparse.ArgumentParser()
ap.add_argument('--res', required=True)
ap.add_argument('--laps', default='compare_laps_20261103.csv')
ap.add_argument('--dump', default=None, help='carpeta con los <gp>.json (default <res>/tyre_sets_20261103)')
ap.add_argument('--out', required=True)
a = ap.parse_args()
dump = a.dump or os.path.join(a.res, 'tyre_sets_20261103')
L = pd.read_csv(os.path.join(a.res, a.laps))


def fig_rows(key):
    """Filas de la Fig. 11: grupos de vueltas consecutivas con la misma recomendacion; representante = mayor |d|."""
    g = L[(L.gp == key) & (L.algo == 'PPO') & (L.scen == '(ii)')].sort_values('lap')
    rows = [dict(lap=int(r.lap), rec=r.rec, d=-r.dpts_real_vs_rec) for r in g[g.real != g.rec].itertuples()]
    groups = []
    for r in rows:
        if groups and groups[-1][-1]['rec'] == r['rec'] and r['lap'] == groups[-1][-1]['lap'] + 1:
            groups[-1].append(r)
        else:
            groups.append([r])
    return [(g_[0]['lap'], g_[-1]['lap'], max(g_, key=lambda r: abs(r['d']))['lap']) for g_ in groups]


def cell(ages):
    return '--' if not ages else ', '.join(str(x) for x in ages)


tex, txt = [], []
for key, gp, drv in GPS:
    d = json.load(open(os.path.join(dump, f'{key}.json')))
    comps = d['available_compounds']   # ascendente = mas duro .. mas blando
    first = True
    for lo, hi, rep in fig_rows(key):
        e = d['laps'][str(rep)]
        lap_txt = f'{lo}--{hi} ({rep})' if lo != hi else str(lo)
        lap_plain = f'{lo}-{hi} ({rep})' if lo != hi else str(lo)
        mounted = f"C{e['mounted']['compound']} ({e['mounted']['age']})"
        cells = [cell(e['pool'][str(c)]) for c in comps]
        head = f"{gp} ({drv})" if first else ''
        tex.append(f"{head} & {lap_txt} & {d['n_laps'] - rep + 1} & {mounted} & " +
                   ' & '.join(f"C{c}: {c_}" for c, c_ in zip(comps, cells)) + r' \\')
        txt.append(f"{head:24s} {lap_plain:>10s}  rem {d['n_laps'] - rep + 1:>2d}  on {mounted:>8s}   " +
                   '  '.join(f"C{c}: {c_:<10s}" for c, c_ in zip(comps, cells)))
        first = False
    tex.append(r'\addlinespace')
body = '\n'.join(tex)
open(a.out, 'w').write(body + '\n')
print('\n'.join(txt))
print('\nwrote', a.out)
