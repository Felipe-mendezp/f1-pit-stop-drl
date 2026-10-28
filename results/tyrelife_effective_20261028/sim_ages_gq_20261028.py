"""2026-10-28 — Edades de los juegos 2024 del simulador con la edad efectiva GQ (v10eGQ, adoptada por el autor).

El simulador lee dos tablas (src/tyre_sets.py):
  data/tyre_sets_2024_reg_20261021.csv          inventario al largar: Age = vueltas antes de la carrera (FastF1)
  data/tyre_race_stint_ages_2024_20261021.csv   edad del juego en cada tanda real de 2024 (rivales con estrategia real)
Aqui se construyen las mismas tablas con la edad GQ:
  inventario: los mismos juegos y la misma regla de devoluciones del reglamento (available_regulation_20261021.py,
    score 5,0, que decide con las vueltas reales); a cada juego usado se le asigna sus vueltas de empuje en las sesiones
    donde se uso (columna Uses de inventory_mixed_2024.csv: sesion, edad inicial-final) + RHO x vueltas no encontradas,
    con la misma definicion de vuelta de empuje y el mismo RHO que ages_20261028.py;
  tandas reales: a0_ef de stint_starts.csv (la misma edad de llegada que uso la estimacion), con los reemplazantes con el
    nombre del titular del auto, como en el simulador.
Las edades se redondean al entero (el simulador las guarda como enteros; error <= 0,5 vueltas).
    ~/.venvs/f1-drl-v2/bin/python results/tyrelife_effective_20261028/sim_ages_gq_20261028.py
Salida: data/tyre_sets_2024_reg_v10eGQ_20261028.csv, data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv,
        sim_ages_gq_summary.txt
"""
import os, sys, re
import numpy as np, pandas as pd
ROOT = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__))
INV = os.path.join(ROOT, 'results', 'tyre_inventory_mixed_2024_20261021')
sys.path.insert(0, HERE)
import ages_20261028 as AG
SUB = {'DOO': 'OCO', 'LAW': 'RIC', 'COL': 'SAR', 'BEA': None}       # BEA: SAI en Arabia Saudita, MAG en Azerbaiyan
BEA = {'Saudi Arabian Grand Prix': 'SAI', 'Azerbaijan Grand Prix': 'MAG'}
out = []


def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)


def push_laps_2024():
    F = pd.concat([pd.read_csv(os.path.join(HERE, 'fpq_laps_2021_2024.csv.gz')),
                   pd.read_csv(os.path.join(HERE, 'fpq_sprint_laps_2021_2024.csv.gz'))], ignore_index=True)
    F = F[F.Year == 2024].dropna(subset=['Compound', 'TyreLife']).copy()
    F['TrackStatus'] = F.TrackStatus.astype(str)
    clean = F.LapTime.notna() & (F.PitIn == 0) & (F.PitOut == 0) & AG.counts(F.TrackStatus.fillna('1'))
    best = F.LapTime.where(clean).groupby([F.GP, F.Session, F.Driver]).transform('min')
    F['push'] = (clean & (F.LapTime <= AG.PUSH * best)).astype(int)
    return F


def set_ages(F, rho):
    """inventory_mixed_2024.csv + columna ef (edad GQ) por juego."""
    S = pd.read_csv(os.path.join(INV, 'inventory_mixed_2024.csv'))
    key = F.groupby(['GP', 'Driver', 'Session', 'Compound', 'TyreLife']).push.max()
    ef, found, total = [], 0, 0
    for gp, d, c, uses in zip(S.GP, S.Driver, S.Compound, S.Uses):
        p = n = miss = 0
        for ses, a, b in re.findall(r'(Practice \d|Sprint Qualifying|Sprint|Qualifying): (\d+)-(\d+)', str(uses)):
            for tl in range(int(a) + 1, int(b) + 1):
                n += 1
                v = key.get((gp, d, ses, c, float(tl)))
                if v is None:
                    miss += 1
                else:
                    p += v
        found += n - miss; total += n
        ef.append(p + rho * miss)
    S['ef'] = ef
    return S, found / max(total, 1)


def main():
    T = pd.read_csv(os.path.join(HERE, 'stint_starts.csv'))
    U = T[T.a0 > 0]
    rho = U.push.sum() / (U.a0 - U.unmatched).sum()
    F = push_laps_2024()
    S, share = set_ages(F, rho)
    say(f'RHO = {rho:.3f}; vueltas previas de los juegos 2024 encontradas en los datos de sesiones: {share:.1%}')
    # inventario por reglamento con la edad GQ: misma seleccion de devoluciones (por vueltas reales)
    cwd = os.getcwd(); os.chdir(INV)
    try:
        src = open('available_regulation_20261021.py').read().split("if __name__ == '__main__':")[0]
        a = "for c, life, uses in zip(g.Compound, g.LapsBeforeRace, g.uses):"
        b = "for c, life, uses, ef_ in zip(g.Compound, g.LapsBeforeRace, g.uses, g.ef):"
        assert src.count(a) == 1
        src = src.replace(a, b)
        for x, y in [("sets.append(dict(comp=c, laps=int(life), last=max(idx) if idx else -1,",
                      "sets.append(dict(comp=c, laps=int(life), ef=float(ef_), last=max(idx) if idx else -1,"),
                     ("sets.append(dict(comp=c, laps=0, last=-1, sprint_laps=0))",
                      "sets.append(dict(comp=c, laps=0, ef=0.0, last=-1, sprint_laps=0))"),
                     ("return [(s['comp'], s['laps']) for s in sets if not s.get('returned')], missing",
                      "return [(s['comp'], s['laps'], s['ef']) for s in sets if not s.get('returned')], missing"),
                     ("S = pd.read_csv('inventory_mixed_2024.csv')", "S = _S_EF.copy()")]:
            assert src.count(x) == 1, x
            src = src.replace(x, y)
        ns = {'_S_EF': S, '__name__': 'regulation_gq'}
        exec(compile(src, 'available_regulation_20261021.py', 'exec'), ns)
        avail, miss = ns['run']('score', (5.0, 0.0))
    finally:
        os.chdir(cwd)
    say(f"devoluciones sin candidato: {miss} (igual que la regla original, available_regulation_variants.csv: 28)")
    G = pd.DataFrame([dict(GP=g, Driver='Driver_' + d, Compound=c, AgeRaw=l, AgeEf=e)
                      for (g, d), v in avail.items() for c, l, e in v])
    ref = pd.read_csv(os.path.join(ROOT, 'data', 'tyre_sets_2024_reg_20261021.csv'))
    k = ['GP', 'Driver', 'Compound']
    assert (G.sort_values(k + ['AgeRaw'])[k + ['AgeRaw']].values == ref.sort_values(k + ['Age'])[k + ['Age']].values).all(), \
        'el inventario no reproduce data/tyre_sets_2024_reg_20261021.csv'
    G['Age'] = G.AgeEf.round().astype(int)
    G = G.sort_values(k + ['Age']).reset_index(drop=True)
    G[k + ['Age']].to_csv(os.path.join(ROOT, 'data', 'tyre_sets_2024_reg_v10eGQ_20261028.csv'), index=False)
    u = G[G.AgeRaw > 0]
    say(f'inventario: {len(G)} juegos (igual al de v10d: {len(ref)}), {len(u)} usados; edad media de los usados '
        f'{u.AgeRaw.mean():.2f} -> {u.AgeEf.mean():.2f} (redondeada {u.Age.mean():.2f}); usados con edad 0 tras redondear '
        f'{int((u.Age == 0).sum())}')
    for gp in ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
               'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']:
        x = G[(G.GP == gp) & (G.AgeRaw > 0)]
        say(f'  {gp}: usados {len(x)}, edad media {x.AgeRaw.mean():.2f} -> {x.Age.mean():.2f}')
    # tandas reales
    R = pd.read_csv(os.path.join(ROOT, 'data', 'tyre_race_stint_ages_2024_20261021.csv'))
    T24 = T[T.Year == 2024].copy()
    T24['PitLap'] = (T24.LapNumber - 1).astype(int)
    inR = set(zip(R.GP, R.Driver))
    T24['Driver'] = 'Driver_' + T24.Driver      # solo se renombra el reemplazante que no figura con su nombre en la tabla v10d
    T24['Driver'] = [d if (g, d) in inR else 'Driver_' + ((BEA.get(g) if d == 'Driver_BEA' else SUB.get(d[7:])) or d[7:])
                     for g, d in zip(T24.GP, T24.Driver)]
    m = R.merge(T24[['GP', 'Driver', 'PitLap', 'a0', 'a0_ef']], on=['GP', 'Driver', 'PitLap'], how='left')
    # filas que no son una tanda nueva sino la continuacion de una (p. ej. vueltas faltantes): edad = TyreLife de la
    # vuelta PitLap; con GQ, TL_GQ de esa vuelta (48 filas, todas verificadas: Age == TL_raw de la vuelta PitLap)
    A = pd.read_csv(os.path.join(HERE, 'ages_2021_2024.csv.gz'))
    A = A[A.Year == 2024].copy()
    A['Driver'] = [d if (g, d) in inR else 'Driver_' + ((BEA.get(g) if d == 'Driver_BEA' else SUB.get(d[7:])) or d[7:])
                   for g, d in zip(A.GP, 'Driver_' + A.Driver)]
    A = A.set_index(['GP', 'Driver', 'LapNumber'])
    cont = m.a0.isna()
    raw_prev = np.array([A.TL_raw.get((g, d, float(l)), np.nan) for g, d, l in zip(m.GP, m.Driver, m.PitLap)])
    gq_prev = np.array([A.TL_GQ.get((g, d, float(l)), np.nan) for g, d, l in zip(m.GP, m.Driver, m.PitLap)])
    assert (raw_prev[cont] == m.Age[cont]).all()
    m.loc[cont, 'a0'] = m.Age[cont]; m.loc[cont, 'a0_ef'] = gq_prev[cont]
    assert (m.a0 == m.Age).all()
    say(f'tandas reales: {int(cont.sum())} filas son continuaciones de una tanda (edad = TL_GQ de la vuelta anterior)')
    m['Age'] = m.a0_ef.round().astype(int)
    m[R.columns].to_csv(os.path.join(ROOT, 'data', 'tyre_race_stint_ages_2024_v10eGQ_20261028.csv'), index=False)
    u = m[m.a0 > 0]
    say(f'tandas reales: {len(m)}, {len(u)} con juego usado; edad media {u.a0.mean():.2f} -> {u.a0_ef.mean():.2f} '
        f'(redondeada {u.Age.mean():.2f})')
    open(os.path.join(HERE, 'sim_ages_gq_summary.txt'), 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
