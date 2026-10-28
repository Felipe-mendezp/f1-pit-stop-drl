"""2026-10-28 — Edad efectiva de los neumaticos (prueba preliminar, pedido del autor): que no cuenten como vuelta de uso
(i) las vueltas de carrera bajo SC/VSC y (ii) las vueltas lentas de practicas y clasificacion (salida, entrada,
enfriamiento, bajo bandera, sin tiempo) en la edad con que un juego usado llega a la carrera.

Datos de carrera: data/session_<anio>_V2.csv con las correcciones de compuesto v10d (data_fix_v10d_20261028).
Practicas/clasificacion: fpq_laps_2021_2024.csv.gz (extract_fpq_20261028.py).

Vuelta de empuje (practicas/clasificacion): con tiempo, sin PitIn/PitOut, sin SC/VSC/bandera roja en TrackStatus y
LapTime <= PUSH x mejor vuelta del piloto en la sesion.
Vuelta de carrera que cuenta: TrackStatus sin 4 (SC), 5 (roja), 6 o 7 (VSC).

Edad con que un juego usado llega a la carrera (A0 = TyreLife de su 1.a vuelta - 1): se busca en practicas/clasificacion
la tanda del mismo piloto y compuesto que termino con TyreLife = A0, y hacia atras la cadena de tandas del mismo juego
(TyreLife inicial - 1 = final de la anterior). A0_ef = vueltas de empuje de la cadena + RHO x vueltas no encontradas,
con RHO = proporcion global de vueltas de empuje en las tandas encontradas.

Variantes de TyreLife (por vuelta de carrera; el factor 2021 se aplica despues, igual que hoy):
  raw : la de FastF1
  G   : A0 + vueltas de la tanda que cuentan (sin SC/VSC)
  Q   : A0_ef + vueltas de la tanda (todas)
  GQ  : A0_ef + vueltas de la tanda que cuentan
Salida: ages_2021_2024.csv.gz, fpq_stints.csv, ages_summary.txt
    ~/.venvs/f1-drl-v2/bin/python results/tyrelife_effective_20261028/ages_20261028.py
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'results', 'sim_v10d_20261028'))
import data_fix_v10d_20261028  # noqa: F401  corrige compuestos al leer los CSV
PUSH = float(os.environ.get('PUSH', 1.10))
BAD = '4567'
out_lines = []


def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out_lines.append(s)


def counts(ts):
    return ~ts.astype(str).str.contains(f'[{BAD}]')


def fpq_stints():
    F = pd.read_csv(os.path.join(HERE, 'fpq_laps_2021_2024.csv.gz'))
    F = F.dropna(subset=['Stint', 'Compound', 'TyreLife'])
    clean = F.LapTime.notna() & (F.PitIn == 0) & (F.PitOut == 0) & counts(F.TrackStatus.fillna(1))
    best = F.LapTime.where(clean).groupby([F.Year, F.GP, F.Session, F.Driver]).transform('min')
    F['push'] = clean & (F.LapTime <= PUSH * best)
    S = F.groupby(['Year', 'GP', 'Session', 'Driver', 'Stint']).agg(
        Compound=('Compound', 'first'), tl_start=('TyreLife', 'min'), tl_end=('TyreLife', 'max'),
        n=('LapNumber', 'size'), push=('push', 'sum')).reset_index()
    S['prior'] = S.tl_start - 1
    S['order'] = S.Session.map({'Practice 1': 1, 'Practice 2': 2, 'Practice 3': 4, 'Sprint Qualifying': 3,
                                'Sprint Shootout': 3, 'Sprint': 3.5, 'Qualifying': 5}).fillna(3)
    S.to_csv(os.path.join(HERE, 'fpq_stints.csv'), index=False)
    for s, x in S.groupby(S.Session.str.startswith('Practice').map({True: 'practicas', False: 'clasif./sprint'})):
        say(f'  {s}: {len(x)} tandas, vueltas de empuje {x.push.sum() / x.n.sum():.3f} del total')
    return S


def chain(S, y, gp, d, comp, a0):
    """vueltas de empuje y vueltas sin encontrar de un juego que llega con edad a0."""
    cand = S[(S.Year == y) & (S.GP == gp) & (S.Driver == d) & (S.Compound == comp)]
    push, left, used = 0, a0, set()
    while left > 0:
        c = cand[(cand.tl_end == left) & ~cand.index.isin(used)]
        if c.empty:
            break
        r = c.sort_values('order').iloc[-1]
        used.add(r.name)
        push += r.push
        left = int(r.prior)
    return push, left


def main():
    say(f'PUSH = {PUSH}')
    S = fpq_stints()
    rows, starts = [], []
    for y in (2021, 2022, 2023, 2024):
        s = pd.read_csv(os.path.join(ROOT, 'data', f'session_{y}_V2.csv')).sort_values(['GP', 'Driver', 'LapNumber'])
        s = s.dropna(subset=['TyreLife', 'LapNumber']).reset_index(drop=True)
        s['cnt'] = counts(s.TrackStatus).astype(int)
        for (gp, d), x in s.groupby(['GP', 'Driver'], sort=False):
            tl, lap, comp, cnt = x.TyreLife.values, x.LapNumber.values, x.Compound.astype(str).values, x.cnt.values
            g = gq = q = 0.0
            for i in range(len(x)):
                new = i == 0 or comp[i] != comp[i - 1] or tl[i] - tl[i - 1] != lap[i] - lap[i - 1]
                if new:
                    a0 = tl[i] - 1
                    p, left = chain(S, y, gp, d, comp[i], int(a0)) if a0 > 0 else (0, 0)
                    starts.append(dict(Year=y, GP=gp, Driver=d, LapNumber=lap[i], Compound=comp[i], a0=a0,
                                       push=p, unmatched=left))
                    k = len(starts) - 1
                    g = a0 + cnt[i]; base = 1
                else:
                    step = lap[i] - lap[i - 1]
                    g += (step - 1) + cnt[i]; base += step
                rows.append((y, gp, d, lap[i], tl[i], g, k, base))
    A = pd.DataFrame(rows, columns=['Year', 'GP', 'Driver', 'LapNumber', 'TL_raw', 'TL_G', 'start', 'laps_in_stint'])
    T = pd.DataFrame(starts)
    used = T[T.a0 > 0]
    rho = used.push.sum() / max((used.a0 - used.unmatched).sum(), 1)
    T['a0_ef'] = np.where(T.a0 > 0, T.push + rho * T.unmatched, 0.0)
    say(f'tandas de carrera {len(T)}; con juego usado {len(used)} ({len(used) / len(T):.0%}); '
        f'vueltas previas encontradas {1 - used.unmatched.sum() / used.a0.sum():.1%}; RHO = {rho:.3f}')
    say('  juego usado: A0 media {:.2f} -> A0_ef media {:.2f}'.format(used.a0.mean(), T.loc[T.a0 > 0, 'a0_ef'].mean()))
    A = A.merge(T[['a0', 'a0_ef']], left_on='start', right_index=True)
    A['TL_Q'] = A.TL_raw - A.a0 + A.a0_ef
    A['TL_GQ'] = A.TL_G - A.a0 + A.a0_ef
    for v in ('TL_G', 'TL_Q', 'TL_GQ'):
        d = A[v] - A.TL_raw
        say(f'  {v}: vueltas con edad distinta {(d != 0).mean():.1%}; diferencia media {d.mean():.2f}, percentil 5 {d.quantile(.05):.2f}')
    A.drop(columns=['start']).to_csv(os.path.join(HERE, 'ages_2021_2024.csv.gz'), index=False)
    T.to_csv(os.path.join(HERE, 'stint_starts.csv'), index=False)
    open(os.path.join(HERE, 'ages_summary.txt'), 'w').write('\n'.join(out_lines) + '\n')


if __name__ == '__main__':
    main()
