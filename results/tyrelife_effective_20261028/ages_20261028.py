"""2026-10-28 — Edad efectiva de los neumaticos (prueba preliminar, pedido del autor): que no cuenten como vuelta de uso
(i) las vueltas de carrera bajo SC/VSC y (ii) las vueltas lentas de practicas y clasificacion (salida, entrada,
enfriamiento, bajo bandera, sin tiempo) en la edad con que un juego usado llega a la carrera.

Datos de carrera: data/session_<anio>_V2.csv con las correcciones de compuesto v10d (data_fix_v10d_20261028).
Practicas/clasificacion: fpq_laps_2021_2024.csv.gz (extract_fpq_20261028.py) + sesiones sprint
fpq_sprint_laps_2021_2024.csv.gz (extract_sprint_20261028.py; agregadas el 28-oct, antes faltaban).
Correccion 28-oct: con pandas 3 dos compuestos NaN no son iguales, y cada vuelta sin compuesto (Austria y Monaco 2022)
se tomaba como una tanda nueva con juego usado; ahora el NaN se compara como texto.

Vuelta de empuje (practicas/clasificacion): con tiempo, sin PitIn/PitOut, sin SC/VSC/bandera roja en TrackStatus y
LapTime <= PUSH x mejor vuelta del piloto en la sesion.
Vuelta de carrera que cuenta: TrackStatus sin 4 (SC), 5 (roja), 6 o 7 (VSC).

Edad con que un juego usado llega a la carrera (A0 = TyreLife de su 1.a vuelta - 1): se busca en practicas/clasificacion
la tanda del mismo piloto y compuesto que termino con TyreLife = A0, y hacia atras la cadena de tandas del mismo juego
(TyreLife inicial - 1 = final de la anterior). A0_ef = vueltas de empuje de la cadena + RHO x vueltas no encontradas,
con RHO = proporcion global de vueltas de empuje en las tandas encontradas.

Las vueltas de empuje se separan en clasificacion (Qualifying, Sprint Qualifying/Shootout: push_q) y el resto
(practicas y sprint: push_o), para probar en grid_20261028.py un peso distinto para las de clasificacion.
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
QSESS = ('Qualifying', 'Sprint Qualifying', 'Sprint Shootout')
# orden cronologico de las sesiones: normal FP1-FP2-FP3-Q; sprint 2021-22 FP1-Q-FP2-Sprint ("Sprint Qualifying" en 2021
# es la carrera sprint, que FastF1 llama Sprint); 2023 FP1-Q-Shootout-Sprint; 2024 FP1-SQ-Sprint-Q
_N = {'Practice 1': 1, 'Practice 2': 2, 'Practice 3': 3, 'Qualifying': 4}
ORDER = {(y, False): _N for y in (2021, 2022, 2023, 2024)}
ORDER.update({(y, True): {'Practice 1': 1, 'Qualifying': 2, 'Practice 2': 3, 'Sprint': 4} for y in (2021, 2022)})
ORDER[(2023, True)] = {'Practice 1': 1, 'Qualifying': 2, 'Sprint Shootout': 3, 'Sprint': 4}
ORDER[(2024, True)] = {'Practice 1': 1, 'Sprint Qualifying': 2, 'Sprint': 3, 'Qualifying': 4}
out_lines = []


def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out_lines.append(s)


def counts(ts):
    return ~ts.astype(str).str.contains(f'[{BAD}]')


def fpq_stints():
    F = pd.concat([pd.read_csv(os.path.join(HERE, 'fpq_laps_2021_2024.csv.gz')),
                   pd.read_csv(os.path.join(HERE, 'fpq_sprint_laps_2021_2024.csv.gz'))], ignore_index=True)
    F['TrackStatus'] = F.TrackStatus.astype(str)
    F = F.dropna(subset=['Stint', 'Compound', 'TyreLife'])
    clean = F.LapTime.notna() & (F.PitIn == 0) & (F.PitOut == 0) & counts(F.TrackStatus.fillna(1))
    best = F.LapTime.where(clean).groupby([F.Year, F.GP, F.Session, F.Driver]).transform('min')
    F['push'] = clean & (F.LapTime <= PUSH * best)
    F['push_q'] = F.push & F.Session.isin(QSESS)
    S = F.groupby(['Year', 'GP', 'Session', 'Driver', 'Stint']).agg(
        Compound=('Compound', 'first'), tl_start=('TyreLife', 'min'), tl_end=('TyreLife', 'max'),
        n=('LapNumber', 'size'), push=('push', 'sum'), push_q=('push_q', 'sum')).reset_index()
    S['prior'] = S.tl_start - 1
    sprint = S.groupby(['Year', 'GP']).Session.transform(lambda x: x.str.startswith('Sprint').any())
    S['order'] = [ORDER[(y, sp)].get(ss, np.nan) for y, sp, ss in zip(S.Year, sprint, S.Session)]
    assert S.order.notna().all(), S[S.order.isna()].Session.unique()
    S.to_csv(os.path.join(HERE, 'fpq_stints.csv'), index=False)
    for s, x in S.groupby(S.Session.str.startswith('Practice').map({True: 'practicas', False: 'clasif./sprint'})):
        say(f'  {s}: {len(x)} tandas, vueltas de empuje {x.push.sum() / x.n.sum():.3f} del total')
    return S


def chain(S, y, gp, d, comp, a0):
    """vueltas de empuje (todas y de clasificacion) y vueltas sin encontrar de un juego que llega con edad a0."""
    cand = S[(S.Year == y) & (S.GP == gp) & (S.Driver == d) & (S.Compound == comp)]
    push, push_q, left, used = 0, 0, a0, set()
    while left > 0:
        c = cand[(cand.tl_end == left) & ~cand.index.isin(used)]
        if c.empty:
            break
        r = c.sort_values('order').iloc[-1]
        used.add(r.name)
        push += r.push; push_q += r.push_q
        left = int(r.prior)
    return push, push_q, left


def main():
    say(f'PUSH = {PUSH}')
    S = fpq_stints()
    rows, starts = [], []
    for y in (2021, 2022, 2023, 2024):
        s = pd.read_csv(os.path.join(ROOT, 'data', f'session_{y}_V2.csv')).sort_values(['GP', 'Driver', 'LapNumber'])
        s = s.dropna(subset=['TyreLife', 'LapNumber']).reset_index(drop=True)
        s['cnt'] = counts(s.TrackStatus).astype(int)
        for (gp, d), x in s.groupby(['GP', 'Driver'], sort=False):
            tl, lap, cnt = x.TyreLife.values, x.LapNumber.values, x.cnt.values
            comp = x.Compound.fillna('NA').astype(str).values      # NaN != NaN rompia la deteccion de tandas
            g = gq = q = 0.0
            for i in range(len(x)):
                new = i == 0 or comp[i] != comp[i - 1] or tl[i] - tl[i - 1] != lap[i] - lap[i - 1]
                if new:
                    a0 = tl[i] - 1
                    p, pq, left = chain(S, y, gp, d, comp[i], int(a0)) if a0 > 0 else (0, 0, 0)
                    starts.append(dict(Year=y, GP=gp, Driver=d, LapNumber=lap[i], Compound=comp[i], a0=a0,
                                       push=p, push_q=pq, unmatched=left))
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
    say(f'  vueltas de empuje de clasificacion: {used.push_q.sum() / max(used.push.sum(), 1):.1%} de las de empuje encontradas; '
        f'juegos de largada {T[T.LapNumber <= 1].a0.gt(0).sum()}, sin rastrear '
        f'{used[used.LapNumber <= 1].unmatched.sum() / max(used[used.LapNumber <= 1].a0.sum(), 1):.1%}; tras parada sin '
        f'rastrear {used[used.LapNumber > 1].unmatched.sum() / max(used[used.LapNumber > 1].a0.sum(), 1):.1%}')
    say('  juego usado: A0 media {:.2f} -> A0_ef media {:.2f}'.format(used.a0.mean(), T.loc[T.a0 > 0, 'a0_ef'].mean()))
    A = A.merge(T[['a0', 'a0_ef', 'push', 'push_q', 'unmatched']], left_on='start', right_index=True)
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
