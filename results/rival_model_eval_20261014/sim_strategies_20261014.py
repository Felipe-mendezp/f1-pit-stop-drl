"""2026-10-14: estrategias completas simuladas por cada modelo de rivales frente a las reales de 2024 (8 GP del paper).

Simulacion independiente del entorno de RL (no hay agente ni dinamica de tiempos): por cada carrera 2024 y cada piloto que
termino, se simula vuelta a vuelta la decision de parar (etapa 1) y el compuesto (etapa 2) con el estado simulado
(compuesto, vida del neumatico, ya uso dos compuestos) y, como exogenos, el estado de pista real (SC/VSC por vuelta), la
posicion real del piloto en cada vuelta y el compuesto de partida real. Ajuste de los modelos con 2021-23.
Reglas comunes a todas las variantes: no parar dos vueltas seguidas; al terminar sin dos compuestos no se fuerza nada
(se informa el % de estrategias ilegales).

Combinaciones:
  A  S0 + R  + K0   (simulador actual)
  B  S1      + K1   (paper)
  C  S2 + R2 + K1
  D  S3 + R2 + K1
  E  S3 + R  + K1   (efecto de las reglas del simulador sobre D)
  F  S3 + R3 + K1   (reglas del simulador en verde; bajo SC/VSC cualquier vuelta del periodo)
  G  S0 + R  + K1   (simulador con logit condicional: efecto de la etapa 2)
  H  S0 + R3 + K1
Metricas por GP y en conjunto: paradas medias y su distribucion, vuelta de la primera parada (fraccion de carrera:
media, cuartiles, distancia de Kolmogorov-Smirnov con la real), largo de stint, % de estrategias de una parada y las
secuencias de roles (duro/medio/blando) mas frecuentes.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rivals_common_20261014 as C
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

NSIM = int(os.environ.get('NSIM', 200))
pd.set_option('display.width', 230)
rng = np.random.default_rng(20261014)
train, test = C.load()
fits = C.fit_stop(train)
Ltr = C.long_df(train)
KM = {'K0': C.K0(Ltr), 'K1': C.K1(Ltr)}
COMBOS = {'A sim (S0+R+K0)': ('S0', 'R', 'K0'), 'B paper (S1+K1)': ('S1', '', 'K1'),
          'C S2+R2+K1': ('S2', 'R2', 'K1'), 'D S3+R2+K1': ('S3', 'R2', 'K1'), 'E S3+R+K1': ('S3', 'R', 'K1'),
          'F S3+R3+K1': ('S3', 'R3', 'K1'), 'G S0+R+K1': ('S0', 'R', 'K1'), 'H S0+R3+K1': ('S0', 'R3', 'K1')}
if os.environ.get('COMBOS'):                                     # subconjunto por letra, p. ej. COMBOS=AE
    COMBOS = {k: v for k, v in COMBOS.items() if k[0] in os.environ['COMBOS']}

raw = pd.read_csv(os.path.join(C.R.DATA_PATH, 'session_2024_V2.csv'))


def real_strategies(gp):
    r = raw[raw.GP == gp].sort_values(['Driver', 'LapNumber'])
    n = int(r.LapNumber.max())
    out = []
    for drv, d in r.groupby('Driver'):
        if d.LapNumber.max() < n - 1 or d.Compound.isin(['WET', 'INTERMEDIATE']).any():
            continue                                             # no termino o uso neumatico de lluvia
        pits = d[d.PitIn == 1].LapNumber.astype(int).tolist()
        pits = [p for p in pits if p < n]
        comps = [d.Compound_Detail.iloc[0]] + [d[d.LapNumber == p + 1].Compound_Detail.iloc[0] for p in pits
                                                if (d.LapNumber == p + 1).any()]
        out.append(dict(Driver=drv, pits=pits, comps=comps, n=n))
    return out


def simulate(gp, combo, drivers):
    s_name, rules, k_name = combo
    fit, km = fits[s_name], KM[k_name]
    t = test[test.GP == gp]
    n = int(t.NLaps.iloc[0])
    lapinfo = t.groupby('LapNumber')[['SC', 'VSC', 'YF', 'yf_first']].max().reindex(range(1, n + 1)).fillna(0)
    pos = (t.pivot_table(index='LapNumber', columns='Driver', values='Position').reindex(range(1, n + 1)).ffill().bfill())
    alts = sorted(c for (y, g, c) in C.ROLES if y == 2024 and g == gp)
    D = [d['Driver'] for d in drivers]
    S = NSIM * len(D)
    drv = np.repeat(D, NSIM)
    comp = np.repeat([d['comps'][0] for d in drivers], NSIM).astype(object)
    first = {d['Driver']: d for d in drivers}
    tl0 = {dd: float(raw[(raw.GP == gp) & (raw.Driver == dd)].sort_values('LapNumber').TyreLife.iloc[0]) for dd in D}
    tyre = np.array([tl0[x] for x in drv])
    used = np.zeros(S, int)
    seen = [{c} for c in comp]
    prev_pit = np.zeros(S, bool)
    pits = [[] for _ in range(S)]
    seq = [[c] for c in comp]
    for lap in range(1, n):
        li = lapinfo.loc[lap]
        X = pd.DataFrame(dict(y21=0, **{f'tl21_{c}': 0.0 for c in C._C21}, SC=int(li.SC), VSC=int(li.VSC), YF=int(li.YF), used_two_compounds=used,
                              EarlyLaps=int(lap <= 10), LateLaps=int(lap >= n - 10), LapsLeft=n - lap,
                              RaceFrac=lap / n, TyreLife=tyre, Compound_Detail=comp,
                              Position=pos.loc[lap, drv].values, GP=gp))
        p = np.asarray(fit.predict(X))
        if li.YF:
            ok = np.full(S, (lap > 5 and li.yf_first == 1)) if rules == 'R' else np.ones(S, bool)
        else:
            ok = C.green_allowed(np.full(S, lap), np.full(S, n - lap), tyre, rules)
        stop = (rng.random(S) < p) & ok & ~prev_pit
        idx = np.flatnonzero(stop)
        if len(idx):
            L = pd.DataFrame([dict(decision_id=i, alt=a, LapsLeft=n - lap, GP=gp, Year=2024,
                                   not_change_compound=int(a == comp[i] and used[i] == 0)) for i in idx for a in alts])
            pr = km.probs(L)
            L['p'] = pr
            for i, g in L.groupby('decision_id'):
                new = rng.choice(g.alt.values, p=g.p.values / g.p.sum())
                comp[i] = new; seen[i].add(new); used[i] = int(len(seen[i]) >= 2)
                pits[i].append(lap); seq[i].append(new)
        tyre = np.where(stop, 1.0, tyre + 1.0)
        prev_pit = stop
    return [dict(Driver=drv[i], pits=pits[i], comps=seq[i], n=n) for i in range(S)]


def metrics(strats, gp):
    roles = {c: C.ROLES[(2024, gp, c)][0] for (y, g, c) in C.ROLES if y == 2024 and g == gp}
    nst = np.array([len(s['pits']) for s in strats])
    fs = np.array([s['pits'][0] / s['n'] for s in strats if s['pits']])
    stints = np.concatenate([np.diff([0] + s['pits'] + [s['n']]) for s in strats])
    seqs = pd.Series(['-'.join(roles.get(c, '?') for c in s['comps']) for s in strats])
    legal = np.mean([len(set(s['comps'])) >= 2 for s in strats])
    return dict(paradas=nst.mean(), p0=np.mean(nst == 0), p1=np.mean(nst == 1), p2=np.mean(nst == 2),
                p3mas=np.mean(nst >= 3), primera_media=fs.mean() if len(fs) else np.nan,
                primera_q25=np.quantile(fs, .25) if len(fs) else np.nan,
                primera_q75=np.quantile(fs, .75) if len(fs) else np.nan,
                stint_medio=stints.mean(), legales=legal, _fs=fs, _seq=seqs.value_counts(normalize=True))


rows, seqrows = [], []
for gp in C.PAPER_GPS:
    real = real_strategies(gp)
    mr = metrics(real, gp)
    rows.append(dict(GP=gp, modelo='REAL', pilotos=len(real), ks_primera=0.0,
                     **{k: v for k, v in mr.items() if not k.startswith('_')}))
    seqrows.append(mr['_seq'].head(4).rename(f'{gp[:12]} REAL'))
    for cname, combo in COMBOS.items():
        sims = simulate(gp, combo, real)
        m = metrics(sims, gp)
        ks = ks_2samp(m['_fs'], mr['_fs']).statistic if len(m['_fs']) and len(mr['_fs']) else np.nan
        rows.append(dict(GP=gp, modelo=cname, pilotos=len(real), ks_primera=ks,
                         **{k: v for k, v in m.items() if not k.startswith('_')}))
        seqrows.append(m['_seq'].head(4).rename(f'{gp[:12]} {cname[:1]}'))
    print('listo', gp, flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(C.HERE, f'sim_strategies_20261014{C.TLSUF}.csv'), index=False)
print(f'\n=== Estrategias simuladas ({NSIM} por piloto) vs reales 2024, por GP')
print(df.round(3).to_string(index=False))
agg = df.groupby('modelo', sort=False)[[c for c in df.columns if c not in ('GP', 'modelo', 'pilotos')]].mean()
print('\n=== Promedio de los 8 GP (ks_primera: distancia KS de la vuelta de la primera parada; 0 = igual a la real)')
print(agg.round(3).to_string())
# error absoluto medio frente a la real, por metrica, promediado en los GP
real_m = df[df.modelo == 'REAL'].set_index('GP')
err = []
for cname in COMBOS:
    x = df[df.modelo == cname].set_index('GP')
    e = {c: (x[c] - real_m[c]).abs().mean() for c in ['paradas', 'p1', 'p2', 'primera_media', 'stint_medio']}
    e['ks_primera'] = x.ks_primera.mean()
    err.append(pd.Series(e, name=cname))
print('\n=== Error absoluto medio frente a la real (promedio de los 8 GP; menor es mejor)')
print(pd.DataFrame(err).round(3).to_string())
seq = pd.concat(seqrows, axis=1).fillna(0)
seq.to_csv(os.path.join(C.HERE, f'sim_strategies_sequences_20261014{C.TLSUF}.csv'))
