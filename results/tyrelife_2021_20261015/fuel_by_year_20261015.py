"""2026-10-15: prueba de la curva de combustible por anio (LapNumber y LapNumber^2 por anio) en la Ec. 1, con f fijo,
y pendiente media por vuelta (1-40) de esa curva por anio. Correr con F1_TYRELIFE_2021=none desde la raiz del repo."""
import os, numpy as np
from scipy.stats import f as Fd
p = os.path.abspath('results/tyrelife_2021_20261015/laptime_factor_20261015.py')
__file__ = p
src = open(p).read().split('\nrows = []')[0]
exec(src)
FUEL = BASE.replace('LapNumber + LapNumber_2', 'LapNumber + LapNumber_2 + LapNumber:C(Year) + LapNumber_2:C(Year)')
GF = {g: GP(g, FUEL) for g in GPS21}
for fv in [0.555, 0.66]:
    ssr0 = ssr1 = 0; k = 0; n = 0; chg = 0
    for g in GPS21:
        o0, o1 = GP_OBJ[g], GF[g]
        a = o0.fit({('2021', '*'): fv})[1]; b = o1.fit({('2021', '*'): fv})[1]
        ssr0 += a / o0.s2; ssr1 += b / o0.s2
        k += np.linalg.matrix_rank(np.column_stack([o1.X0, o1.D])) - np.linalg.matrix_rank(np.column_stack([o0.X0, o0.D]))
        n += len(o0.y)
    print(f'f={fv}: chi2 (ponderado) de agregar combustible por anio = {ssr0 - ssr1:.1f}, gl = {k}, vueltas = {n}')
# forma de la curva de combustible por anio: pendiente media de LapNumber (vuelta 1 a 40) por anio, modelo FUEL con f=0.66
import patsy
res = {}
for g in GPS21:
    o = GF[g]; b, *_ = o.fit({('2021', '*'): 0.66})
    names = patsy.dmatrix(FUEL.split('~')[1].format(ref=o.d.Compound_Detail.value_counts().idxmax(),
                          yref=sorted(o.d.Year.unique())[0]), o.d).design_info.column_names
    bb = dict(zip(names, b))
    yrs = sorted(o.d.Year.unique())
    for y in yrs:
        l1 = bb['LapNumber'] + bb.get(f'LapNumber:C(Year)[T.{y}]', 0)
        l2 = bb['LapNumber_2'] + bb.get(f'LapNumber_2:C(Year)[T.{y}]', 0)
        res.setdefault(y, []).append((l1 * 40 + l2 * 1600) / 40)  # s por vuelta, promedio vueltas 1-40
for y, v in sorted(res.items()):
    print(y, f'cambio medio por vuelta (1-40): mediana {np.median(v):+.4f} s, GP={len(v)}')
