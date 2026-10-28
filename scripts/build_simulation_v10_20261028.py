import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

"""
2026-10-28 — Simulador v10 definitivo = simulation_v10_20261027 con los coeficientes de piloto de la Ec. 2 re-estimados
con la definicion v10 (Ec. 1 por carrera sin pasos por el pit lane y TyreLife 2021 x 0.740):
results/driver_coef_v10_20261028/v10/eq2_fallbacks_20261014.py. En el arbol del 27-oct los pilotos eran los de v6
(factor legacy, con pasos); diferencia <= 0.005 s/vuelta (results/driver_coef_v10_20261028/README.md).

Por GP solo cambia laptime_clear.pkl (dummies de los 19 pilotos != VER; el resto de coeficientes y el intercepto son los
del arbol del 27-oct). Los demas archivos son links a los archivos reales del arbol del 27-oct. Se verifica que
laptime_clear difiera solo en los dummies de piloto. El arbol mono-piloto no cambia (simulation_onedriver_v10_20261027:
el entorno de un piloto predice con VER, referencia de la Ec. 1).
    python scripts/build_simulation_v10_20261028.py
"""
import importlib.util
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_regressions import apply_coef_override
from build_simulation_v5_20261010 import PAPER_GPS, SEAT, DUMMY
from environment.f1_env_all_drivers import ALL_DRIVERS

BASE = config.MODELS_DIR / 'simulation_v10_20261027'
EQ2 = config.ROOT_DIR / 'results' / 'driver_coef_v10_20261028' / 'v10' / 'eq2_fallbacks_20261014.py'


def main():
    dst = config.MODELS_DIR / 'simulation_v10_20261028'
    if os.path.exists(dst):
        raise SystemExit(f'{dst} ya existe; no se sobrescribe')
    spec = importlib.util.spec_from_file_location('eq2v10', EQ2)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    eq2 = mod.Eq2()
    print('carreras excluidas:', sorted(eq2.excluded))
    os.makedirs(dst)
    audit, src = [], pd.read_csv(BASE / 'sources.csv')
    for gp in PAPER_GPS:
        gdir = dst / gp; os.makedirs(gdir)
        for f in sorted(os.listdir(BASE / gp)):
            if f == 'laptime_clear.pkl':
                continue
            os.symlink(os.path.relpath(os.path.realpath(BASE / gp / f), gdir), gdir / f)
        with open(BASE / gp / 'laptime_clear.pkl', 'rb') as fh:
            fp = pickle.load(fh)
        for name in ALL_DRIVERS:
            drv = name.replace('Driver_', '')
            key = DUMMY.format(drv)
            old = float(fp.coef_[fp.feature_names_.index(key)]) if key in fp.feature_names_ else 0.0
            if drv == 'VER':
                audit.append(dict(GP=gp, driver=drv, coef_v10_20261027=old, coef_v10_20261028=0.0, spec='referencia'))
                continue
            r = eq2.predict(2024, gp, SEAT.get((gp, drv), drv))
            if not pd.notna(r['pred']):
                raise SystemExit(f'{gp} {drv}: sin prediccion')
            apply_coef_override(fp, key, r['pred'])
            audit.append(dict(GP=gp, driver=drv, coef_v10_20261027=old, coef_v10_20261028=r['pred'], spec=r['spec']))
        with open(gdir / 'laptime_clear.pkl', 'wb') as fh:
            pickle.dump(fp, fh)
    src.loc[src.file == 'laptime_clear.pkl', 'source'] = ('results/sim_v10_20261027/artifacts/<GP>/laptime_clear.pkl + '
                                                          'pilotos de results/driver_coef_v10_20261028/v10')
    src.to_csv(dst / 'sources.csv', index=False)
    A = pd.DataFrame(audit)
    A['dif'] = A.coef_v10_20261028 - A.coef_v10_20261027
    A.to_csv(dst / 'driver_coef_audit.csv', index=False)
    with open(dst / 'README.md', 'w') as fh:
        fh.write('Arbol 2026-10-28 (scripts/build_simulation_v10_20261028.py): simulation_v10_20261027 con los '
                 'coeficientes de piloto de la Ec. 2 re-estimados con la definicion v10 '
                 '(results/driver_coef_v10_20261028/). Origen de cada archivo en sources.csv; cambios de piloto en '
                 'driver_coef_audit.csv.\n')
    verify(dst, A)
    print('->', dst)


def verify(dst, A):
    from src.model_loader import ModelLoader
    broken = [os.path.join(r, n) for r, _, fs in os.walk(dst) for n in fs
              if os.path.islink(os.path.join(r, n)) and not os.path.exists(os.path.join(r, n))]
    assert not broken, f'links rotos: {broken}'
    new, old = ModelLoader(str(dst)), ModelLoader(str(BASE))
    for gp in PAPER_GPS:
        mn, mo = new.load(gp, use_cache=False), old.load(gp, use_cache=False)
        cn, co = mn.laptime_clear, mo.laptime_clear
        assert cn.feature_names_ == co.feature_names_ and float(cn.intercept_) == float(co.intercept_), gp
        d = np.asarray(cn.coef_, float) - np.asarray(co.coef_, float)
        changed = [n for n, x in zip(cn.feature_names_, d) if x != 0]
        assert all(n.startswith("C(Driver,") for n in changed), (gp, changed)
        assert list(mn.laptime_yf.coef_) == list(mo.laptime_yf.coef_), gp
        a = A[A.GP == gp]
        print(f'  {gp:26s} pilotos cambiados {len(changed):2d} | max |dif| {a.dif.abs().max():.4f} s/vuelta')


if __name__ == '__main__':
    main()
