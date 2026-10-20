import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

"""
2026-10-10 — Simulador v5 (= v4 opcion3 + coeficientes de piloto corregidos por la Ec. 2; etapa B del
camino 2). Estudio en results/driver_coef_study_20261010/.

    base:   trained_models/simulation_yfv4_opcion3_20261007   (opcion3 + YF v4)
    salida: trained_models/simulation_v5_20261010

Por GP solo cambia laptime_clear.pkl: el coeficiente C(Driver)[T.X] de cada uno de los 20 pilotos del
simulador (agente y rivales; VER sigue siendo la referencia = 0) se reemplaza por la prediccion de la
Ec. 2 porcentual para ese piloto en ese GP de 2024, ajustada solo con carreras anteriores al GP
(2021-23 y las de 2024 previas), con respaldos sin clasificacion o con menos carreras previas
(eq2_fallbacks_20261010.py). El resto de los coeficientes (neumaticos, anio, intercepto) no cambia, y
los demas archivos son links simbolicos a la base. DRL y QMIP leen el mismo laptime_clear.

Asientos ocupados por un reemplazante en 2024 (el simulador conserva el nombre del titular, como en
config.json): se usan los datos del piloto que corrio ese auto (SEAT).

    python scripts/build_simulation_v5_20261010.py
"""

import argparse
import importlib.util
import pickle

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_regressions import apply_coef_override
from environment.f1_env_all_drivers import ALL_DRIVERS

STUDY = config.ROOT_DIR / 'results' / 'driver_coef_study_20261010'
BASE = config.MODELS_DIR / 'simulation_yfv4_opcion3_20261007'
PAPER_GPS = ['Bahrain Grand Prix', 'Belgian Grand Prix', 'Dutch Grand Prix', 'Hungarian Grand Prix',
             'Miami Grand Prix', 'Saudi Arabian Grand Prix', 'Singapore Grand Prix',
             'United States Grand Prix']
SEAT = {('Saudi Arabian Grand Prix', 'SAI'): 'BEA',
        ('Singapore Grand Prix', 'SAR'): 'COL',
        ('United States Grand Prix', 'SAR'): 'COL',
        ('United States Grand Prix', 'RIC'): 'LAW'}
DUMMY = "C(Driver, Treatment(reference='VER'))[T.{}]"


def load_eq2():
    spec = importlib.util.spec_from_file_location('eq2_fallbacks', STUDY / 'eq2_fallbacks_20261010.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.Eq2()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(config.MODELS_DIR / 'simulation_v5_20261010'))
    a = ap.parse_args()
    dst = a.out
    if os.path.basename(os.path.normpath(dst)) == 'simulation' or os.path.basename(dst).startswith('rl_agents'):
        raise SystemExit('refusing to write into a shipped tree')
    if os.path.exists(dst):
        raise SystemExit(f'{dst} already exists; remove it or pass --out')
    eq2 = load_eq2()
    truth = pd.read_csv(STUDY / 'race_coefs.csv').query('Year == 2024').set_index(['GP', 'Driver'])

    os.makedirs(dst)
    audit = []
    for gp in PAPER_GPS:
        with open(BASE / gp / 'laptime_clear.pkl', 'rb') as fh:
            fp = pickle.load(fh)
        for name in ALL_DRIVERS:
            drv = name.replace('Driver_', '')
            src = SEAT.get((gp, drv), drv)
            r = eq2.predict(2024, gp, src)
            if not pd.notna(r['pred']):
                raise SystemExit(f'{gp} {drv}: sin prediccion de la Ec. 2')
            key = DUMMY.format(drv)
            had = key in fp.feature_names_
            old = float(fp.coef_[fp.feature_names_.index(key)]) if had else 0.0
            if drv != 'VER':
                # piloto sin datos del GP en 2021-23: el arbol base no tiene su dummy (corria al ritmo de VER)
                apply_coef_override(fp, key, r['pred'])
            real = truth.coef.get((gp, src), float('nan'))
            audit.append(dict(GP=gp, driver=drv, data_driver=src, dummy_en_base=had or drv == 'VER', coef_hist=old, coef_eq2=r['pred'],
                              coef_real_2024=real, spec=r['spec'], n_train=r['n_train'],
                              **{k: r.get(k) for k in ('P1_pct', 'P2_pct', 'P3_pct', 'FP_pct', 'Q_pct', 'pole')}))
        gdir = os.path.join(dst, gp); os.makedirs(gdir)
        for f in os.listdir(BASE / gp):
            if f != 'laptime_clear.pkl':
                os.symlink(os.path.relpath(BASE / gp / f, gdir), os.path.join(gdir, f))
        with open(os.path.join(gdir, 'laptime_clear.pkl'), 'wb') as fh:
            pickle.dump(fp, fh)
        print(f'{gp}: listo')
    A = pd.DataFrame(audit)
    A.to_csv(os.path.join(dst, 'driver_coef_audit.csv'), index=False)
    with open(os.path.join(dst, 'README.md'), 'w') as fh:
        fh.write('Arbol 2026-10-10 (scripts/build_simulation_v5_20261010.py): base '
                 'simulation_yfv4_opcion3_20261007 con los coeficientes de piloto de laptime_clear.pkl '
                 'reemplazados por la Ec. 2 porcentual (results/driver_coef_study_20261010/). '
                 'Auditoria antes/despues en driver_coef_audit.csv; el resto son links a la base.\n')
    print('->', dst)


if __name__ == '__main__':
    main()
