import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

"""
2026-10-14 — Simulador v6 = v5h8 con los coeficientes de piloto de la Ec. 2 recalculados excluyendo las carreras
en que el piloto de referencia (VER) tiene < 15 vueltas limpias (results/driver_coef_study_20261010/
eq2_fallbacks_20261014.py; en la practica Australia 2024, VER con 2 vueltas, que desplazaba en bloque los
coeficientes de esa carrera y por la ventana de 3 carreras previas distorsionaba Miami hasta ~0.5 s/vuelta).

    base:   trained_models/simulation_v5h8_20261012   (v5 + neumaticos opcion3h en los 8 GP; YF v4)
    salida: trained_models/simulation_v6_20261014

Por GP solo cambia laptime_clear.pkl (dummies de piloto de los 20 pilotos; VER = 0); el resto son links a los
archivos reales de la base. Bahrain y Saudi (anteriores a Australia) deben quedar identicos (se verifica).
Los abandonos de rivales de v6 son una opcion del entorno (F1_RIVAL_DNF), no del arbol.

    python scripts/build_simulation_v6_20261014.py
"""
import importlib.util
import pickle
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_regressions import apply_coef_override
from environment.f1_env_all_drivers import ALL_DRIVERS
from build_simulation_v5_20261010 import PAPER_GPS, SEAT, DUMMY, STUDY

BASE = config.MODELS_DIR / 'simulation_v5h8_20261012'
UNCHANGED = {'Bahrain Grand Prix', 'Saudi Arabian Grand Prix'}


def main():
    dst = config.MODELS_DIR / 'simulation_v6_20261014'
    if os.path.exists(dst):
        raise SystemExit(f'{dst} ya existe; no se sobrescribe')
    spec = importlib.util.spec_from_file_location('eq2v6', STUDY / 'eq2_fallbacks_20261014.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    eq2 = mod.Eq2()
    print('carreras excluidas:', sorted(eq2.excluded))
    os.makedirs(dst)
    audit = []
    for gp in PAPER_GPS:
        with open(BASE / gp / 'laptime_clear.pkl', 'rb') as fh:
            fp = pickle.load(fh)
        for name in ALL_DRIVERS:
            drv = name.replace('Driver_', '')
            key = DUMMY.format(drv)
            old = float(fp.coef_[fp.feature_names_.index(key)]) if key in fp.feature_names_ else 0.0
            if drv == 'VER':
                audit.append(dict(GP=gp, driver=drv, coef_v5h8=old, coef_v6=0.0, spec='referencia')); continue
            r = eq2.predict(2024, gp, SEAT.get((gp, drv), drv))
            if not pd.notna(r['pred']):
                raise SystemExit(f'{gp} {drv}: sin prediccion')
            apply_coef_override(fp, key, r['pred'])
            audit.append(dict(GP=gp, driver=drv, coef_v5h8=old, coef_v6=r['pred'], spec=r['spec']))
        gdir = os.path.join(dst, gp); os.makedirs(gdir)
        for f in os.listdir(BASE / gp):
            if f != 'laptime_clear.pkl' or gp in UNCHANGED:
                os.symlink(os.path.relpath(os.path.realpath(BASE / gp / f), gdir), os.path.join(gdir, f))
        if gp not in UNCHANGED:   # Bahrain/Saudi: mismo archivo de la base (la Ec. 2 da lo mismo salvo ~1e-7)
            with open(os.path.join(gdir, 'laptime_clear.pkl'), 'wb') as fh:
                pickle.dump(fp, fh)
    A = pd.DataFrame(audit); A['diff'] = A.coef_v6 - A.coef_v5h8
    A.to_csv(os.path.join(dst, 'driver_coef_audit.csv'), index=False)
    S = A.groupby('GP')['diff'].agg(lambda x: x.abs().max())
    print('max |cambio| por GP (s/vuelta):'); print(S.round(4).to_string())
    assert all(S[g] < 1e-6 for g in UNCHANGED), 'Bahrain/Saudi cambiaron'
    for f in ('tyre_coef_audit.csv',):
        if os.path.exists(BASE / f):
            os.symlink(os.path.relpath(os.path.realpath(BASE / f), dst), os.path.join(dst, f))
    with open(os.path.join(dst, 'README.md'), 'w') as fh:
        fh.write('Arbol 2026-10-14 (scripts/build_simulation_v6_20261014.py): simulation_v5h8_20261012 con los '
                 'coeficientes de piloto de la Ec. 2 recalculados sin las carreras en que VER tiene < 15 vueltas '
                 'limpias (Australia 2024). Auditoria en driver_coef_audit.csv; el resto son links a la base.\n')
    print('->', dst)


if __name__ == '__main__':
    main()
