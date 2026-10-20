import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

"""
2026-10-04 — Arboles de simulacion con coeficientes de neumatico alternativos
(estudio results/tyre_coef_study_20260930/pooled_20261004.py).

  --option opcion2 : modelo conjunto libre + efecto de celda (sin monotonia)
  --option opcion3 : modelo conjunto monotono, paso C4->C5 = paso C3->C4

Salida NUEVA (nada se borra ni se sobrescribe):
    trained_models/simulation_tyre_<opcion>_20261004/<GP>/
Para los 8 GP del paper (sin Emilia Romagna, filtro A) se reemplaza SOLO laptime_clear.pkl:
  Ec. 1 re-estimada con 2021-23 + filtro (A) (ediciones con >10 % WET/INT fuera) y los
  coeficientes de compuesto (fijo vs referencia y TyreLife x compuesto) de la opcion.
Todos los demas archivos del GP son enlaces simbolicos relativos al simulador publicado
(trained_models/simulation): YF 2024, logits de adelantamiento y rivales, config.json.
QMIP lee los mismos coeficientes desde laptime_clear (utils_qmip), asi que DRL y QMIP quedan
calibrados igual al usar F1_SIMULATION_DIR=<arbol nuevo>.
"""

import argparse
import importlib.util
import io
import contextlib
import pickle

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_regressions import apply_coef_override
from reg import FastLinearPredictor
import models.regressions as R

STUDY = config.ROOT_DIR / 'results' / 'tyre_coef_study_20260930'
SRC = config.MODELS_DIR / 'simulation'
PAPER_GPS = ['Bahrain Grand Prix', 'Belgian Grand Prix', 'Dutch Grand Prix', 'Hungarian Grand Prix',
             'Miami Grand Prix', 'Saudi Arabian Grand Prix', 'Singapore Grand Prix',
             'United States Grand Prix']


def load_study():
    spec = importlib.util.spec_from_file_location('pooled_20261004', STUDY / 'pooled_20261004.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--option', required=True, choices=['opcion2', 'opcion3'])
    ap.add_argument('--out', default=None, help='default: trained_models/simulation_tyre_<opcion>_20261004')
    a = ap.parse_args()
    dst = a.out or str(config.MODELS_DIR / f'simulation_tyre_{a.option}_20261004')
    if os.path.abspath(dst) == os.path.abspath(SRC):
        raise SystemExit('refusing to write into the shipped simulator')
    if os.path.exists(dst):
        raise SystemExit(f'{dst} already exists; remove it or pass --out')
    coefs = pd.read_csv(STUDY / 'pooled_20261004' / f'coefs_{a.option}.csv')
    dh = load_study().hist_laps_A()

    os.makedirs(dst)
    audit = []
    for gp in PAPER_GPS:
        with contextlib.redirect_stdout(io.StringIO()):
            lreg, _ = R.filtrar_outliers_iqr(dh, gp)
        P = lreg.params
        ref = P.index[P.index.str.contains('Compound_Detail, Treatment')][0].split("'")[1]
        cg = coefs[coefs.GP == gp]
        assert (cg.ref == ref).all(), f'{gp}: reference mismatch {ref} vs {set(cg.ref)}'
        fp = FastLinearPredictor(lreg, use_float32=True)
        for _, r in cg.iterrows():
            if r.compound != ref:
                apply_coef_override(fp, f"C(Compound_Detail, Treatment(reference='{ref}'))[T.{r.compound}]",
                                    float(r.offset_vs_ref))
            apply_coef_override(fp, f"TyreLife:C(Compound_Detail)[{r.compound}]", float(r.slope))
            audit.append(dict(GP=gp, ref=ref, compound=r.compound, offset_vs_ref=r.offset_vs_ref, slope=r.slope))
        gdir = os.path.join(dst, gp); os.makedirs(gdir)
        for f in os.listdir(SRC / gp):
            if f == 'laptime_clear.pkl':
                continue
            os.symlink(os.path.relpath(SRC / gp / f, gdir), os.path.join(gdir, f))
        with open(os.path.join(gdir, 'laptime_clear.pkl'), 'wb') as fh:
            pickle.dump(fp, fh)
        print(f'{gp}: ref {ref}, n={int(lreg.nobs)}', cg[['compound', 'offset_vs_ref', 'slope']].round(3).values.tolist())
    pd.DataFrame(audit).to_csv(os.path.join(dst, 'tyre_coef_audit.csv'), index=False)
    with open(os.path.join(dst, 'README.md'), 'w') as fh:
        fh.write(f'Built by scripts/build_simulation_tyrecoef_20261004.py --option {a.option}.\n'
                 'Only <GP>/laptime_clear.pkl is new; everything else links to trained_models/simulation.\n')
    print('->', dst)


if __name__ == '__main__':
    main()
