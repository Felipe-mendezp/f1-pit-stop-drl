"""2026-10-15 — laptime_clear.pkl (Ec. 1 en pista limpia) del simulador v7, por GP del paper.

Misma composicion que v6 (simulation_v6_20261014/<GP>/laptime_clear.pkl), rehecha con la vida del neumatico de
2021 segun F1_TYRELIFE_2021:
  1. Ec. 1 del GP re-estimada con 2021-23, filtro (A) e IQR (como build_simulation_tyrecoef_20261004.py);
  2. coeficientes de compuesto (offset vs referencia y TyreLife x compuesto) de tyre_<tag>/coefs_opcion3h.csv;
  3. coeficientes de piloto de los 20 pilotos del simulador = los de v6 (Ec. 2; con un factor comun a todos los
     compuestos, las regresiones por carrera de la Ec. 2 no cambian: verificado, diferencia maxima 3e-12).
Control: con F1_TYRELIFE_2021=legacy y TYRE_DIR=results/tyre_coef_study_20260930/pooled_hier_20261011 (tau 0.743
de v6) debe reproducir el laptime_clear de v6.
Salida: results/sim_v7_20261015/artifacts_<tag>/<GP>/laptime_clear.pkl (el arbol se arma aparte) y
laptime_clear_hist.pkl (pasos 1-2, con los coeficientes de piloto historicos de la Ec. 1: entrada del logit de
adelantamiento, como el publicado usaba la Ec. 1 historica; no va al arbol).
"""
import io, os, sys, contextlib, pickle, importlib.util
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
import models.regressions as R
from reg import FastLinearPredictor
from train_regressions import apply_coef_override
from environment.f1_env_all_drivers import ALL_DRIVERS

TAG = os.environ.get('F1_TYRELIFE_2021', 'legacy')
HERE = os.path.dirname(os.path.abspath(__file__))
TYRE = os.environ.get('TYRE_DIR', os.path.join(HERE, f'tyre_{TAG}'))
OUT = os.path.join(HERE, f'artifacts_{TAG}')
V6 = config.MODELS_DIR / 'simulation_v6_20261014'
GPS = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
       'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']
DUMMY = "C(Driver, Treatment(reference='VER'))[T.{}]"
spec = importlib.util.spec_from_file_location('p0', os.path.join(ROOT, 'results', 'tyre_coef_study_20260930',
                                                                 'pooled_20261004.py'))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)


def main():
    coefs = pd.read_csv(os.path.join(TYRE, 'coefs_opcion3h.csv'))
    dh = P.hist_laps_A()
    rows = []
    for gp in GPS:
        with contextlib.redirect_stdout(io.StringIO()):
            lreg, _ = R.filtrar_outliers_iqr(dh, gp)
        fp = FastLinearPredictor(lreg, use_float32=True)
        ref = fp.reference_categories_['Compound_Detail']
        cg = coefs[coefs.GP == gp]
        assert (cg.ref == ref).all(), (gp, ref)
        for _, r in cg.iterrows():
            if r.compound != ref:
                apply_coef_override(fp, f"C(Compound_Detail, Treatment(reference='{ref}'))[T.{r.compound}]",
                                    float(r.offset_vs_ref))
            apply_coef_override(fp, f"TyreLife:C(Compound_Detail)[{r.compound}]", float(r.slope))
        os.makedirs(os.path.join(OUT, gp), exist_ok=True)
        with open(os.path.join(OUT, gp, 'laptime_clear_hist.pkl'), 'wb') as fh:   # pilotos historicos (adelantamiento)
            pickle.dump(fp, fh)
        with open(V6 / gp / 'laptime_clear.pkl', 'rb') as fh:
            v6 = pickle.load(fh)
        c6 = dict(zip(v6.feature_names_, np.asarray(v6.coef_, float)))
        for name in ALL_DRIVERS:
            drv = name.replace('Driver_', '')
            if drv != 'VER':
                apply_coef_override(fp, DUMMY.format(drv), c6[DUMMY.format(drv)])
        with open(os.path.join(OUT, gp, 'laptime_clear.pkl'), 'wb') as fh:
            pickle.dump(fp, fh)
        c7 = dict(zip(fp.feature_names_, np.asarray(fp.coef_, float)))
        for k in sorted(set(c6) | set(c7)):
            rows.append(dict(GP=gp, coef=k, v6=c6.get(k, np.nan), new=c7.get(k, np.nan)))
        d = [abs(c7.get(k, np.nan) - c6.get(k, np.nan)) for k in set(c6) | set(c7)]
        print(f'{gp}: n={int(lreg.nobs)}, coef {len(c7)} (v6 {len(c6)}), max |dif| con v6 = {np.nanmax(d):.2e}, '
              f'faltan {sorted(set(c6) ^ set(c7))}')
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'laptime_coefs_vs_v6.csv'), index=False)


if __name__ == '__main__':
    main()
