"""2026-10-27 — laptime_clear.pkl (Ec. 1) del simulador v10: igual que results/sim_v7_20261015/laptime_v7_20261015.py
(Ec. 1 del GP con 2021-23, filtro A e IQR; neumaticos de tyre_v10/coefs_opcion3h.csv; pilotos de v6), sin las vueltas de
paso por el pit lane sin parada. Requiere F1_TYRELIFE_2021=v10 (v7 con CONTROL=1). Control: CONTROL=1 (sin parche, neumaticos v7) debe
reproducir el laptime_clear de simulation_v7_20261015 (no escribe nada).
    F1_TYRELIFE_2021=v10 ~/.venvs/f1-drl-v2/bin/python results/sim_v10_20261027/laptime_v10_20261027.py
Salida: results/sim_v10_20261027/artifacts/<GP>/laptime_clear.pkl, laptime_clear_hist.pkl (pilotos historicos,
entrada del logit de adelantamiento) y laptime_coefs_vs_v7.csv
"""
import io, os, sys, contextlib, pickle, importlib.util
ROOT = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts'), HERE]
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
import models.regressions as R
import common_v10_20261027 as C
from reg import FastLinearPredictor
from train_regressions import apply_coef_override
from environment.f1_env_all_drivers import ALL_DRIVERS

assert os.environ.get('F1_TYRELIFE_2021') == ('v7' if os.environ.get('CONTROL') == '1' else 'v10eGQ')  # v10e, 'F1_TYRELIFE_2021=v10 (v7 con CONTROL=1)'
CONTROL = os.environ.get('CONTROL') == '1'
if not CONTROL:
    C.patch_regressions(R)
TYRE = os.path.join(ROOT, 'results', 'sim_v7_20261015', 'tyre_v7') if CONTROL else os.path.join(HERE, 'tyre_v10')
OUT = os.path.join(HERE, 'artifacts')
V6, V7 = config.MODELS_DIR / 'simulation_v6_20261014', config.MODELS_DIR / 'simulation_v7_20261015'
GPS = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
       'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']
DUMMY = "C(Driver, Treatment(reference='VER'))[T.{}]"
spec = importlib.util.spec_from_file_location('p0', os.path.join(ROOT, 'results', 'tyre_coef_study_20260930', 'pooled_20261004.py'))
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
                apply_coef_override(fp, f"C(Compound_Detail, Treatment(reference='{ref}'))[T.{r.compound}]", float(r.offset_vs_ref))
            apply_coef_override(fp, f"TyreLife:C(Compound_Detail)[{r.compound}]", float(r.slope))
        if not CONTROL:   # Ec. 1 con pilotos historicos: entrada del logit de adelantamiento (como v7)
            os.makedirs(os.path.join(OUT, gp), exist_ok=True)
            with open(os.path.join(OUT, gp, 'laptime_clear_hist.pkl'), 'wb') as fh:
                pickle.dump(fp, fh)
        with open(V6 / gp / 'laptime_clear.pkl', 'rb') as fh:
            c6 = dict(zip((v6 := pickle.load(fh)).feature_names_, np.asarray(v6.coef_, float)))
        for name in ALL_DRIVERS:
            drv = name.replace('Driver_', '')
            if drv != 'VER':
                apply_coef_override(fp, DUMMY.format(drv), c6[DUMMY.format(drv)])
        with open(V7 / gp / 'laptime_clear.pkl', 'rb') as fh:
            v7 = pickle.load(fh)
        c7 = dict(zip(v7.feature_names_, np.asarray(v7.coef_, float)))
        cn = dict(zip(fp.feature_names_, np.asarray(fp.coef_, float)))
        d = max(abs(cn.get(k, np.nan) - c7.get(k, np.nan)) for k in set(c7) | set(cn))
        di = abs(float(fp.intercept_) - float(v7.intercept_))
        print(f'{gp}: n={int(lreg.nobs)}, max |dif| coef con v7 = {d:.2e}, intercepto {di:.2e}, '
              f'faltan {sorted(set(c7) ^ set(cn))}')
        for k in sorted(set(c7) | set(cn)):
            rows.append(dict(GP=gp, coef=k, v7=c7.get(k, np.nan), v10=cn.get(k, np.nan)))
        rows.append(dict(GP=gp, coef='Intercept', v7=float(v7.intercept_), v10=float(fp.intercept_)))
        if not CONTROL:
            os.makedirs(os.path.join(OUT, gp), exist_ok=True)
            with open(os.path.join(OUT, gp, 'laptime_clear.pkl'), 'wb') as fh:
                pickle.dump(fp, fh)
    if not CONTROL:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, 'laptime_coefs_vs_v7.csv'), index=False)


if __name__ == '__main__':
    main()
