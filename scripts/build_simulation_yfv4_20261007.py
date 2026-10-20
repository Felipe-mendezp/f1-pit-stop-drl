import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

"""
2026-10-07 — Árboles de simulación con el modelo YF v4 (data/yf_model_v4_20261007.csv, generado por
scripts/fit_yf_model_v4_20261007.py). Uno por calibración de neumáticos:

    control -> trained_models/simulation_yfv4_control_20261007/   (base: simulation)
    opcion2 -> trained_models/simulation_yfv4_opcion2_20261007/   (base: simulation_tyre_opcion2_20261004)
    opcion3 -> trained_models/simulation_yfv4_opcion3_20261007/   (base: simulation_tyre_opcion3_20261004)

Cada árbol es un overlay: por GP, laptime_yf_fast_v2.pkl y laptime_yf.pkl son archivos NUEVOS; el resto
apunta con links simbólicos relativos al mismo archivo que usa la base (laptime_clear de la calibración,
logits, config). Nada existente se modifica. Solo los 8 GP del paper (sin Emilia Romagna).

    python scripts/build_simulation_yfv4_20261007.py [--bases control opcion2 opcion3]
"""

import argparse
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_yf_models import YF_TERMS, build_yf_predictor

CSV = config.DATA_DIR / 'yf_model_v4_20261007.csv'
BASES = {'control': 'simulation',
         'opcion2': 'simulation_tyre_opcion2_20261004',
         'opcion3': 'simulation_tyre_opcion3_20261004'}
GPS = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
       'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']
NEW_FILES = ('laptime_yf_fast_v2.pkl', 'laptime_yf.pkl')


def build(base_key, coefs):
    src = config.MODELS_DIR / BASES[base_key]
    dst = config.MODELS_DIR / f'simulation_yfv4_{base_key}_20261007'
    assert src.is_dir(), f'no existe la base {src}'
    if dst.exists():
        raise SystemExit(f'{dst} ya existe; no se sobrescribe. Bórralo tú si quieres regenerarlo.')
    audit = []
    for gp in GPS:
        os.makedirs(dst / gp)
        for name in sorted(os.listdir(src / gp)):
            if name in NEW_FILES:
                continue
            # destino real del archivo de la base, como link relativo desde el árbol nuevo
            real = os.path.realpath(src / gp / name)
            os.symlink(os.path.relpath(real, dst / gp), dst / gp / name)
        with open(os.path.realpath(src / gp / 'laptime_yf_fast_v2.pkl'), 'rb') as f:
            old = dict(zip(YF_TERMS, [float(c) for c in pickle.load(f).coef_]))
        new = {t: float(coefs.loc[gp, t]) for t in YF_TERMS}
        sm_res, fast = build_yf_predictor(new)
        for name, obj in (('laptime_yf_fast_v2.pkl', fast), ('laptime_yf.pkl', sm_res)):
            with open(dst / gp / name, 'wb') as f:
                pickle.dump(obj, f)
        for t in YF_TERMS:
            audit.append({'GP': gp, 'term': t, 'old': round(old[t], 4), 'new_v4': round(new[t], 4)})
    pd.DataFrame(audit).to_csv(dst / 'yf_v4_coef_audit.csv', index=False)
    with open(dst / 'README.md', 'w') as f:
        f.write(f'Árbol 2026-10-07 (scripts/build_simulation_yfv4_20261007.py): base {BASES[base_key]} con\n'
                'el modelo YF v4 (data/yf_model_v4_20261007.csv; estudio en\n'
                'results/yf_pitloss_study_20261007/). Nuevos: laptime_yf_fast_v2.pkl y laptime_yf.pkl;\n'
                'el resto son links a la base. Auditoría old vs new en yf_v4_coef_audit.csv.\n')
    return dst


def verify(dst, coefs):
    """Relee del disco con el cargador del simulador y comprueba coeficientes y links."""
    from src.model_loader import ModelLoader
    broken = [os.path.join(r, n) for r, _, fs in os.walk(dst) for n in fs
              if os.path.islink(os.path.join(r, n)) and not os.path.exists(os.path.join(r, n))]
    assert not broken, f'links rotos: {broken}'
    loader = ModelLoader(str(dst))
    for gp in GPS:
        m = loader.load(gp, use_cache=False).laptime_yf
        got = dict(zip(YF_TERMS, [float(c) for c in m.coef_]))
        assert all(abs(got[t] - coefs.loc[gp, t]) < 1e-3 for t in YF_TERMS), (gp, got)
        sc = m.predict_single({'SC': 1.0})
        vsc_pit = m.predict_single({'VSC': 1.0, 'PitIn': 1.0}) - m.predict_single({'VSC': 1.0})
        print(f'  {gp:26s} SC {sc:7.2f}  VSC {got["VSC"]:7.2f}  PitIn {vsc_pit:5.2f}  '
              f'PitOut {got["PitOut"]:5.2f}  (pérdida {got["PitIn"] + got["PitOut"]:5.2f})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bases', nargs='+', default=list(BASES), choices=list(BASES))
    a = ap.parse_args()
    coefs = pd.read_csv(CSV).set_index('GP')
    missing = [gp for gp in GPS if gp not in coefs.index]
    assert not missing, f'GPs sin coeficientes en {CSV}: {missing}'
    assert np.isfinite(coefs.loc[GPS, YF_TERMS].values).all()
    for b in a.bases:
        dst = build(b, coefs)
        print(f'{dst}:')
        verify(dst, coefs)


if __name__ == '__main__':
    main()
