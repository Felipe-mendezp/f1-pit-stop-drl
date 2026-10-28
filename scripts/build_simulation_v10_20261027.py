import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

"""
2026-10-27 — Simulador v10 (docs/PROPUESTA_V10_20261027.md, raiz del proyecto) = v7 con las paradas reales en vez de
PitIn/PitOut de FastF1 (pasos por el pit lane) y el factor de TyreLife 2021 re-estimado (0.740,
data/tyrelife_2021_factors_v10.json). Archivos nuevos por GP, desde results/sim_v10_20261027/:
    laptime_clear.pkl            Ec. 1 sin pasos por el pit lane + neumaticos tyre_v10 (laptime_v10_20261027.py)
    rival_pitstop_logit.pkl      LogitStop por evento con paradas reales (rivals_v10_20261027.py)
    rival_compound_logit.pkl     K1 con paradas reales (rivals_v10_20261027.py)
    overtake_logit.pkl, overtake_threshold.pkl   adelantamiento re-estimado con el factor v10 (overtake_v10_20261027.py)
    laptime_yf_fast_v2.pkl, laptime_yf.pkl   modelo SC/VSC sin pasos por el pit lane (yf_model_v10_20261027.csv)
El resto (config.json, laptime_yf_fast.pkl) son links a los archivos reales de v7. Los dos logits de
rivales son globales: un archivo real en Bahrain y links desde los demas GP. Arbol mono-piloto: los 8 GP -> v10.
Nada existente se modifica.
    python scripts/build_simulation_v10_20261027.py
"""
import filecmp
import pickle
import shutil
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_yf_models import YF_TERMS, build_yf_predictor

BASE = config.MODELS_DIR / 'simulation_v7_20261015'
ART = config.ROOT_DIR / 'results' / 'sim_v10_20261027' / 'artifacts'
YF_CSV = config.ROOT_DIR / 'results' / 'sim_v10_20261027' / 'yf_model_v10_20261027.csv'
NEW = ('laptime_clear.pkl', 'rival_pitstop_logit.pkl', 'rival_compound_logit.pkl', 'overtake_logit.pkl',
       'overtake_threshold.pkl')
YF_NEW = ('laptime_yf_fast_v2.pkl', 'laptime_yf.pkl')
SHARED = ('rival_pitstop_logit.pkl', 'rival_compound_logit.pkl')
GPS = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
       'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']


def main():
    dst = config.MODELS_DIR / 'simulation_v10_20261027'
    dst1 = config.MODELS_DIR / 'simulation_onedriver_v10_20261027'
    for d in (dst, dst1):
        if os.path.exists(d):
            raise SystemExit(f'{d} ya existe; no se sobrescribe')
    yf = pd.read_csv(YF_CSV).set_index('GP')
    os.makedirs(dst); os.makedirs(dst1)
    rows = []
    for gp in GPS:
        gdir = dst / gp; os.makedirs(gdir)
        for f in sorted(os.listdir(BASE / gp)):
            if f in YF_NEW:
                continue
            if f in SHARED and gp != GPS[0]:
                assert filecmp.cmp(ART / gp / f, ART / GPS[0] / f, shallow=False), (gp, f)
                os.symlink(os.path.join('..', GPS[0], f), gdir / f)
                rows.append(dict(GP=gp, file=f, source=f'{dst.name}/{GPS[0]}/{f}'))
            elif f in NEW:
                shutil.copy2(ART / gp / f, gdir / f)
                rows.append(dict(GP=gp, file=f, source=str((ART / gp / f).relative_to(config.ROOT_DIR))))
            else:
                real = os.path.realpath(BASE / gp / f)
                os.symlink(os.path.relpath(real, gdir), gdir / f)
                rows.append(dict(GP=gp, file=f, source=os.path.relpath(real, config.ROOT_DIR)))
        sm_res, fast = build_yf_predictor({t: float(yf.loc[gp, t]) for t in YF_TERMS})
        for name, obj in (('laptime_yf_fast_v2.pkl', fast), ('laptime_yf.pkl', sm_res)):
            with open(gdir / name, 'wb') as fh:
                pickle.dump(obj, fh)
            rows.append(dict(GP=gp, file=name, source=str(YF_CSV.relative_to(config.ROOT_DIR))))
        assert all(os.path.exists(gdir / f) for f in NEW + YF_NEW), gp
        os.symlink(os.path.join('..', dst.name, gp), dst1 / gp)
        os.symlink(os.path.join(gp, 'laptime_clear.pkl'), dst1 / f'{gp}_fast_predictor.pkl')
    pd.DataFrame(rows).to_csv(dst / 'sources.csv', index=False)
    for f in ('driver_coef_audit.csv',):
        os.symlink(os.path.relpath(os.path.realpath(BASE / f), dst), dst / f)
    with open(dst / 'README.md', 'w') as fh:
        fh.write('Arbol 2026-10-27 (scripts/build_simulation_v10_20261027.py): v7 con paradas reales en lugar de PitIn de '
                 'FastF1 (Ec. 1 y modelo SC/VSC sin pasos por el pit lane, LogitStop por evento, K1 con paradas reales), '
                 'TyreLife 2021 x 0.740 y adelantamiento re-estimado. Origen de cada archivo en sources.csv.\n')
    with open(dst1 / 'README.md', 'w') as fh:
        fh.write('Arbol 2026-10-27 (scripts/build_simulation_v10_20261027.py), mono-piloto: los 8 GP -> '
                 'simulation_v10_20261027.\n')
    verify(dst, yf)
    print('->', dst, '|', dst1)


def verify(dst, yf):
    from src.model_loader import ModelLoader
    broken = [os.path.join(r, n) for r, _, fs in os.walk(dst) for n in fs
              if os.path.islink(os.path.join(r, n)) and not os.path.exists(os.path.join(r, n))]
    assert not broken, f'links rotos: {broken}'
    loader = ModelLoader(str(dst))
    for gp in GPS:
        m = loader.load(gp, use_cache=False)
        got = dict(zip(YF_TERMS, [float(c) for c in m.laptime_yf.coef_]))
        assert all(abs(got[t] - yf.loc[gp, t]) < 1e-3 for t in YF_TERMS), (gp, got)
        with open(ART / gp / 'laptime_clear.pkl', 'rb') as fh:
            lc = pickle.load(fh)
        assert list(m.laptime_clear.coef_) == list(lc.coef_), gp
        print(f'  {gp:26s} perdida de parada SC/VSC {got["PitIn"] + got["PitOut"]:5.2f} s | SC {got["SC"]:7.2f}')


if __name__ == '__main__':
    main()
