import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

"""
2026-10-15 — Simulador v7 = v6 con la vida del neumatico de 2021 reescalada por el factor unico 0.716
(data/tyrelife_2021_factors_v7.json; docs/DISENO_TYRELIFE_2021_20261015.md, Fase 2) y el logit condicional (K1)
para el compuesto de los rivales. Archivos nuevos por GP (de results/sim_v7_20261015/artifacts_v7/):
    laptime_clear.pkl         Ec. 1 re-estimada + neumaticos opcion3h v7 + pilotos de v6 (Ec. 2, sin cambio)
    rival_pitstop_logit.pkl   LogitStop (S0) re-estimado
    rival_compound_logit.pkl  K1 (src/models/cond_logit_compound.CondLogitCompoundView)
    overtake_logit.pkl, overtake_threshold.pkl  logit de adelantamiento re-estimado
El resto (YF v4, config.json, incluido el de Miami con SC/VSC 0.5) son links a los archivos reales de v6.

    multi-piloto: trained_models/simulation_v7_20261015/
    mono-piloto:  trained_models/simulation_onedriver_v7_20261015/  (GP -> v7; <GP>_fast_predictor.pkl ->
                  <GP>/laptime_clear.pkl, como simulation_onedriver_opcion3h8_20261012)

    python scripts/build_simulation_v7_20261015.py
"""
import filecmp
import shutil
import pandas as pd

BASE = config.MODELS_DIR / 'simulation_v6_20261014'
ART = config.ROOT_DIR / 'results' / 'sim_v7_20261015' / 'artifacts_v7'
NEW = ('laptime_clear.pkl', 'rival_pitstop_logit.pkl', 'rival_compound_logit.pkl', 'overtake_logit.pkl',
       'overtake_threshold.pkl')
SHARED = ('rival_pitstop_logit.pkl', 'rival_compound_logit.pkl')   # modelos globales (iguales en los 8 GP)
GPS = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
       'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']


def main():
    dst, dst1 = config.MODELS_DIR / 'simulation_v7_20261015', config.MODELS_DIR / 'simulation_onedriver_v7_20261015'
    for d in (dst, dst1):
        if os.path.exists(d):
            raise SystemExit(f'{d} ya existe; no se sobrescribe')
    os.makedirs(dst); os.makedirs(dst1)
    rows = []
    for gp in GPS:
        gdir = dst / gp; os.makedirs(gdir)
        for f in sorted(os.listdir(BASE / gp)):
            if f in SHARED and gp != GPS[0]:
                # mismo modelo global en todos los GP: un solo archivo real (en el primer GP) y links
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
        assert all(os.path.exists(gdir / f) for f in NEW), gp
        os.symlink(os.path.join('..', dst.name, gp), dst1 / gp)
        os.symlink(os.path.join(gp, 'laptime_clear.pkl'), dst1 / f'{gp}_fast_predictor.pkl')
    pd.DataFrame(rows).to_csv(dst / 'sources.csv', index=False)
    for f in ('driver_coef_audit.csv',):
        os.symlink(os.path.relpath(os.path.realpath(BASE / f), dst), dst / f)
    with open(dst / 'README.md', 'w') as fh:
        fh.write('Arbol 2026-10-15 (scripts/build_simulation_v7_20261015.py): v6 con TyreLife 2021 x 0.716 en la Ec. 1, '
                 'neumaticos opcion3h, LogitStop y logit de adelantamiento re-estimados, y K1 para el compuesto de los '
                 'rivales. Origen de cada archivo en sources.csv.\n')
    with open(dst1 / 'README.md', 'w') as fh:
        fh.write('Arbol 2026-10-15 (scripts/build_simulation_v7_20261015.py), mono-piloto: los 8 GP -> '
                 'simulation_v7_20261015.\n')
    print('->', dst, '|', dst1)


if __name__ == '__main__':
    main()
