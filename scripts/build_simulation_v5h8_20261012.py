import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import compat; compat.setup_all()

"""
2026-10-12 — Simulador v5h8 = v5 con los neumaticos de la "opcion 3 jerarquica" (opcion3h) en los 8 GP
(scripts/build_simulation_v5h_20261011.py solo los cambio en Saudi). Mismo cambio por GP que ese script
(offset por compuesto y TyreLife x compuesto de coefs_opcion3h.csv; register verifica que la base sea
opcion3 2021-23). Saudi enlaza al GP de simulation_v5h_20261011 (identico), asi los agentes de Saudi
de rl_agents_v5h_20261011 valen para este arbol.

    multi-piloto: trained_models/simulation_v5h8_20261012/
    mono-piloto:  trained_models/simulation_onedriver_opcion3h8_20261012/  (GP -> v5h8;
                  <GP>_fast_predictor.pkl -> <GP>/laptime_clear.pkl; el piloto VER tiene coef. de piloto 0)

    python scripts/build_simulation_v5h8_20261012.py
"""
import pickle
import pandas as pd
import evaluate_world2024_20261010 as W
from build_simulation_v5h_20261011 import MD, V5, NEWC, GPS8

V5H = MD / 'simulation_v5h_20261011'
SAUDI = 'Saudi Arabian Grand Prix'


def main():
    dst = MD / 'simulation_v5h8_20261012'; dst1 = MD / 'simulation_onedriver_opcion3h8_20261012'
    for d in (dst, dst1):
        if os.path.exists(d):
            raise SystemExit(f'{d} ya existe')
    os.makedirs(dst); os.makedirs(dst1)
    W.WORLD = str(NEWC)
    audit = []
    for gp in GPS8:
        with open(V5 / gp / 'laptime_clear.pkl', 'rb') as fh:
            fp = pickle.load(fh)
        changed = W.register(gp, fp, '2024')
        audit += [dict(GP=gp, coef=n, opcion3=b, opcion3h=w) for n, b, w in changed]
        if gp == SAUDI:
            with open(V5H / gp / 'laptime_clear.pkl', 'rb') as fh:
                assert (pickle.load(fh).coef_ == W._SPEC[gp]['world']).all(), 'Saudi v5h distinto'
            os.symlink(os.path.join('..', V5H.name, gp), dst / gp)
        else:
            fp.coef_[:] = W._SPEC[gp]['world']
            gdir = dst / gp; os.makedirs(gdir)
            for f in os.listdir(V5 / gp):
                if f != 'laptime_clear.pkl':
                    os.symlink(os.readlink(V5 / gp / f), gdir / f)   # links relativos de v5: misma profundidad
            with open(gdir / 'laptime_clear.pkl', 'wb') as fh:
                pickle.dump(fp, fh)
        os.symlink(os.path.join('..', dst.name, gp), dst1 / gp)
        os.symlink(os.path.join(gp, 'laptime_clear.pkl'), dst1 / f'{gp}_fast_predictor.pkl')
        print(f'{gp}: {len(changed)} coeficientes cambiados')
    pd.DataFrame(audit).to_csv(dst / 'tyre_coef_audit.csv', index=False)
    for d, txt in [(dst, 'multi-piloto: v5 + neumaticos opcion3h en los 8 GP (Saudi -> simulation_v5h_20261011).'),
                   (dst1, 'mono-piloto: los 8 GP -> simulation_v5h8_20261012.')]:
        with open(d / 'README.md', 'w') as fh:
            fh.write(f'Arbol 2026-10-12 (scripts/build_simulation_v5h8_20261012.py), {txt}\n')
    print('->', dst, '|', dst1)


if __name__ == '__main__':
    main()
