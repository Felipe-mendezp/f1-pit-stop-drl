import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import compat; compat.setup_all()

"""
2026-10-11 — Simulador v5h = v5 con los coeficientes de neumatico de la "opcion 3 jerarquica"
(results/tyre_coef_study_20260930/pooled_hier_20261011.py: nivel de degradacion por GP como efecto
aleatorio, tau = 0.743 por ML) en los GP indicados (por defecto solo Saudi, el unico que cambia de forma
material; en los otros 7 GP las pendientes cambian <= 0.0008 s/vuelta^2).

    multi-piloto: trained_models/simulation_v5h_20261011/
        GP en --gps: laptime_clear.pkl = el de v5 con offset por compuesto y TyreLife x compuesto de
                     opcion3h (mismo cambio que scripts/evaluate_world2024_20261010.register, que verifica
                     que el de partida sea opcion3 2021-23); el resto de archivos, links a v5.
        demas GP:    link al directorio de v5.
    mono-piloto:  trained_models/simulation_onedriver_opcion3h_20261011/
        GP en --gps: link al GP de v5h; demas: link al GP de simulation_yfv4_opcion3_20261007 (igual que
        simulation_onedriver_opcion3_20261011). <GP>_fast_predictor.pkl -> <GP>/laptime_clear.pkl.

    python scripts/build_simulation_v5h_20261011.py [--gps 'Saudi Arabian Grand Prix' ...]
"""
import argparse, pickle
import pandas as pd
import evaluate_world2024_20261010 as W

MD = config.MODELS_DIR
V5 = MD / 'simulation_v5_20261010'
OP3 = MD / 'simulation_yfv4_opcion3_20261007'
NEWC = config.ROOT_DIR / 'results' / 'tyre_coef_study_20260930' / 'pooled_hier_20261011' / 'coefs_opcion3h.csv'
GPS8 = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
        'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gps', nargs='+', default=['Saudi Arabian Grand Prix'])
    a = ap.parse_args()
    dst = MD / 'simulation_v5h_20261011'; dst1 = MD / 'simulation_onedriver_opcion3h_20261011'
    for d in (dst, dst1):
        if os.path.exists(d):
            raise SystemExit(f'{d} ya existe')
    os.makedirs(dst); os.makedirs(dst1)
    W.WORLD = str(NEWC)                       # "mundo" = opcion3h; register controla que la base sea opcion3
    audit = []
    for gp in GPS8:
        if gp in a.gps:
            with open(V5 / gp / 'laptime_clear.pkl', 'rb') as fh:
                fp = pickle.load(fh)
            changed = W.register(gp, fp, '2024')
            fp.coef_[:] = W._SPEC[gp]['world']
            gdir = dst / gp; os.makedirs(gdir)
            for f in os.listdir(V5 / gp):
                if f != 'laptime_clear.pkl':
                    os.symlink(os.readlink(V5 / gp / f), gdir / f)   # links relativos de v5: misma profundidad
            with open(gdir / 'laptime_clear.pkl', 'wb') as fh:
                pickle.dump(fp, fh)
            audit += [dict(GP=gp, coef=n, opcion3=b, opcion3h=w) for n, b, w in changed]
            os.symlink(os.path.join('..', dst.name, gp), dst1 / gp)
            print(f'{gp}: {len(changed)} coeficientes cambiados')
        else:
            os.symlink(os.path.join('..', V5.name, gp), dst / gp)
            os.symlink(os.path.join('..', OP3.name, gp), dst1 / gp)
        os.symlink(os.path.join(gp, 'laptime_clear.pkl'), dst1 / f'{gp}_fast_predictor.pkl')
    pd.DataFrame(audit).to_csv(dst / 'tyre_coef_audit.csv', index=False)
    for d, txt in [(dst, 'multi-piloto: v5 + neumaticos opcion3h en ' + ', '.join(a.gps) + '; demas GP = v5.'),
                   (dst1, 'mono-piloto: ' + ', '.join(a.gps) + ' -> simulation_v5h_20261011; demas GP -> simulation_yfv4_opcion3_20261007.')]:
        with open(d / 'README.md', 'w') as fh:
            fh.write(f'Arbol 2026-10-11 (scripts/build_simulation_v5h_20261011.py), {txt}\n')
    print('->', dst, '|', dst1)


if __name__ == '__main__':
    main()
