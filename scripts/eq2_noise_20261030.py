"""Ruido de la Ec. 2 en el coeficiente de piloto (2026-10-30).

Envoltorio de la cadena v10f (va primero). En cada carrera simulada, el coeficiente de cada piloto es el predicho por la
Ec. 2 mas e ~ N(0, 0.28^2) s/vuelta, fijo durante la carrera; el ruido por vuelta de la Ec. 1 no cambia
(F1_RACE_EFFECT_SD=0.28, F1_RACE_EFFECT_KEEP_STD=1). 0.28 = sd del error de prediccion de la Ec. 2 en los 8 GP de 2024,
neta del error de muestreo del coeficiente realizado (results/dispersion_hyp_20261030/README.md §6-7).
    python scripts/eq2_noise_20261030.py scripts/overtake_v10f_20261029.py ...
"""
import os, sys, runpy
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SD = '0.28'
os.environ['F1_RACE_EFFECT_SD'] = SD
os.environ['F1_RACE_EFFECT_KEEP_STD'] = '1'
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    print('[eq2 noise] F1_RACE_EFFECT_SD =', SD, '| F1_RACE_EFFECT_KEEP_STD = 1', flush=True)
    target = sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(_ROOT, sys.argv[1])
    sys.argv = [target] + sys.argv[2:]
    runpy.run_path(target, run_name='__main__')
