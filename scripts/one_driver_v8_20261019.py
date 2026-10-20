"""Mono-piloto v8 = v7 + penalizaciones del multi-piloto y accion 0 en la vuelta 0 = compuesto medio — 2026-10-19.

Fija (si no vienen dadas) las variables del entorno mono-piloto (src/environment/f1_env_one_driver.py) para que la
recompensa use los mismos valores que el multi-piloto v8 ('mix', scripts/rival_start_v8_20261018.py):
    F1_ONE_DRIVER_LAP0_ACTION0_MEDIUM=1, F1_ONE_DRIVER_LAP0_PENALTY=-1000   (antes: -150 y compuesto sorteado)
    F1_ONE_DRIVER_PENALTY_CONSECUTIVE_PIT=-200                             (antes -20)
    F1_ONE_DRIVER_PENALTY_EARLY_PIT=-100                                   (antes -15; paradas en las vueltas 1-5)
    F1_ONE_DRIVER_PENALTY_PIT_LAST_LAP=-150                                (antes -30)
La penalizacion por no cambiar de compuesto ya era -1000 (F1_ONE_DRIVER_DSQ_PENALTY). Todo lo demas (simulador
trained_models/simulation_onedriver_v7_20261015, sin filtro, hiperparametros) es igual que v7: se ejecuta
scripts/one_driver_v7_20261015.py con los mismos argumentos.

    python scripts/one_driver_v8_20261019.py {verify|train|screen|fig5|m1} <args de one_driver_v7_20261015.py>
"""
import os, sys, runpy
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V8 = {'F1_ONE_DRIVER_LAP0_ACTION0_MEDIUM': '1', 'F1_ONE_DRIVER_LAP0_PENALTY': '-1000',
      'F1_ONE_DRIVER_PENALTY_CONSECUTIVE_PIT': '-200', 'F1_ONE_DRIVER_PENALTY_EARLY_PIT': '-100',
      'F1_ONE_DRIVER_PENALTY_PIT_LAST_LAP': '-150'}
for _k, _v in V8.items():
    os.environ.setdefault(_k, _v)
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    print('[v8 mono-piloto]', '  '.join(f'{k}={os.environ[k]}' for k in V8), flush=True)
    sys.argv[0] = os.path.join(_ROOT, 'scripts', 'one_driver_v7_20261015.py')
    runpy.run_path(sys.argv[0], run_name='__main__')
