"""2026-10-27 — Logit de adelantamiento del simulador v10: results/sim_v7_20261015/overtake_v7_20261015.py (procedimiento
publicado, src/models/logit_overtakes_pit.py) con el factor de TyreLife 2021 de v10 y los tiempos predichos por la Ec. 1 v10
con pilotos historicos (artifacts/<GP>/laptime_clear_hist.pkl). Las covariables de pit del rival siguen siendo
PitIn/PitOut (solo vueltas verdes; decision del autor, item B.4 de docs/PROPUESTA_V10_20261027.md).
    F1_TYRELIFE_2021=v10 ~/.venvs/f1-drl-v2/bin/python results/sim_v10_20261027/overtake_v10_20261027.py
Salida: results/sim_v10_20261027/overtake_v10/ (params.csv, prob_table.csv) y artifacts/<GP>/overtake_{logit,threshold}.pkl
"""
import os
ROOT = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__))
assert os.environ.get('F1_TYRELIFE_2021') == 'v10eGQ'  # v10e, 'F1_TYRELIFE_2021=v10'
os.environ['PRED'] = 'v7'; os.environ['GPS'] = 'paper'
# 2026-10-28 (noche): antes este script no importaba los ganchos de lectura, y el logit de adelantamiento se estimaba con
# los compuestos sin corregir y la edad de FastF1 (aunque con la Ec. 1 de esta carpeta). Ahora instala data_fix_v10d
# (compuestos) y age_fix_v10e (TL_GQ) via common_v10_20261027, igual que los demas pasos de la cadena.
import sys; sys.path.insert(0, HERE); sys.path[:0] = [ROOT, os.path.join(ROOT, 'src')]
import common_v10_20261027  # noqa: F401
path = os.path.join(ROOT, 'results', 'sim_v7_20261015', 'overtake_v7_20261015.py')
src = open(path).read()
old_out = "OUT = os.path.join(HERE, f'overtake_{TAG}_{PRED}_{GPS}'); os.makedirs(OUT, exist_ok=True)"
old_art = "ART = os.path.join(HERE, f'artifacts_{TAG}')"
assert src.count(old_out) == 1 and src.count(old_art) == 1
src = src.replace(old_out, f"OUT = {os.path.join(HERE, 'overtake_v10')!r}; os.makedirs(OUT, exist_ok=True)")
src = src.replace(old_art, f"ART = {os.path.join(HERE, 'artifacts')!r}")
exec(compile(src, path, 'exec'), {'__name__': '__main__', '__file__': path})
