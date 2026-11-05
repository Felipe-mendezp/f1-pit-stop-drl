"""2026-11-04 — Stints 2021-23 por GP y compuesto y coeficientes conjuntos para tab:tire_coefs con la cadena v10eGQ (sin escribir en results/sim_v10eGQ_20261028).
    F1_TYRELIFE_2021=v10eGQ python results/paper_tables_v10eGQ_20261104/stints_v10eGQ_20261104.py"""
import importlib.util, os, sys
ROOT = os.getcwd(); C_DIR = os.path.join(ROOT, 'results', 'sim_v10eGQ_20261028')
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), C_DIR]
import config, compat; compat.setup_all()
import pandas as pd
import models.regressions as R
import common_v10_20261027 as C
assert os.environ.get('F1_TYRELIFE_2021') == 'v10eGQ'
C.patch_regressions(R)
spec = importlib.util.spec_from_file_location('t7', os.path.join(ROOT, 'results', 'sim_v7_20261015', 'tyre_v7_20261015.py'))
T7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(T7)
P = T7.P
dh = P.hist_laps_A()
GPS8 = ['Bahrain', 'Belgian', 'Dutch', 'Hungarian', 'Miami', 'Saudi Arabian', 'Singapore', 'United States']
H = {g: P.fit_gp(dh, g) for g in P.MAP if g in set(dh.GP)}
print('circuitos', len(H))
ref = pd.read_csv(os.path.join(C_DIR, 'tyre_v10', 'coefs_opcion3h.csv'))
rows = []
for g in GPS8:
    gg = g + ' Grand Prix'; x = ref[ref.GP == gg]
    base = x.offset_vs_ref.iloc[0]
    for _, r in x.iterrows():
        rows.append(dict(gp=g, c=r.compound, stints=int(H[gg]['stints'].get(r.compound, 0)), b1=r.offset_vs_ref - base, b2=r.slope))
T = pd.DataFrame(rows); T.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tab_tire_coefs.csv'), index=False); print(T.round(4).to_string())
