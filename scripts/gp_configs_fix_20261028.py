"""Correccion de las estrategias reales 2024 de los rivales en GP_CONFIGS — 2026-10-28 (docs/AUDITORIA_V10_20261028.md,
raiz del proyecto; decision del autor del 28-oct: corregir).

GP_CONFIGS (scripts/evaluate_all_gps.py) codifica {0: compuesto de partida, vuelta de PitIn: compuesto siguiente}. Contra
data/session_2024_V2.csv y la hoja 'OpenF1 Stints' de data/f1_2024_tire_usage.xlsx (fuente independiente para el
compuesto; Lapchive coincide) hay 7 entradas mal en los GP del paper:
    Bahrain RUS   paradas en 15 y 32 -> 11 y 31 (PitIn de los datos; 32 era la vuelta de PitOut)
    Miami   PIA   3.a tanda HARD -> MEDIUM
    Miami   HAM   parte en MEDIUM con HARD despues -> al reves (parte en HARD)
    Miami   RUS   25 -> 24 (vuelta de PitOut -> PitIn);  ALO 23 -> 22 (idem)
    Miami   MAG   la 'parada' de la vuelta 31 es un paso por el pit lane sin cambio de juego (v10): se quita
    US      ALB   2.a tanda HARD -> MEDIUM (MEDIUM 1-3, MEDIUM 4-33, HARD 34-)
En Belgica GP_CONFIGS esta bien y los datos de FastF1 mal (ALO, RIC, STR, TSU: la ultima tanda es HARD segun OpenF1 y
Lapchive; ver results/compound_check_2024_20261028/). evaluate_all_gps.py no se modifica: GP_CONFIGS se cambia en memoria y
luego se ejecuta el script pedido. Aplica tambien los pilotos v5hd (scripts/gp_drivers_v5hd_20261014.py, idempotente),
asi que REEMPLAZA a ese envoltorio en las cadenas de evaluacion v10.
    python scripts/gp_configs_fix_20261028.py --check        # aplica v5hd + esta correccion y compara con los datos
    python scripts/gp_configs_fix_20261028.py <script.py> <args>
"""
import os, sys, runpy
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

# gp: {piloto: (estrategia publicada, estrategia corregida)}
FIXES = {
    'Bahrain Grand Prix': {'Driver_RUS': ({0: 3, 15: 1, 32: 1}, {0: 3, 11: 1, 31: 1})},
    'Miami Grand Prix': {
        'Driver_PIA': ({0: 3, 27: 2, 40: 2}, {0: 3, 27: 2, 40: 3}),
        'Driver_HAM': ({0: 3, 26: 2}, {0: 2, 26: 3}),
        'Driver_RUS': ({0: 3, 25: 2}, {0: 3, 24: 2}),
        'Driver_ALO': ({0: 2, 23: 3}, {0: 2, 22: 3}),
        'Driver_MAG': ({0: 2, 22: 3, 28: 3, 31: 3}, {0: 2, 22: 3, 28: 3}),
    },
    'United States Grand Prix': {'Driver_ALB': ({0: 3, 3: 2, 33: 2}, {0: 3, 3: 3, 33: 2})},
}


def install():
    import config  # noqa: F401
    import compat; compat.setup_all()
    import evaluate_all_gps as A
    if getattr(A, '_gp_configs_fix_20261028', False):
        return A
    n = 0
    for e in A.GP_CONFIGS:
        for drv, (old, new) in FIXES.get(e['gp_name'], {}).items():
            cur = e['rival_strategies'].get(drv)
            if cur is None:          # el piloto es el agente en esta configuracion
                continue
            assert cur == old, (e['gp_name'], drv, cur)
            e['rival_strategies'][drv] = dict(new)
            n += 1
    A._gp_configs_fix_20261028 = True
    print(f'[gp_configs_fix_20261028] {n} estrategias de rivales corregidas', flush=True)
    return A


def check():
    import pandas as pd
    import gp_drivers_v5hd_20261014 as V
    V.install()
    A = install()
    s = pd.read_csv(os.path.join(_ROOT, 'data', 'session_2024_V2.csv')).sort_values(['GP', 'Driver', 'LapNumber'])
    O = pd.read_excel(os.path.join(_ROOT, 'data', 'f1_2024_tire_usage.xlsx'), 'OpenF1 Stints', header=3)
    O = O[O['Sesión'] == 'Race'].dropna(subset=['Vuelta inicial', 'Vuelta final'])
    O['code'] = O.Piloto.map(lambda x: {'Nico Hulkenberg': 'HUL', 'Guanyu Zhou': 'ZHO', 'Kevin Magnussen': 'MAG'}
                             .get(x, x.split()[-1].upper()[:3]))
    seat = {('Saudi Arabian Grand Prix', 'SAI'): 'BEA', ('Singapore Grand Prix', 'SAR'): 'COL',
            ('United States Grand Prix', 'SAR'): 'COL', ('United States Grand Prix', 'RIC'): 'LAW'}
    import json
    bad = 0
    for e in A.GP_CONFIGS:
        g = e['gp_name']
        if g not in set(s.GP) or g == 'Emilia Romagna Grand Prix':
            continue
        comps = json.load(open(os.path.join(_ROOT, 'trained_models', 'simulation', g, 'config.json')))['compounds']
        num = dict(zip(['HARD', 'MEDIUM', 'SOFT'], [int(c[1]) for c in comps]))
        for drv, st in e['rival_strategies'].items():
            d = seat.get((g, drv[7:]), drv[7:])
            sd = s[(s.GP == g) & (s.Driver == d)].reset_index(drop=True)
            od = O[(O['Gran Premio'] == g) & (O.code == d)]
            comp = {lap: c for r in od.itertuples(index=False) for lap, c in
                    ((l, str(r[12]).upper()) for l in range(int(r[9]), int(r[10]) + 1))}
            if sd.empty or not comp:
                print('  sin datos', g, d); continue
            nx = sd.shift(-1)
            real = {0: num.get(comp.get(1))}
            for i, r in sd.iterrows():
                if r.PitIn == 1 and i + 1 < len(sd) and (nx.Compound[i] != r.Compound or nx.TyreLife[i] < r.TyreLife + 1
                                                         or comp.get(int(r.LapNumber) + 1) != comp.get(int(r.LapNumber))):
                    real[int(r.LapNumber)] = num.get(comp.get(int(r.LapNumber) + 1))
            cfg = {k: v for k, v in st.items() if v != -1}
            if cfg != real:
                bad += 1
                print(f'  DIF {g} {drv}: GP_CONFIGS {st} | datos+OpenF1 {real}')
    print('diferencias tras la correccion:', bad)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    if sys.argv[1] == '--check':
        check(); sys.exit(0)
    import gp_drivers_v5hd_20261014 as _V
    _V.install()
    install()
    target = sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(_ROOT, sys.argv[1])
    sys.argv = [target] + sys.argv[2:]
    runpy.run_path(target, run_name='__main__')
