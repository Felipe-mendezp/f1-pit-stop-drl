import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
2026-10-24 — Apoyo a la decision con la historia real del piloto (pedido del autor). Envoltorio de
scripts/ds_rollouts_belief_20261012.py (mismos argumentos) que fuerza las acciones del agente SOLO en el prefijo
comun (vueltas 0..n-1); las proyecciones desde la vuelta n siguen con la politica, igual que el barrido v9r2.
    F1_DS_FORCE=start : en la vuelta 0 el agente larga con el compuesto real 2024 del piloto; despues decide la politica.
    F1_DS_FORCE=hist  : largada y paradas reales 2024 del piloto (vuelta de entrada L -> compuesto de la tanda
                        siguiente, misma convencion que las estrategias reales de los rivales en GP_CONFIGS); en las
                        demas vueltas del prefijo, no parar.
Estrategia real desde data/session_2024_V2.csv (nueva tanda = baja TyreLife o cambia Compound_Detail). La edad del juego
de largada no se fuerza (el entorno toma el juego menos gastado del compuesto). Fuera del prefijo no cambia nada: la
sustitucion de la accion solo esta activa mientras corre run_prefix_multi.
    python scripts/ds_rollouts_realhist_20261024.py <args de ds_rollouts_belief>
    python scripts/ds_rollouts_realhist_20261024.py --run scripts/plot_decision_stints.py <args>   # Fig. 9
"""

import pandas as pd

import ds_rollouts_belief_20261012 as B   # parchea D.run_prefix_multi / D.rollout_forced (creencia/mundo)

D = B.D
MODE = os.environ.get('F1_DS_FORCE')
assert MODE in ('start', 'hist'), 'F1_DS_FORCE=start|hist'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ACTIVE = {'on': False, 'plan': {}}


def real_plan(gp, driver):
    """{vuelta: numero de compuesto} con 0 = largada y L = vuelta de entrada a pits."""
    s = pd.read_csv(os.path.join(ROOT, 'data', 'session_2024_V2.csv'))
    d = s[(s.GP == gp) & (s.Driver == driver.replace('Driver_', ''))].sort_values('LapNumber')
    assert len(d), (gp, driver)
    num = lambda c: int(str(c).lstrip('C'))
    plan, prev = {0: num(d.Compound_Detail.iloc[0])}, d.iloc[0]
    for _, r in d.iloc[1:].iterrows():
        if r.TyreLife < prev.TyreLife or r.Compound_Detail != prev.Compound_Detail:
            plan[int(prev.LapNumber)] = num(r.Compound_Detail)
        prev = r
    return plan if MODE == 'hist' else {0: plan[0]}


def _patch_class(cls):
    if getattr(cls, '_realhist_patched', False):
        return
    orig = cls.step

    def step(self, action):
        if _ACTIVE['on']:
            lap = int(self.lap_number)
            if lap in _ACTIVE['plan']:
                action = self.available_compounds.index(_ACTIVE['plan'][lap]) + 1
            elif MODE == 'hist':
                action = 0
        return orig(self, action)

    cls.step, cls._realhist_patched = step, True


_prefix_belief = D.run_prefix_multi


def _prefix(model, vec_env, raw_env, laps, initial_compounds, prefix_seed):
    _patch_class(type(raw_env))
    _ACTIVE['plan'] = real_plan(_ARGS['gp'], _ARGS['driver'])
    print(f'[realhist] F1_DS_FORCE={MODE} plan={_ACTIVE["plan"]}', flush=True)
    _ACTIVE['on'] = True
    try:
        return _prefix_belief(model, vec_env, raw_env, laps, initial_compounds, prefix_seed)
    finally:
        _ACTIVE['on'] = False


_ARGS = {'gp': sys.argv[sys.argv.index('--gp') + 1], 'driver': sys.argv[sys.argv.index('--driver') + 1]}

if __name__ == '__main__':
    if sys.argv[1] == '--run':
        # Otro script (p. ej. plot_decision_stints.py) con el mismo prefijo forzado. Ese script usa el prefijo original
        # (sin el cambio creencia/mundo de ds_rollouts_belief), asi que se envuelve el original.
        import runpy
        _prefix_belief = B._orig_prefix
        D.run_prefix_multi = _prefix
        target = sys.argv[2] if os.path.isabs(sys.argv[2]) else os.path.join(ROOT, sys.argv[2])
        sys.argv = [target] + sys.argv[3:]
        runpy.run_path(target, run_name='__main__')
    else:
        D.run_prefix_multi = _prefix
        D.main()
