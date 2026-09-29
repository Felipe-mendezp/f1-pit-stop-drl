"""Single-driver (sin rivales, Fig. 5 / Tabla N.1) con FILTRO DE COMPUESTOS POCO USADOS — 2026-09-25.

Decisión del autor y su profesor guía: un compuesto cuyo uso en la muestra de la regresión
(carreras 2021–2023 del GP) es < 5 % de las vueltas tiene coeficientes poco confiables; en el
escenario SIN RIVALES ni el agente ni el QMIP pueden usarlo. El entorno multi-piloto (Figs. 6/7)
NO cambia. Nada existente se modifica: este módulo parchea en memoria y delega en los scripts
originales.

  build-filter   calcula el % de vueltas por compuesto y escribe el JSON del filtro
  verify         QMIP filtrado == DP exacto sobre los compuestos retenidos (9 GPs)
  train  <args de train_one_driver_paperhp_20260924.py>
  screen <args de screen_one_driver_20260923.py>
  fig5   <args de plot_fig5_stints_one_driver.py>
  m1     --out-dir <dir>

Parches (instalados antes de importar los scripts originales):
  * environment.f1_env_one_driver.F1EnvOneDriverV2Optimized -> subclase que filtra
    available_compounds por GP y usa Discrete(n_compuestos + 1) (con Discrete(4) y 2 compuestos
    la acción 3 sería un no-op que no avanza la vuelta).
  * evaluate_qmip_vs_drl.build_qmip_params -> el compuesto excluido recibe +1000 s/vuelta en
    beta_0 (el QMIP de 3 neumáticos nunca lo elige) y available_compounds queda filtrado.
  SELECTED_GP / PAPER_GPS siguen con los 3 compuestos: los colores de la Fig. 5 conservan su rol
  (duro/medio/blando) y el filtrado ocurre dentro del entorno.
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import config
import compat; compat.setup_all()
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import json

THRESHOLD = 0.05
FILTER_JSON = os.path.join(str(config.DATA_DIR), 'compound_filter_one_driver_5pct_20260925.json')
BIG_M = 1000.0


def _norm(c):
    return int(c[1:]) if isinstance(c, str) else int(c)


def build_filter(threshold=THRESHOLD, out=FILTER_JSON):
    """% de vueltas por compuesto en la muestra de la regresión (all_laps: 2021-2023, sin WET/INT,
    mismo remapeo C2->C5 de Emilia que train_regressions.py)."""
    import pandas as pd
    from models.regressions import all_laps
    from plot_fig5_stints_one_driver import SELECTED_GP
    L = all_laps()
    L['GP'] = L['GP'].replace('Styrian Grand Prix', 'Austrian Grand Prix')
    m = L['GP'] == 'Emilia Romagna Grand Prix'
    L.loc[m, 'Compound_Detail'] = L.loc[m, 'Compound_Detail'].replace('C2', 'C5')
    res = {'threshold': threshold, 'source': 'models.regressions.all_laps() (2021-2023 race laps)',
           'gps': {}}
    for gp, comps in SELECTED_GP.items():
        g = L[L.GP == gp]
        share = {c: float((g.Compound_Detail == c).mean()) for c in comps}
        keep = [c for c in comps if share[c] >= threshold]
        res['gps'][gp] = {'compounds_2024': comps, 'share_2021_2023': share, 'keep': keep,
                          'dropped': [c for c in comps if c not in keep]}
        print(f"{gp:28s} " + '  '.join(f"{c}:{100 * share[c]:5.1f}%" for c in comps)
              + (f"   -> excluye {res['gps'][gp]['dropped']}" if len(keep) < len(comps) else ''))
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"-> {out}")
    return res


def load_filter(path=FILTER_JSON):
    with open(path) as f:
        d = json.load(f)
    return {gp: [_norm(c) for c in v['keep']] for gp, v in d['gps'].items() if v['dropped']}


_INSTALLED = False


def install(path=FILTER_JSON):
    """Parchea entorno y QMIP en memoria. Llamar ANTES de importar los scripts originales."""
    global _INSTALLED
    if _INSTALLED:
        return
    FILTER = load_filter(path)
    from gymnasium.spaces import Discrete
    import environment.f1_env_one_driver as envmod
    import evaluate_qmip_vs_drl as E6
    Orig = envmod.F1EnvOneDriverV2Optimized

    class F1EnvOneDriverFiltered(Orig):
        def __init__(self, n_laps, available_compounds, gp, driver, model_laptime, verbose=False):
            keep = FILTER.get(gp)
            if keep:
                available_compounds = [c for c in available_compounds if _norm(c) in keep]
            super().__init__(n_laps, available_compounds, gp, driver, model_laptime, verbose)
            self.action_space = Discrete(self._n_compounds + 1)

    F1EnvOneDriverFiltered.__name__ = 'F1EnvOneDriverV2Optimized'
    envmod.F1EnvOneDriverV2Optimized = F1EnvOneDriverFiltered

    _orig_bqp = E6.build_qmip_params

    def build_qmip_params_filtered(loader, gp_name):
        qp = _orig_bqp(loader, gp_name)
        keep = FILTER.get(gp_name)
        if keep:
            for c, name in qp['compound_to_qmip'].items():
                if c not in keep:
                    qp['betas_dict'][name][0] += BIG_M
            qp['available_compounds'] = [c for c in qp['available_compounds'] if c in keep]
        return qp

    E6.build_qmip_params = build_qmip_params_filtered
    _INSTALLED = True
    print(f"[filtro 5%] instalado: {({gp[:-11]: keep for gp, keep in FILTER.items()})}")


def verify():
    """QMIP filtrado (MPC sobre el env filtrado) == DP exacto sobre los compuestos retenidos."""
    install()
    import pickle
    from plot_fig5_stints_one_driver import SELECTED_GP, LAPS_DD
    from screen_one_driver_20260923 import rollout_full, qmip_actor
    from environment.f1_env_one_driver import F1EnvOneDriverV2Optimized as Env
    from evaluate_qmip_vs_drl import build_qmip_params
    from optimization.utils_qmip import extract_simplified_calculator
    from diag_dp_one_driver_20260925 import solve, fmt
    from model_loader import ModelLoader
    loader = ModelLoader(); ok = True
    for gp, comps in SELECTED_GP.items():
        n = LAPS_DD[gp]
        pred = pickle.load(open(str(config.SIMULATION_DIR / f'{gp}_fast_predictor.pkl'), 'rb'))
        env = Env(n_laps=n, available_compounds=comps, gp=gp, driver='VER', model_laptime=pred)
        t_q, changed, stops, plan = rollout_full(env, qmip_actor(build_qmip_params(loader, gp)))
        calc = extract_simplified_calculator(pred, env.available_compounds)
        t_dp, c0, p = solve(calc, env.available_compounds, n, allow_last=False)
        good = abs(t_q - t_dp) < 1e-3 and changed
        ok &= good
        print(f"{gp:28s} comps={env.available_compounds} A={env.action_space.n}  QMIP {t_q:9.2f} [{plan}]"
              f"  DP {t_dp:9.2f} {fmt(c0, p, n)}  {'OK' if good else '!! DIFIERE'}")
    print('VERIFY', 'OK' if ok else 'FALLÓ')
    sys.exit(0 if ok else 1)


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == 'build-filter':
        build_filter()
        return
    if cmd == 'verify':
        verify()
        return
    install()
    if cmd == 'train':
        import train_one_driver_paperhp_20260924 as T
        sys.argv = ['train_one_driver_paperhp_20260924.py'] + rest
        T.main()
    elif cmd == 'screen':
        import screen_one_driver_20260923 as S
        sys.argv = ['screen_one_driver_20260923.py'] + rest
        S.main()
    elif cmd == 'fig5':
        import plot_fig5_stints_one_driver as P
        sys.argv = ['plot_fig5_stints_one_driver.py'] + rest
        P.main()
    elif cmd == 'm1':
        import argparse
        ap = argparse.ArgumentParser(); ap.add_argument('--out-dir', required=True)
        a = ap.parse_args(rest)
        os.makedirs(a.out_dir, exist_ok=True)
        import generate_appendix_tables as G
        G.OUT_DIR = a.out_dir
        G.table_m1()
    else:
        print(__doc__); sys.exit(2)


if __name__ == '__main__':
    main()
