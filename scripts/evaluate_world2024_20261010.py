import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import compat; compat.setup_all()

"""
2026-10-10 — Figs. 6 y 7 en un "mundo 2024": coeficientes de neumatico re-estimados con 2021-2024.

Los agentes (F1_RL_AGENTS_DIR) y la QMIP creen los coeficientes del simulador F1_SIMULATION_DIR
(opcion3 2021-23); el entorno simula las vueltas en pista limpia con los coeficientes de compuesto
(offset vs referencia y TyreLife x compuesto) de
results/tyre_coef_study_20260930/world2024_20261010/coefs_opcion3_2021_24.csv
(misma metodologia opcion3, 2024 como un anio mas), re-expresados respecto de la referencia del
simulador. El resto de la Ec. 1 y el modelo de bandera amarilla no cambian.
Mismo protocolo que evaluate_qmip_vs_drl.py (--fig 6, semilla 42) y evaluate_all_gps_with_qmip.py
(--fig 7, semilla 0): PPO, RecurrentPPO y QMIP, 500 carreras por GP, mismas semillas.
2026-10-12: --algos (los 5 DRL) y F1_WORLD2024_TYRES=opcion3h (creencia y mundo de la opcion 3 jerarquica;
con F1_SIMULATION_DIR=trained_models/simulation_v5h8_20261012, F1_RL_AGENTS_DIR=.../rl_agents_v5h_20261011/selected).
--world none reproduce la evaluacion original (control de que el envoltorio no cambia nada).
2026-10-14: --real-dnf (Fig. 6) fija los abandonos de los rivales a los reales de 2024 (ver _REAL_DNF).
No escribe en trained_models/: salida en results/world2024_20261010/fig<6|7>_<world>/<GP>.csv
    source reproduce/env_paper_v3_20261007.sh
    F1_SIMULATION_DIR=... F1_RL_AGENTS_DIR=... python scripts/evaluate_world2024_20261010.py --fig 6 --gps Bahrain_ALO
"""

import argparse
import numpy as np
import pandas as pd

import evaluate_qmip_vs_drl as E
import evaluate_all_gps_with_qmip as E7
from environment.f1_env_all_drivers import F1EnvAllDrivers
from evaluate_all_gps import GP_CONFIGS
from model_loader import ModelLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(ROOT, 'results', 'tyre_coef_study_20260930')
# F1_WORLD2024_TYRES (2026-10-12): 'opcion3' (por defecto, como antes) u 'opcion3h' (opcion 3 jerarquica:
# creencia = simulation_v5h8_20261012, mundo = world2024h_20261012/coefs_opcion3h_2021_24.csv)
TYRES = os.environ.get('F1_WORLD2024_TYRES', 'opcion3')
WORLD, BELIEF = {
    'opcion3': (os.path.join(STUDY, 'world2024_20261010', 'coefs_opcion3_2021_24.csv'),
                os.path.join(STUDY, 'pooled_20261004', 'coefs_opcion3.csv')),
    'opcion3h': (os.path.join(STUDY, 'world2024h_20261012', 'coefs_opcion3h_2021_24.csv'),
                 os.path.join(STUDY, 'pooled_hier_20261011', 'coefs_opcion3h.csv')),
    # 2026-10-15: v7 (TyreLife 2021 x 0.716): creencia = simulation_v7_20261015, mundo = opcion3h 2021-24 con el mismo factor
    'v7': (os.path.join(ROOT, 'results', 'sim_v7_20261015', 'tyre_v7', 'coefs_opcion3h_2021_24.csv'),
           os.path.join(ROOT, 'results', 'sim_v7_20261015', 'tyre_v7', 'coefs_opcion3h.csv')),
    # 2026-10-27: v10 (paradas reales, TyreLife 2021 x 0.740): creencia = simulation_v10_20261027, mundo = opcion3h 2021-24 v10
    'v10': (os.path.join(ROOT, 'results', 'sim_v10_20261027', 'tyre_v10', 'coefs_opcion3h_2021_24.csv'),
            os.path.join(ROOT, 'results', 'sim_v10_20261027', 'tyre_v10', 'coefs_opcion3h.csv')),
}[TYRES]

_SPEC = {}  # gp -> dict(pred, base, world)


def register(gp, pred, world):
    ref = pred.reference_categories_['Compound_Detail']
    names = list(pred.feature_names_)
    base = pred.coef_.copy()
    coef = base.copy()
    # control: el simulador cargado debe tener los coeficientes opcion3 2021-23
    b = pd.read_csv(BELIEF); b = b[b.GP == gp].set_index('compound')
    w = pd.read_csv(WORLD); w = w[w.GP == gp].set_index('compound')
    for t, label in [(b, 'creencia'), (w, 'mundo')]:
        for c in t.index:
            off = t.offset_vs_ref[c] - t.offset_vs_ref[ref]
            i_s = names.index(f'TyreLife:C(Compound_Detail)[{c}]')
            if label == 'creencia':
                assert abs(float(base[i_s]) - t.slope[c]) < 1e-5, (gp, c, 'slope')
                if c != ref:
                    i_o = names.index(f"C(Compound_Detail, Treatment(reference='{ref}'))[T.{c}]")
                    assert abs(float(base[i_o]) - off) < 1e-5, (gp, c, 'offset')
            elif world == '2024':
                coef[i_s] = t.slope[c]
                if c != ref:
                    coef[names.index(f"C(Compound_Detail, Treatment(reference='{ref}'))[T.{c}]")] = off
    _SPEC[gp] = dict(pred=pred, base=base, world=coef)
    return [(n, float(base[i]), float(coef[i])) for i, n in enumerate(names) if base[i] != coef[i]]


def restore(gp):
    s = _SPEC[gp]; s['pred'].coef_[:] = s['base']


# 2026-10-14: registro de paradas por carrera. Los evaluadores publicados (pit_stops) cuentan solo los cambios de
# compuesto; una parada a un neumatico nuevo del mismo compuesto no se cuenta (D28 en
# docs/PAPER_CODIGO_CORRESPONDENCIA.md). Aqui se cuenta toda accion != 0 desde la vuelta 1 (como
# select_seeds_v5_20261010.py) y se agregan al CSV las columnas n_pits_all y pit_laps. No cambia la simulacion.
_TRACES, _CUR = [], []
_orig_step = F1EnvAllDrivers.step
def _step(self, action):
    lap, before = int(self.lap_number), int(self.agent_driver.compound)
    out = _orig_step(self, action)
    _CUR.append((lap, int(action), before, int(self.agent_driver.compound)))
    return out
F1EnvAllDrivers.step = _step


def _flush():
    global _CUR
    if _CUR:
        _TRACES.append(_CUR)
    _CUR = []


# 2026-10-14: --real-dnf (Fig. 6). Con rivales simulados, cada rival que abandono en 2024 abandona en su vuelta real
# (el -1 de rival_strategies en GP_CONFIGS, el mismo que usa la Fig. 7); los demas rivales nunca abandonan. Usa el
# mecanismo _rival_dnf_lap del entorno (mismo momento que el -1) y reemplaza el sorteo de F1_RIVAL_DNF si lo hay.
# Sin --real-dnf no cambia nada.
_REAL_DNF = {}  # gp -> {rival: vuelta del -1}


def real_dnf_laps(entry):
    return {d: lap for d, st in entry['rival_strategies'].items() for lap, c in st.items() if c == -1}


_orig_reset = F1EnvAllDrivers.reset
def _reset(self, *a, **kw):
    _flush()
    s = _SPEC.get(self.gp)
    if s is not None:
        s['pred'].coef_[:] = s['world']
    out = _orig_reset(self, *a, **kw)
    if self.gp in _REAL_DNF and not self.rival_actions_dict:
        names = {d.name for d in self.rival_drivers}
        assert set(_REAL_DNF[self.gp]) <= names, (self.gp, set(_REAL_DNF[self.gp]) - names)
        self._rival_dnf_lap = dict(_REAL_DNF[self.gp])
    return out
F1EnvAllDrivers.reset = _reset


def _add_pit_columns(csv):
    """Agrega n_pits_all y pit_laps al CSV recien guardado (una traza por fila, en el orden de evaluacion).
    Control: los cambios de compuesto de cada traza deben coincidir con pit_stops de esa fila."""
    _flush()
    df = pd.read_csv(csv)
    assert len(_TRACES) == len(df), f'{csv}: {len(_TRACES)} trazas vs {len(df)} filas'
    laps = [[l for l, a, _, _ in t if l >= 1 and a != 0] for t in _TRACES]
    changes = [sum(1 for l, _, b, c in t if l >= 1 and b != c) for t in _TRACES]
    # el contador DRL de los evaluadores mira el compuesto ANTES de cada step, asi que no ve un cambio en el
    # ultimo step de la carrera; se acepta esa diferencia y se informa (columna last_lap_pit)
    changes_nolast = [sum(1 for l, _, b, c in t[:-1] if l >= 1 and b != c) for t in _TRACES]
    pit = df['pit_stops'].tolist()
    assert all(p in (c, c2) for p, c, c2 in zip(pit, changes, changes_nolast)), f'{csv}: trazas desalineadas con pit_stops'
    df['n_pits_all'] = [len(x) for x in laps]
    df['pit_laps'] = [' '.join(map(str, x)) for x in laps]
    df['last_lap_pit'] = [int(bool(t) and t[-1][1] != 0) for t in _TRACES]
    df.to_csv(csv, index=False)
    _TRACES.clear()
    extra = (df['n_pits_all'] != df['pit_stops']).groupby(df['algo']).sum()
    print(f'paradas al mismo compuesto (carreras con n_pits_all != pit_stops): {extra.to_dict()}')

_orig_qmip = E.build_qmip_params
def _qmip_params(loader, gp_name):  # QMIP planifica con lo que cree (opcion3 2021-23)
    restore(gp_name)
    return _orig_qmip(loader, gp_name)
E.build_qmip_params = _qmip_params
E7.build_qmip_params = _qmip_params
DRL = ['PPO', 'RecurrentPPO']   # --algos lo cambia (2026-10-12)
E.ALGO_ORDER = DRL + ['QMIP']
E7.DRL_ALGOS = DRL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fig', type=int, choices=[6, 7], required=True)
    ap.add_argument('--world', choices=['2024', 'none'], default='2024')
    ap.add_argument('--gps', nargs='+', required=True)
    ap.add_argument('--n-sims', type=int, default=E.N_SIMULATIONS)
    ap.add_argument('--max-pits', type=int, default=3)
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'world2024_20261010'))
    ap.add_argument('--algos', nargs='+', default=DRL, choices=['DQN', 'A2C', 'TRPO', 'PPO', 'RecurrentPPO'],
                    help='metodos DRL a evaluar (la QMIP siempre)')
    ap.add_argument('--real-dnf', action='store_true',
                    help='Fig. 6: los rivales abandonan en su vuelta real de 2024 (salida fig6_<world>_realdnf)')
    args = ap.parse_args()
    assert not (args.real_dnf and args.fig != 6), '--real-dnf solo aplica a la Fig. 6 (la Fig. 7 ya usa los reales)'
    E.ALGO_ORDER = E7.ALGO_ORDER = list(args.algos) + ['QMIP']
    E7.DRL_ALGOS = list(args.algos)
    print('neumaticos:', TYRES, '| creencia:', BELIEF, '| mundo:', WORLD, '| metodos:', E.ALGO_ORDER, flush=True)
    seed = 42 if args.fig == 6 else 0
    out = os.path.join(args.out, f'fig{args.fig}_{args.world}' + ('_realdnf' if args.real_dnf else ''))
    os.makedirs(out, exist_ok=True)
    loader = ModelLoader()
    for entry in [c for c in GP_CONFIGS if c['model_dir'] in args.gps]:
        gp = entry['gp_name']
        _flush(); _TRACES.clear()
        if args.real_dnf:
            _REAL_DNF[gp] = real_dnf_laps(entry)
            print(f'{gp}: abandonos reales {_REAL_DNF[gp]}', flush=True)
        changed = register(gp, loader.load(gp).laptime_clear, args.world)
        print(f'{gp}: {len(changed)} coeficientes cambiados en el mundo')
        for n, b, w in changed:
            print(f'   {n}: {b:.4f} -> {w:.4f}')
        if args.fig == 6:
            res = E.evaluate_gp(entry, loader, n_simulations=args.n_sims, base_seed=seed, p=args.max_pits)
        else:
            res = E7.evaluate_gp_with_qmip(entry, loader, n_simulations=args.n_sims, base_seed=seed, p=args.max_pits)
        restore(gp)
        csv = os.path.join(out, f"{entry['model_dir']}.csv")
        E.save_cache([{'display_name': entry['display_name'], 'results': res}], path=csv)
        _add_pit_columns(csv)
    print('agents:', config.RL_AGENTS_DIR, '| sim:', config.SIMULATION_DIR if hasattr(config, 'SIMULATION_DIR') else '?', '->', out)


if __name__ == '__main__':
    main()
