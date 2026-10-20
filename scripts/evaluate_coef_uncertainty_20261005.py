import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import compat; compat.setup_all()

"""
2026-10-05 — Figs. 6 y 7 con INCERTIDUMBRE EN LOS COEFICIENTES DE NEUMATICO.

Mismo protocolo que evaluate_qmip_vs_drl.py (--fig 6, carreras estocasticas, semillas 42..) o
evaluate_all_gps_with_qmip.py (--fig 7, replay 2024, semillas 0..), con PPO, RecurrentPPO y QMIP y
500 carreras por GP, pero al inicio
de cada carrera se sortean el offset y la pendiente de cada compuesto 2024:
    coef ~ N(valor del simulador, (sigma * k)^2),  k = 1 si estimado, k = --factor si imputado
con sigma de results/tyre_coef_study_20260930/coef_uncertainty_20261005.csv (estimados: error estandar
agrupado por stint; imputados: CV medio de las pendientes / se medio de los offsets del GP). Pendientes
truncadas en 0. El sorteo queda fijo durante la carrera; el ruido por vuelta del simulador no cambia.
--world opcion2|opcion3 (06-oct): el centro del sorteo pasa a ser el coeficiente del simulador alternativo
(tyre_coef_audit.csv de trained_models/simulation_tyre_<opcion>_20261004, re-expresado respecto de la
referencia del simulador publicado; el resto de la regresion publicada no cambia); con --factor 0 y
sin sigma de los estimados (--no-est-noise) es un "mundo" fijo distinto del que creen DRL y QMIP.
Los coeficientes de una carrera dependen solo de su semilla: PPO, RPPO y QMIP enfrentan los mismos sorteos.
QMIP planifica con los valores puntuales (betas calculados con los coeficientes sin sortear).
Solo afecta al modelo de vuelta en pista limpia (laptime_clear); el de bandera amarilla no cambia.

Agentes: los de F1_RL_AGENTS_DIR (por defecto los publicados); --tag solo nombra la salida
(p. ej. F1_RL_AGENTS_DIR=trained_models/rl_agents_tyre_control_20261004 ... --tag control).
No escribe en trained_models/: salida en results/coef_uncertainty_20261005/fig<6|7>_<tag>_f<factor>/.
    python scripts/evaluate_coef_uncertainty_20261005.py --fig 6 --factor 3 [--tag publicado] [--gps Belgian_NOR]
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
TABLE = os.path.join(ROOT, 'results', 'tyre_coef_study_20260930', 'coef_uncertainty_20261005.csv')
SALT = 20261005
PAPER_GPS = ['Bahrain_ALO', 'Saudi_Arabian_NOR', 'Miami_OCO', 'Hungarian_SAI', 'Belgian_NOR', 'Dutch_RUS',
             'Singapore_HUL', 'United_States_PIA']

_SPEC = {}  # gp -> dict(pred, base, idx, sd, slope_mask)
_LOG = []   # (gp, race key, drawn coefs) for the audit file


def register(gp, pred, table, factor, world=None, est_noise=True):
    t = table[(table.GP == gp) & (table.estado != 'referencia')]
    ref = pred.reference_categories_['Compound_Detail']
    world_val = None
    if world:  # option table (offset vs its own ref, slope); re-expressed vs the published ref
        a = pd.read_csv(os.path.join(ROOT, 'trained_models', f'simulation_tyre_{world}_20261004',
                                     'tyre_coef_audit.csv'))
        a = a[a.GP == gp].drop_duplicates('compound').set_index('compound')
        off = a.offset_vs_ref - a.offset_vs_ref.get(ref, 0.0)
        world_val = {(c, 'offset'): off[c] for c in a.index} | {(c, 'pendiente'): a.slope[c] for c in a.index}
    names = list(pred.feature_names_)
    idx, sd, slope = [], [], []
    for r in t.itertuples():
        name = (f'TyreLife:C(Compound_Detail)[{r.compuesto}]' if r.tipo == 'pendiente'
                else f"C(Compound_Detail, Treatment(reference='{ref}'))[T.{r.compuesto}]")
        i = names.index(name)
        assert abs(float(pred.coef_[i]) - r.valor) < 1e-5, (gp, name)
        idx.append(i)
        sd.append(r.sigma_f1 * (factor if r.estado == 'imputado' else (1.0 if est_noise else 0.0)))
        slope.append(r.tipo == 'pendiente')
    base = pred.coef_.copy()
    center = (base[idx].astype(float) if world_val is None else
              np.array([world_val[(r.compuesto, r.tipo)] for r in t.itertuples()], dtype=float))
    _SPEC[gp] = dict(pred=pred, base=base, center=center, idx=np.array(idx), sd=np.array(sd),
                     slope=np.array(slope), labels=[f'{r.compuesto}:{r.tipo}' for r in t.itertuples()])


def restore(gp):
    s = _SPEC[gp]; s['pred'].coef_[:] = s['base']


_orig_reset = F1EnvAllDrivers.reset
def _reset(self, *a, **kw):
    s = _SPEC.get(self.gp)
    if s is not None:
        key = np.random.get_state()[1][:2]  # set by np.random.seed(seed) just before reset in both runners
        rng = np.random.default_rng([SALT, int(key[0]), int(key[1])])
        vals = s['center'] + rng.standard_normal(len(s['idx'])) * s['sd']
        vals[s['slope']] = np.maximum(vals[s['slope']], 0.0)
        coef = s['base'].copy(); coef[s['idx']] = vals
        s['pred'].coef_[:] = coef
        _LOG.append((self.gp, int(key[0]), *np.round(vals, 5)))
    return _orig_reset(self, *a, **kw)
F1EnvAllDrivers.reset = _reset

_orig_qmip = E.build_qmip_params
def _qmip_params(loader, gp_name):  # QMIP plans with the point estimates
    restore(gp_name)
    return _orig_qmip(loader, gp_name)
E.build_qmip_params = _qmip_params
E7.build_qmip_params = _qmip_params
E.ALGO_ORDER = ['PPO', 'RecurrentPPO', 'QMIP']
E7.DRL_ALGOS = ['PPO', 'RecurrentPPO']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fig', type=int, choices=[6, 7], default=6)
    ap.add_argument('--tag', default='publicado')
    ap.add_argument('--factor', type=float, default=3.0)
    ap.add_argument('--world', choices=['opcion2', 'opcion3'], default=None)
    ap.add_argument('--no-est-noise', action='store_true')
    ap.add_argument('--gps', nargs='+', default=PAPER_GPS)
    ap.add_argument('--n-sims', type=int, default=E.N_SIMULATIONS)
    ap.add_argument('--seed', type=int, default=None, help='default: 42 (Fig. 6) / 0 (Fig. 7), as in the paper')
    ap.add_argument('--max-pits', type=int, default=3)
    args = ap.parse_args()
    seed = args.seed if args.seed is not None else (42 if args.fig == 6 else 0)
    out = os.path.join(ROOT, 'results', 'coef_uncertainty_20261005', f'fig{args.fig}_{args.tag}_f{args.factor:g}' + (f'_mundo-{args.world}' if args.world else '')
                       + ('_sinruidoest' if args.no_est_noise else ''))
    os.makedirs(out, exist_ok=True)
    table = pd.read_csv(TABLE)
    loader = ModelLoader()
    all_res = []
    for entry in [c for c in GP_CONFIGS if c['model_dir'] in args.gps]:
        gp = entry['gp_name']
        register(gp, loader.load(gp).laptime_clear, table, args.factor, args.world, not args.no_est_noise)
        if args.fig == 6:
            res = E.evaluate_gp(entry, loader, n_simulations=args.n_sims, base_seed=seed, p=args.max_pits)
        else:
            res = E7.evaluate_gp_with_qmip(entry, loader, n_simulations=args.n_sims, base_seed=seed, p=args.max_pits)
        restore(gp)
        if res:
            all_res.append({'display_name': entry['display_name'], 'results': res})
            E.save_cache(all_res, path=os.path.join(out, 'qmip_vs_drl_coefunc_results.csv'))
    rows = []
    for gp, key, *vals in _LOG:
        rows.append(dict(GP=gp, key=key, **dict(zip(_SPEC[gp]['labels'], vals))))
    pd.DataFrame(rows).drop_duplicates().to_csv(os.path.join(out, 'coef_draws.csv'), index=False)
    print('agents:', config.RL_AGENTS_DIR, '->', out)


if __name__ == '__main__':
    main()
