import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import compat; compat.setup_all()

"""
2026-10-27 — Validacion predictiva del simulador con la estrategia real 2024 (pedido del autor, analisis 2).

El piloto del agente ejecuta su largada y sus paradas reales de 2024 durante TODA la carrera (misma derivacion que
scripts/ds_rollouts_realhist_20261024.py: vuelta de entrada L -> compuesto de la tanda siguiente; en las demas vueltas
no para). Mismo entorno, mundo 2024 y semillas que las Figs. 6-7 (scripts/evaluate_world2024_20261010.py):
    fig7 : rivales con sus estrategias reales, SC/VSC y abandonos reales (semilla base 0) -> la carrera es una
           repeticion de 2024 en la que solo varian el ruido de tiempos de vuelta y los adelantamientos;
    fig6 : rivales por logit, SC/VSC pre-sorteados como la Fig. 6, abandonos reales (--real-dnf; semilla base 42).
Guarda la posicion final de TODOS los pilotos en cada carrera (para comparar con la clasificacion real en pista).
No usa agentes ni escribe en trained_models/. Salida: <out>/<scen>/<GP_DRV>.csv (sim, driver, position, is_dnf, ...).
Correr con el entorno de las Figs. 6-7 (reproduce/pipeline_v9r2_20261023.sh) y a traves del envoltorio de pilotos:
    python scripts/gp_drivers_v5hd_20261014.py scripts/validate_real_strategy_20261027.py --scen fig7 --gps Saudi_Arabian_NOR --n-sims 20
"""

import argparse, time
import numpy as np
import pandas as pd
import torch

import evaluate_world2024_20261010 as W   # parchea reset/step del entorno: mundo 2024 y abandonos reales (Fig. 6)
import evaluate_qmip_vs_drl as E
from environment.f1_env_all_drivers import create_f1_env_all_drivers
from evaluate_all_gps import GP_CONFIGS, load_gp_config, build_rival_strategies
from model_loader import ModelLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def real_plan(gp, driver):
    """{vuelta: numero de compuesto}, 0 = largada, L = vuelta de entrada a pits (copia de ds_rollouts_realhist, que
    lee sys.argv al importarse)."""
    s = pd.read_csv(os.path.join(ROOT, 'data', 'session_2024_V2.csv'))
    d = s[(s.GP == gp) & (s.Driver == driver.replace('Driver_', ''))].sort_values('LapNumber')
    assert len(d), (gp, driver)
    num = lambda c: int(str(c).lstrip('C'))
    plan, prev = {0: num(d.Compound_Detail.iloc[0])}, d.iloc[0]
    for _, r in d.iloc[1:].iterrows():
        if r.TyreLife < prev.TyreLife or r.Compound_Detail != prev.Compound_Detail:
            plan[int(prev.LapNumber)] = num(r.Compound_Detail)
        prev = r
    return plan


def run_race(env, plan, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.np_random = np.random.default_rng(seed)
    env.action_space.seed(seed)
    env.reset()
    done, pits, info = False, [], {}
    while not done:
        lap = int(env.lap_number)
        a = env.available_compounds.index(plan[lap]) + 1 if lap in plan else 0
        if lap >= 1 and a:
            pits.append(lap)
        _, _, term, trunc, info = env.step(a)
        done = term or trunc
    rows = [dict(driver=d.name.replace('Driver_', ''), position=int(d.position), is_dnf=bool(d.is_dnf),
                 agent=d is env.agent_driver) for d in env.drivers]
    return rows, pits, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scen', choices=['fig6', 'fig7'], required=True)
    ap.add_argument('--gps', nargs='+', required=True)
    ap.add_argument('--n-sims', type=int, default=500)
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'real_strategy_check_20261027'))
    a = ap.parse_args()
    out = os.path.join(a.out, a.scen)
    os.makedirs(out, exist_ok=True)
    loader = ModelLoader()
    for entry in [c for c in GP_CONFIGS if c['model_dir'] in a.gps]:
        t0 = time.time()
        gp, drv = entry['gp_name'], entry['agent_driver']
        gpc = load_gp_config(gp)
        init_pos = gpc['initial_positions_list_2024']
        plan = real_plan(gp, drv)
        W.register(gp, loader.load(gp).laptime_clear, '2024')
        env = create_f1_env_all_drivers(gp=gp, driver=drv, deterministic=False, yf_enabled=True,
                                        initial_positions=init_pos, loader=loader, verbose=False)
        if a.scen == 'fig7':
            rival_strategies, init_comp = build_rival_strategies(entry)
            env.set_evaluation_mode(eval_mode=True, laps_vsc=gpc['laps_vsc_2024'], laps_sc=gpc['laps_sc_2024'],
                                    initial_positions=init_pos, rival_actions=rival_strategies,
                                    rival_initial_compounds=init_comp)
            base, scen = 0, None
        else:
            W._REAL_DNF[gp] = W.real_dnf_laps(entry)
            m = loader.load(gp)
            scen = E.pregenenerate_race_scenarios(n_sims=a.n_sims, n_laps=m.n_laps, race_vsc_prob=m.vsc_prob,
                                                  race_sc_prob=m.sc_prob, base_seed=42 * 1000)
            base = 42
        print(f'{gp} {drv} plan {plan} ({a.scen})', flush=True)
        recs = []
        for i in range(a.n_sims):
            if scen is not None:
                env.set_evaluation_mode(eval_mode=True, laps_vsc=scen[i][0], laps_sc=scen[i][1],
                                        initial_positions=init_pos, rival_actions={})
            rows, pits, info = run_race(env, plan, base + i)
            for r in rows:
                r.update(sim=i, seed=base + i, agent_pits=' '.join(map(str, pits)))
            recs += rows
        W.restore(gp)
        W._TRACES.clear(); W._CUR.clear()
        df = pd.DataFrame(recs)
        df.to_csv(os.path.join(out, f"{entry['model_dir']}.csv"), index=False)
        ag = df[df.agent]
        print(f"  {drv}: pos media {ag.position.mean():.2f} | paradas {ag.agent_pits.value_counts().head(2).to_dict()} "
              f"| {time.time() - t0:.0f} s", flush=True)


if __name__ == '__main__':
    main()
