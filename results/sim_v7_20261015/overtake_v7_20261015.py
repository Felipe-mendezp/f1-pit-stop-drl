"""2026-10-15 — Logit de adelantamiento del simulador v7 (mismo procedimiento que el publicado,
src/models/logit_overtakes_pit.py, sin modificarlo).

Cambian solo las entradas:
  - vida del neumatico de 2021 segun F1_TYRELIFE_2021 (load_data la aplica via src/tyrelife_2021.py);
  - tiempos predichos con la Ec. 1 del simulador de la version (PRED: 'v7' = artifacts_<tag>/<GP>/laptime_clear_hist.pkl, la Ec. 1 de v7 con los
    coeficientes de piloto historicos de 2021-23, igual que el publicado, que usa la Ec. 1 historica;
    'publicado' = trained_models/simulation/<GP>_fast_predictor.pkl, como el original).
GPS: 'paper' (8 GP del simulador) o 'todos' (los 9 del original, con Emilia Romagna; solo con PRED=publicado).
El modelo tiene intercepto y pendiente propios por GP, asi que quitar Emilia Romagna no cambia los demas GP
(se verifica en el control).
Control: F1_TYRELIFE_2021=legacy PRED=publicado GPS=todos reproduce los overtake_logit.pkl publicados.
Salida: results/sim_v7_20261015/overtake_<tag>_<pred>_<gps>/ (params.csv, prob_table.csv y, con PRED=v7, los
pickles en artifacts_<tag>/<GP>/overtake_{logit,threshold}.pkl).
"""
import os, sys, pickle
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
import models.logit_overtakes_pit as O

TAG = os.environ.get('F1_TYRELIFE_2021', 'legacy')
PRED = os.environ.get('PRED', 'v7')
GPS = os.environ.get('GPS', 'paper')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, f'overtake_{TAG}_{PRED}_{GPS}'); os.makedirs(OUT, exist_ok=True)
ART = os.path.join(HERE, f'artifacts_{TAG}')
PUB = config.MODELS_DIR / 'simulation'


def main():
    laps, selected, reg = O.load_data()
    if GPS == 'paper':
        selected = {g: c for g, c in selected.items() if g != 'Emilia Romagna Grand Prix'}
    if PRED == 'v7':
        reg = {}
        for gp in selected:
            with open(os.path.join(ART, gp, 'laptime_clear_hist.pkl'), 'rb') as fh:
                reg[gp] = pickle.load(fh)
    df = O.optimize_f1_analysis(laps, selected, reg)
    # mismos filtros que el __main__ del original
    df = df[df['LapNumber'] > 3]
    df = df[df['LapNumber'] < df['LastLap']]
    df = df.dropna(subset=['interval_front_real', 'delta_t_lap', 'DRS', 'overtake', 'rival_PitOut'])
    df['delta_total'] = df['interval_front_real'] + df['delta_t_lap']
    df = df[(df['delta_total'] >= -30) & (df['delta_total'] <= 30.0)]
    res = O.enhanced_logit_modeling_with_evaluation(df, show_plots=False, save_plots=False)
    O.overtake_probability_table(res).to_csv(os.path.join(OUT, 'prob_table.csv'), index=False)
    rows = []
    for gp, m in res['models'].items():
        with open(PUB / gp / 'overtake_logit.pkl', 'rb') as fh:
            pub = pickle.load(fh)
        with open(PUB / gp / 'overtake_threshold.pkl', 'rb') as fh:
            pth = pickle.load(fh)
        st = res['gp_stats'][gp]
        rows.append(dict(GP=gp, const=m.params['const'], slope=m.params['delta_total'],
                         threshold=res['metrics'][gp]['threshold'], auc=res['metrics'][gp]['auc'],
                         n=st['total_samples'], overtakes=st['overtakes'],
                         const_pub=pub.params['const'], slope_pub=pub.params['delta_total'], threshold_pub=pth))
        if PRED == 'v7':
            with open(os.path.join(ART, gp, 'overtake_logit.pkl'), 'wb') as fh:
                pickle.dump(m, fh)
            with open(os.path.join(ART, gp, 'overtake_threshold.pkl'), 'wb') as fh:
                pickle.dump(res['metrics'][gp]['threshold'], fh)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, 'params.csv'), index=False)
    pd.set_option('display.width', 220)
    print(T.round(4).to_string(index=False))
    print('max |dif| con el publicado: const', round((T.const - T.const_pub).abs().max(), 6),
          '| slope', round((T.slope - T.slope_pub).abs().max(), 6))


if __name__ == '__main__':
    main()
