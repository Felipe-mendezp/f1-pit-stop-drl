"""2026-10-29 — Logit de adelantamiento v10eGQ corregido (reemplaza a overtake_v10_20261027.py en la cadena).

Mismo procedimiento que results/sim_v7_20261015/overtake_v7_20261015.py (src/models/logit_overtakes_pit.py: logit
pooled con intercepto y pendiente de delta_total propios de cada GP, Ec. 1 v10eGQ con pilotos historicos, TL_GQ y factor
2021), con dos cambios en la muestra (diagnostico del 28-29 oct, results/sim_v10eGQ_20261028/README_adelantamiento_20261029.md):
  1. Etiqueta por par: adelanta = termina la vuelta L+1 delante del auto que tenia inmediatamente adelante al final de
     la vuelta L (antes: gana posicion neta y TrackStatus(L) == 1, que no contaba los adelantamientos a un rival que
     para en boxes si el de atras tambien pierde una posicion, y ponia 0 en toda vuelta L no verde).
  2. Muestra: solo pares cuya vuelta L+1 no tiene SC, VSC ni bandera roja (TrackStatus sin 4, 5, 6, 7); antes entraban
     vueltas con SC/VSC con etiqueta 0, lo que aplanaba las curvas (Saudi Arabia, Singapur).
Se usan todos los gaps iniciales (el simulador aplica el logit a todos los pares en pista libre con
F1_OVERTAKE_MAX_GAP=inf, y sobre el gap predicho medio con F1_OVERTAKE_MEAN_GAP=1).
    F1_TYRELIFE_2021=v10eGQ ~/.venvs/f1-drl-v2/bin/python results/sim_v10eGQ_20261028/overtake_v10f_20261029.py
Salida: results/sim_v10eGQ_20261028/overtake_v10f/ (params.csv, prob_table.csv, sample_counts.csv) y
artifacts/<GP>/overtake_{logit,threshold}.pkl
"""
import os, sys, pickle
ROOT = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__))
assert os.environ.get('F1_TYRELIFE_2021') == 'v10eGQ'
sys.path.insert(0, HERE); sys.path[:0] = [ROOT, os.path.join(ROOT, 'src')]
import common_v10_20261027  # noqa: F401  (ganchos de compuestos v10d y edad TL_GQ)
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
import models.logit_overtakes_pit as O

OUT = os.path.join(HERE, 'overtake_v10f'); os.makedirs(OUT, exist_ok=True)
ART = os.path.join(HERE, 'artifacts')
D = [-20, -10, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 10, 20]


def label_and_filter(full):
    """Muestra con los filtros del original, etiqueta por par y vuelta L+1 sin SC/VSC/roja."""
    df = full[(full.LapNumber > 3) & (full.LapNumber < full.LastLap)]
    df = df.dropna(subset=['interval_front_real', 'delta_t_lap', 'DRS', 'overtake', 'rival_PitOut']).copy()
    # busquedas sobre el df completo (sin filtrar): rival = auto en la posicion de adelante en la vuelta L
    pos = full.set_index(['Year', 'GP', 'LapNumber', 'Position']).Driver.to_dict()
    idx = full.set_index(['Year', 'GP', 'LapNumber', 'Driver'])
    npos, last = idx.next_position.to_dict(), idx.LastLap.to_dict()
    ts = full.set_index(['Year', 'GP', 'Driver', 'LapNumber']).TrackStatus.to_dict()
    key = list(zip(df.Year, df.GP, df.LapNumber))
    riv = [pos.get((y, g, l, p - 1)) for (y, g, l), p in zip(key, df.Position)]
    r_np = np.array([npos.get((y, g, l, r), np.nan) for (y, g, l), r in zip(key, riv)], dtype=float)
    r_last = np.array([last.get((y, g, l, r), np.nan) for (y, g, l), r in zip(key, riv)], dtype=float)
    df['overtake_old'] = df['overtake']
    df['overtake'] = (df.next_position.notna().values & ~np.isnan(r_np) & (df.next_position.values < r_np)
                      & (r_last != df.LapNumber.values)).astype(int)
    ts1 = pd.Series([str(ts.get((y, g, d, l + 1))) for y, g, d, l in zip(df.Year, df.GP, df.Driver, df.LapNumber)],
                    index=df.index)
    df['clear_L1'] = ~ts1.str.replace('.0', '', regex=False).str.contains('[4567]')
    return df


def main():
    laps, selected, _ = O.load_data()
    selected = {g: c for g, c in selected.items() if g != 'Emilia Romagna Grand Prix'}
    reg = {}
    for gp in selected:
        with open(os.path.join(ART, gp, 'laptime_clear_hist.pkl'), 'rb') as fh:
            reg[gp] = pickle.load(fh)
    full = O.optimize_f1_analysis(laps, selected, reg)
    df = label_and_filter(full)
    dt = df.interval_front_real + df.delta_t_lap
    inside = (dt >= -30) & (dt <= 5)
    cnt = df[inside].groupby('GP').agg(n_antes=('overtake_old', 'size'), ot_antes=('overtake_old', 'sum'),
                                       n_L1_sin_sc=('clear_L1', 'sum'))
    sample = df[df.clear_L1]
    res = O.enhanced_logit_modeling_with_evaluation(sample, show_plots=False, save_plots=False)
    rows = []
    for gp, m in res['models'].items():
        st = res['gp_stats'][gp]
        a, b = m.params['const'], m.params['delta_total']
        rows.append(dict(GP=gp, const=a, slope=b, threshold=res['metrics'][gp]['threshold'], auc=res['metrics'][gp]['auc'],
                         n=st['total_samples'], overtakes=st['overtakes'],
                         **{f'p{d:+g}': 1 / (1 + np.exp(-(a + b * d))) for d in D}))
        with open(os.path.join(ART, gp, 'overtake_logit.pkl'), 'wb') as fh:
            pickle.dump(m, fh)
        with open(os.path.join(ART, gp, 'overtake_threshold.pkl'), 'wb') as fh:
            pickle.dump(res['metrics'][gp]['threshold'], fh)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, 'params.csv'), index=False)
    O.overtake_probability_table(res).to_csv(os.path.join(OUT, 'prob_table.csv'), index=False)
    cnt.join(T.set_index('GP')[['n', 'overtakes']]).to_csv(os.path.join(OUT, 'sample_counts.csv'))
    pd.set_option('display.width', 250)
    print(cnt.join(T.set_index('GP')[['n', 'overtakes']]).to_string())
    print(T[['GP', 'const', 'slope', 'n', 'overtakes']].round(4).to_string(index=False))
    P = T.set_index('GP')[[f'p{d:+g}' for d in D]] * 100
    print('\nprobabilidad de adelantamiento (%) por delta predicho (s):'); print(P.round(1).to_string())


if __name__ == '__main__':
    main()
