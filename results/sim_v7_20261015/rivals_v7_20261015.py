"""2026-10-15 — Modelos de rivales del simulador v7.

  rival_pitstop_logit.pkl : LogitStop del simulador (S0, src/models/rival_models.train_pit_stop_logit), re-estimado
                            con la vida del neumatico de 2021 segun F1_TYRELIFE_2021.
  rival_compound_logit.pkl: logit condicional (K1 de results/rival_model_eval_20261014/) en lugar del binario
                            normalizado (K0), como src/models/cond_logit_compound.CondLogitCompoundView. No usa la vida
                            del neumatico.
Control: con F1_TYRELIFE_2021=legacy, S0 debe ser igual al pickle de trained_models/simulation.
Salida: results/sim_v7_20261015/artifacts_<tag>/<GP>/rival_{pitstop,compound}_logit.pkl (mismo objeto en cada GP).
"""
import os, re, sys, pickle
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'results', 'rival_model_eval_20261014'))
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
from models import rival_models as R
from models.cond_logit_compound import CondLogitCompoundView
import rivals_common_20261014 as RC

TAG = os.environ.get('F1_TYRELIFE_2021', 'legacy')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, f'artifacts_{TAG}')
GPS = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
       'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']


def k1_view(Ltr):
    m = RC.K1(Ltr)
    p, se = {}, {}
    for c, b, s in zip(m.cols, m.fit.params.values, m.fit.bse.values):
        if c == 'not_change_compound':
            k = c
        else:
            comp = re.search(r'\[(?:T\.)?(C\d)\]', c).group(1)
            k = f'C(alt)[{comp}]' + (':LapsLeft' if c.endswith(':LapsLeft') else '')
        p[k], se[k] = float(b), float(s)
    p['C(alt)[C3]'] = 0.0; p['C(alt)[C3]:LapsLeft'] = 0.0
    return CondLogitCompoundView(pd.Series(p), pd.Series(se)), m


def main():
    print('factor 2021:', R.TYRELIFE_ADJUST)
    laps = R.preprocess_laps(R.load_lap_data())
    stop, _ = R.train_pit_stop_logit(laps, print_summary=False)
    with open(config.MODELS_DIR / 'simulation' / GPS[0] / 'rival_pitstop_logit.pkl', 'rb') as fh:
        pub = pickle.load(fh)
    d = (stop.params - pub.params).abs().max()
    print(f'S0: {len(stop.params)} parametros; max |dif| con el publicado = {d:.2e}')
    train, test = RC.load()
    view, m = k1_view(RC.long_df(train))
    Lte = RC.long_df(test)
    pv = np.concatenate([view.predict(g) for _, g in Lte.groupby('decision_id', sort=False)])
    pk = pd.Series(m.probs(Lte), index=Lte.index).loc[Lte.groupby('decision_id', sort=False).apply(lambda g: g.index).explode()].values
    print('K1:', view.params.round(4).to_dict())
    print(f'K1 vista vs ConditionalLogit en 2024: max |dif| = {np.max(np.abs(pv - pk.astype(float))):.2e}')
    for gp in GPS:
        os.makedirs(os.path.join(OUT, gp), exist_ok=True)
        with open(os.path.join(OUT, gp, 'rival_pitstop_logit.pkl'), 'wb') as fh:
            pickle.dump(stop, fh)
        with open(os.path.join(OUT, gp, 'rival_compound_logit.pkl'), 'wb') as fh:
            pickle.dump(view, fh)
    T = pd.DataFrame(dict(v7=stop.params, publicado=pub.params))
    T.to_csv(os.path.join(OUT, 'rival_stop_params_vs_published.csv'))
    print(T[T.index.str.contains('TyreLife|Intercept|Early|Late|SC|used|Position')].round(4).to_string())


if __name__ == '__main__':
    main()
