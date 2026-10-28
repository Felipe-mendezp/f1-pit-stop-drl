"""2026-10-27 — Modelos de rivales del simulador v10 (docs/PROPUESTA_V10_20261027.md, items 1 y 3).

  rival_pitstop_logit.pkl : LogitStop (formula S0 de src/models/rival_models.train_pit_stop_logit) estimado POR EVENTO:
      vueltas verdes como siempre; cada periodo SC/VSC (vueltas consecutivas con SC o VSC, rivals_common_20261014.prep)
      aporta una fila por piloto, su primera vuelta del periodo, con PitIn = 1 si hubo parada real en cualquier vuelta del
      periodo. Coherente con el simulador: los rivales solo paran en la primera vuelta de SC/VSC.
  rival_compound_logit.pkl: K1 (logit condicional) con la muestra de paradas reales.
Parada real = PitIn = 1 sin paso por el pit lane (common_v10_20261027.rival_laps). Requiere F1_TYRELIFE_2021=v10 (v7 con CONTROL=1).
Control: CONTROL=1 (PitIn crudo, sin agregar) reproduce los pickles de simulation_v7_20261015 (no escribe nada).
    F1_TYRELIFE_2021=v10 ~/.venvs/f1-drl-v2/bin/python results/sim_v10_20261027/rivals_v10_20261027.py
Salida: results/sim_v10_20261027/artifacts/<GP>/rival_{pitstop,compound}_logit.pkl y rival_stop_params_vs_v7.csv
"""
import os, re, sys, pickle, importlib.util, warnings; warnings.filterwarnings('ignore')
ROOT = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'results', 'rival_model_eval_20261014'), HERE]
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from models import rival_models as RM
from models.cond_logit_compound import CondLogitCompoundView
import rivals_common_20261014 as RC
import common_v10_20261027 as CV

assert os.environ.get('F1_TYRELIFE_2021') == ('v7' if os.environ.get('CONTROL') == '1' else 'v10eGQ')  # v10e, 'F1_TYRELIFE_2021=v10 (v7 con CONTROL=1)'
CONTROL = os.environ.get('CONTROL') == '1'
OUT = os.path.join(HERE, 'artifacts')
V7 = config.MODELS_DIR / 'simulation_v7_20261015'
GPS = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
       'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']
FORMULA = "PitIn ~ SC + VSC + used_two_compounds + EarlyLaps + LateLaps + TyreLife:C(Compound_Detail) + Position + GP"
spec = importlib.util.spec_from_file_location('r7', os.path.join(ROOT, 'results', 'sim_v7_20261015', 'rivals_v7_20261015.py'))
R7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(R7)


def event_level(d):
    """Verde igual; en cada periodo SC/VSC, una fila por piloto (su primera vuelta del periodo) con PitIn = max."""
    g = d[d.yf_period == 0]
    y = d[d.yf_period > 0].sort_values(['Year', 'GP', 'Driver', 'LapNumber'])
    k = ['Year', 'GP', 'Driver', 'yf_period']
    first = y.groupby(k, sort=False).head(1).copy()
    first['PitIn'] = y.groupby(k, sort=False).PitIn.max().values
    return pd.concat([g, first], ignore_index=True), first


def main():
    raw = RM.load_lap_data() if CONTROL else CV.rival_laps(RM)
    d = RC.prep(raw.drop(columns=['PitIn_raw'], errors='ignore'))
    if CONTROL:
        D = d
    else:
        D, first = event_level(d)
        print(f'filas: {len(d)} -> {len(D)} (periodos SC/VSC: {len(first)} filas piloto-periodo, '
              f'paradas {int(first.PitIn.sum())})')
    stop = smf.logit(FORMULA, data=D).fit(disp=False)
    view, m = R7.k1_view(RC.long_df(d))
    with open(V7 / GPS[0] / 'rival_pitstop_logit.pkl', 'rb') as fh:
        s7 = pickle.load(fh)
    with open(V7 / GPS[0] / 'rival_compound_logit.pkl', 'rb') as fh:
        k7 = pickle.load(fh)
    ds = (stop.params - s7.params).abs().max(); dk = (view.params - k7.params.reindex(view.params.index)).abs().max()
    print(f'LogitStop: max |dif| con v7 = {ds:.2e} | K1: max |dif| con v7 = {dk:.2e} | K1 decisiones: {RC.long_df(d).decision_id.nunique()}')
    if CONTROL:
        return
    for name in ('SC', 'VSC'):
        z = first[(first[name] == 1) & (first.LapNumber > 5)]
        print(f'1a vuelta {name}: prob predicha v10 {stop.predict(z).mean():.3f} | v7 {s7.predict(z).mean():.3f} | '
              f'real (parar en el periodo) {z.PitIn.mean():.3f} (n={len(z)})')
    for gp in GPS:
        os.makedirs(os.path.join(OUT, gp), exist_ok=True)
        with open(os.path.join(OUT, gp, 'rival_pitstop_logit.pkl'), 'wb') as fh:
            pickle.dump(stop, fh)
        with open(os.path.join(OUT, gp, 'rival_compound_logit.pkl'), 'wb') as fh:
            pickle.dump(view, fh)
    T = pd.DataFrame(dict(v7=s7.params, v10=stop.params, se_v10=stop.bse))
    T.to_csv(os.path.join(OUT, 'rival_stop_params_vs_v7.csv'))
    K = pd.DataFrame(dict(v7=k7.params, v10=view.params))
    K.to_csv(os.path.join(OUT, 'rival_compound_params_vs_v7.csv'))
    print(T[T.index.str.contains('Intercept|Early|Late|SC|used|Position')].round(4).to_string())
    print(K.round(4).to_string())


if __name__ == '__main__':
    main()
