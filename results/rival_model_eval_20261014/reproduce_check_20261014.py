"""2026-10-14: control. Reajusta S0 (LogitStop del simulador) y K0 (eleccion de compuesto) con el preprocesamiento de
src/models/rival_models.py y compara con los pickles del simulador (iguales en simulation/ y simulation_v6_20261014/)."""
import os, sys, pickle, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import config, compat; compat.setup_all()
import numpy as np
from models import rival_models as R

laps = R.preprocess_laps(R.load_lap_data())
s0, _ = R.train_pit_stop_logit(laps, print_summary=False)
k0 = R.train_compound_choice_logit(laps, print_summary=False)
from model_loader import ModelLoader
M = ModelLoader().load('Belgian Grand Prix')
for name, fit, f in [('S0', s0, 'rival_pitstop_logit'), ('K0', k0, 'rival_compound_logit')]:
    pub = getattr(M, f)
    common = fit.params.index.intersection(pub.params.index)
    d = (fit.params[common] - pub.params[common]).abs().max()
    print(f'{name}: nobs {int(fit.nobs)} vs pickle {int(pub.nobs)}; params {len(fit.params)} vs {len(pub.params)}, '
          f'comunes {len(common)}, max |diff| = {d:.2e}')
