"""Equivalencia de la vida del neumatico de 2021 con la de 2022-23 (2026-10-15).

La vida del neumatico (TyreLife) de 2021 se multiplica por un factor por compuesto antes de estimar los modelos que la
usan (Ec. 1, rivales, adelantamiento, Ec. 2 via regressions.load_data). Unica fuente del factor:
    F1_TYRELIFE_2021 = legacy (por defecto): factores historicos (codigo publicado; reproduce v6 y anteriores)
                       <tag>: data/tyrelife_2021_factors_<tag>.json  ({"C0": f0, ..., "C5": f5})
Diseno y estimacion: docs/DISENO_TYRELIFE_2021_20261015.md, results/tyrelife_2021_20261015/.
Las etiquetas de compuesto de los datos siguen la nomenclatura de 2023 (el C1 de 2021-22 figura como C0).
"""
import json
import os

import config

LEGACY = {'C0': 0.65, 'C1': 1.0, 'C2': 0.90, 'C3': 0.98, 'C4': 0.92, 'C5': 0.92}


def factors(version=None):
    """Factores por compuesto de la version pedida (o de F1_TYRELIFE_2021)."""
    version = version or os.environ.get('F1_TYRELIFE_2021', 'legacy')
    if version == 'legacy':
        return dict(LEGACY)
    with open(os.path.join(config.DATA_DIR, f'tyrelife_2021_factors_{version}.json')) as fh:
        return {k: float(v) for k, v in json.load(fh).items() if k.startswith('C')}


def adjust_2021(laps_2021, f=None):
    """Multiplica TyreLife de las vueltas de 2021 por el factor de su compuesto (en el lugar y devuelve el df).
    Compuestos sin factor (lluvia) quedan en NaN, como en el codigo original (se descartan luego con dropna)."""
    f = factors() if f is None else f
    laps_2021['TyreLife'] = laps_2021['TyreLife'] * laps_2021['Compound_Detail'].map(f)
    return laps_2021
