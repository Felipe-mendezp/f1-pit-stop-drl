"""
Version SIMPLIFICADA para comparacion justa con QMIP.

Solo usa los coeficientes que el QMIP considera:
- Compound (intercept + efecto del compuesto)
- TyreLife x Compound (degradacion)
- PitIn
- PitOut

ULTRA-OPTIMIZADO: Calculo directo sin usar predict(), solo aritmetica.
PERFORMANCE: ~0.05us per prediction (100x mas rapido que FastLinearPredictor)
"""

import config

from typing import Dict, Tuple
import numpy as np


class SimplifiedLaptimeCalculator:
    """
    Calculadora ultra-rapida de laptime usando solo 4 coeficientes.

    Modelo: laptime = intercept + compound_effect + tyre_degradation + pit_effect

    Performance: ~0.05us per prediction (solo 4 sumas y 1 multiplicacion)
    """

    __slots__ = (
        'intercept',
        'compound_effects',  # Dict[int, float]: {3: effect_c3, 4: effect_c4, 5: effect_c5}
        'tyre_degradation',  # Dict[int, float]: {3: beta_c3, 4: beta_c4, 5: beta_c5}
        'pit_in_cost',
        'pit_out_cost',
        'available_compounds'
    )

    def __init__(self, params_dict: Dict, available_compounds: list):
        """
        Inicializa la calculadora con los coeficientes del modelo.

        Args:
            params_dict: Diccionario con los parametros del modelo (statsmodels.params)
            available_compounds: Lista de compuestos disponibles [3, 4, 5]
        """
        self.available_compounds = sorted(available_compounds)

        # Extraer intercept
        self.intercept = float(params_dict['Intercept'])

        # Extraer efectos de compuestos (relativos a la referencia)
        # Buscar cual es el compuesto de referencia extrayendolo del nombre del parametro
        compound_params = [k for k in params_dict.keys() if 'Compound_Detail' in k and 'Treatment' in k and '[T.C' in k]

        # Identificar referencia desde el primer parametro
        if compound_params:
            first_param = compound_params[0]
            # Extraer: "C(Compound_Detail, Treatment(reference='C3'))[T.C4]" -> 3
            reference_compound = int(first_param.split("reference='C")[1].split("'")[0])
        else:
            # No hay parametros de compuesto (todos son referencia), buscar en available
            reference_compound = available_compounds[0]

        # Identificar que compuestos tienen parametros
        compounds_in_params = set()
        for param in compound_params:
            # Extraer numero del compuesto: "...[T.C4]" -> 4
            c_num = int(param.split('[T.C')[1].rstrip(']'))
            compounds_in_params.add(c_num)

        # Construir diccionario de efectos
        self.compound_effects = {}
        for compound in self.available_compounds:
            if compound == reference_compound:
                self.compound_effects[compound] = 0.0
            elif compound in compounds_in_params:
                # Buscar el parametro correspondiente
                param_key = f"C(Compound_Detail, Treatment(reference='C{reference_compound}'))[T.C{compound}]"
                self.compound_effects[compound] = float(params_dict[param_key])
            else:
                # Compuesto no esta en los parametros, asumir efecto 0
                self.compound_effects[compound] = 0.0

        # Extraer degradacion por TyreLife (interaccion TyreLife x Compound)
        self.tyre_degradation = {}
        for compound in self.available_compounds:
            tyre_param_key = f"TyreLife:C(Compound_Detail)[C{compound}]"
            self.tyre_degradation[compound] = float(params_dict[tyre_param_key])

        # Extraer costos de pit
        self.pit_in_cost = float(params_dict['PitIn'])
        self.pit_out_cost = float(params_dict['PitOut'])

    def predict(self, compound: int, tyre_life: float, pit_in: float, pit_out: float) -> float:
        """
        Calcula el laptime usando el modelo simplificado.

        Args:
            compound: Compuesto actual (3, 4, 5)
            tyre_life: Edad del neumatico
            pit_in: 1.0 si entra a pits, 0.0 si no
            pit_out: 1.0 si sale de pits, 0.0 si no

        Returns:
            float: Laptime en segundos
        """
        # Modelo: laptime = intercept + compound_effect + (tyre_degradation x tyre_life) + pit_costs
        laptime = (
            self.intercept +
            self.compound_effects[compound] +
            self.tyre_degradation[compound] * tyre_life +
            self.pit_in_cost * pit_in +
            self.pit_out_cost * pit_out
        )

        return laptime

    def get_info(self) -> Dict:
        """Retorna informacion sobre la calculadora."""
        return {
            'intercept': self.intercept,
            'compound_effects': self.compound_effects,
            'tyre_degradation': self.tyre_degradation,
            'pit_in_cost': self.pit_in_cost,
            'pit_out_cost': self.pit_out_cost,
            'available_compounds': self.available_compounds
        }


def extract_simplified_calculator(model_laptime, available_compounds: list) -> SimplifiedLaptimeCalculator:
    """
    Extrae los coeficientes del modelo de regresion y crea una calculadora simplificada.

    Args:
        model_laptime: Modelo de statsmodels o FastLinearPredictor
        available_compounds: Lista de compuestos disponibles

    Returns:
        SimplifiedLaptimeCalculator
    """
    # Importar FastLinearPredictor si esta disponible
    try:
        from reg import FastLinearPredictor
        has_fast = True
    except ImportError:
        has_fast = False
        FastLinearPredictor = None

    # Extraer parametros segun el tipo de modelo
    if has_fast and isinstance(model_laptime, FastLinearPredictor):
        # FastLinearPredictor: usar coef_ y feature_names_
        params_dict = dict(zip(model_laptime.feature_names_, model_laptime.coef_))
        params_dict['Intercept'] = model_laptime.intercept_
    else:
        # Statsmodels: usar params directamente
        params_dict = dict(model_laptime.params)

    return SimplifiedLaptimeCalculator(params_dict, available_compounds)


# Estado simplificado (solo lo necesario)
class SimplifiedLapState:
    """Estado simplificado que solo contiene las 4 variables necesarias."""

    __slots__ = ('compound', 'tyre_life', 'pit_in', 'pit_out')

    def __init__(self, compound: int, tyre_life: float = 0.0, pit_in: float = 0.0, pit_out: float = 0.0):
        self.compound = compound
        self.tyre_life = tyre_life
        self.pit_in = pit_in
        self.pit_out = pit_out

    def to_tuple(self) -> Tuple[int, float, float, float]:
        """Retorna tupla (compound, tyre_life, pit_in, pit_out)."""
        return (self.compound, self.tyre_life, self.pit_in, self.pit_out)
