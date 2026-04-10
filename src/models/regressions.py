"""
Regression functions for lap-time modeling.

Contains data loading, filtering, OLS calibration per GP, and iterative
IQR outlier removal.  The FastLinearPredictor class lives in src/reg.py.
"""
import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from config import DATA_DIR


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """
    Load session CSVs for 2021-2024 and apply the TyreLife adjustment
    for 2021 data.

    Returns:
        tuple: (laps_2021, laps_2022, laps_2023, laps_2024)
    """
    laps_2021 = pd.read_csv(os.path.join(DATA_DIR, 'session_2021_V2.csv'))
    laps_2022 = pd.read_csv(os.path.join(DATA_DIR, 'session_2022_V2.csv'))
    laps_2023 = pd.read_csv(os.path.join(DATA_DIR, 'session_2023_V2.csv'))
    laps_2024 = pd.read_csv(os.path.join(DATA_DIR, 'session_2024_V2.csv'))

    tyrelife_adjust = {'C0': 0.65,
                       'C1': 1.0,
                       'C2': 0.90,
                       'C3': 0.98,
                       'C4': 0.92,
                       'C5': 0.92}

    laps_2021['TyreLife'] = laps_2021['TyreLife'] * laps_2021['Compound_Detail'].map(tyrelife_adjust)

    return laps_2021, laps_2022, laps_2023, laps_2024


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def filter_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina ciertos Grandes Premios y filas con compound WET - INT.

    Args:
    - laps (pd.DataFrame): DataFrame con datos de vueltas de carreras.

    Returns:
    - pd.DataFrame: DataFrame limpio.
    """
    # Eliminamos los GP donde se uso Compound WET o INTERMEDIATE
    laps_filter = laps[~laps['Compound'].isin(['WET', 'INTERMEDIATE'])]
    laps_filter = laps_filter.dropna()

    return laps_filter


def all_laps(outliers=False):
    """
    Concatenate 2021-2023 laps, apply transformations, and optionally
    remove outliers (laps > median + 15 s).
    """
    laps_2021, laps_2022, laps_2023, _laps_2024 = load_data()

    # Concatenamos las vuelta del 2021 al 2023 (NO incluir 2024)
    laps_2021_2022_2023 = pd.concat([laps_2021, laps_2022, laps_2023])
    laps_2021_2022_2023 = filter_laps(laps_2021_2022_2023)

    laps_2021_2022_2023['Interval_front_real'] = laps_2021_2022_2023['Interval_front'].copy()
    # DRS
    laps_2021_2022_2023['DRS'] = (
        ((laps_2021_2022_2023['Interval_front'] < 1.0)) &
        (laps_2021_2022_2023['LapNumber'] > 2)).astype(int)

    # IMPORTANTE: Guardar interval_front_real ANTES de la transformacion
    laps_2021_2022_2023['interval_front_real'] = laps_2021_2022_2023['Interval_front'].copy()

    # exp de los intervalos
    laps_2021_2022_2023['Interval_front'] = np.exp(-laps_2021_2022_2023['Interval_front'])
    laps_2021_2022_2023['Interval_behind'] = np.exp(-laps_2021_2022_2023['Interval_behind'])
    laps_2021_2022_2023['LapNumber_2'] = laps_2021_2022_2023['LapNumber'] ** 2

    first_lap = laps_2021_2022_2023['LapNumber'] == 1.0
    laps_2021_2022_2023['FirstLap_pos'] = laps_2021_2022_2023['Position'] * first_lap


    if outliers == False:
        # Calcular la media de LapTime por GP y Year
        mean_lap_times = laps_2021_2022_2023.groupby(['Year', 'GP'])['LapTime'].transform('median')

        # Condicion: LapTime 15s mayor a la media
        too_slow = laps_2021_2022_2023['LapTime'] >= mean_lap_times + 15

        # Condicion para excepciones
        exceptions = (laps_2021_2022_2023['PitIn'] == 1.0) | (laps_2021_2022_2023['PitOut'] == 1.0) | (laps_2021_2022_2023['LapNumber'] == 1)

        # Filtrar: eliminar solo los casos que son muy lentos y **no** cumplen una excepcion
        laps_2021_2022_2023 = laps_2021_2022_2023[~(too_slow & ~exceptions)]

    return laps_2021_2022_2023


# ---------------------------------------------------------------------------
# OLS calibration
# ---------------------------------------------------------------------------

def calibrar_regresion_gp(data_gp):
    """
    Ajusta una regresion lineal para LapTime en un GP especifico usando statsmodels.
    Elimina la primera categoria de Driver y Compound_Detail para evitar multicolinealidad.

    Parameters
    ----------
    data_gp : pandas.DataFrame
        DataFrame con los datos de vueltas ya filtrados por GP y TrackStatus.

    Returns
    -------
    modelo : statsmodels.regression.linear_model.RegressionResultsWrapper
        Modelo calibrado final.
    """
    if data_gp.empty:
        raise ValueError("No hay datos para calibrar la regresion")

    # convertir Year a string para tratarlo como categorica
    data_gp = data_gp.copy()
    data_gp['Year'] = data_gp['Year'].astype(str)

    # referencias para categoricas
    driver_ref = 'VER'

    # compuesto mas usado
    compound_ref = data_gp["Compound_Detail"].value_counts().idxmax()

    # menor year
    year_ref = sorted(data_gp["Year"].unique())[0]

    # formula de regresion
    formula = (
        "LapTime ~ LapNumber + LapNumber_2 + FirstLap_pos + Interval_front + Interval_behind "
        "+ PitIn + PitOut + DRS "
        f"+ C(Driver, Treatment(reference='{driver_ref}')) "
        f"+ C(Compound_Detail, Treatment(reference='{compound_ref}')) "
        f"+ C(Year, Treatment(reference='{year_ref}')) "
        f"+ TyreLife:C(Compound_Detail)"
    )
    modelo = smf.ols(formula=formula, data=data_gp).fit()
    return modelo, data_gp


def filtrar_outliers_iqr(df, gp_name, k=3, max_iter=10):
    """
    Filtra iterativamente outliers usando el metodo IQR sobre los residuos de la regresion
    y recalibra el modelo con los datos filtrados.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con los datos de vueltas.
    gp_name : str
        Nombre del GP sobre el cual filtrar y calibrar la regresion.
    k : float
        Multiplicador del IQR para filtrar outliers.
    max_iter : int
        Numero maximo de iteraciones para evitar loops infinitos.

    Returns
    -------
    modelo : statsmodels.regression.linear_model.RegressionResultsWrapper
        Modelo calibrado final despues de filtrar outliers.
    """
    # filtrar por GP y TrackStatus, luego resetear indice
    data_gp = df[(df["GP"] == gp_name) & (df['TrackStatus'] == 1.0)].copy().reset_index(drop=True)
    if data_gp.empty:
        raise ValueError(f"No hay datos para el GP '{gp_name}'")


    counter = 0
    while True:
        modelo, _ = calibrar_regresion_gp(data_gp)
        residuals = modelo.resid

        q3, q1 = np.percentile(residuals, [75, 25])
        iqr = k * (q3 - q1)
        upper_bound = q3 + iqr
        lower_bound = q1 - iqr

        mask = (residuals >= lower_bound) & (residuals <= upper_bound)

        if mask.all() or counter >= max_iter:
            break
        else:
            data_gp = data_gp.loc[residuals.index[mask]].reset_index(drop=True)
            counter += 1
    print(f"\n{gp_name}")
    print(f"Iteraciones metodo inter: {counter}")

    return modelo, data_gp
