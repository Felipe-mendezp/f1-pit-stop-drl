import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
import compat; compat.setup_all()  # shims necesarios para despicklear FastLinearPredictor
from tyrelife_2021 import adjust_2021

"""
Logit de adelantamientos PIT-AWARE (propuesta del autor, 2026-08-07).

Version del autor de logit_overtakes.py con el re-etiquetado que captura la
entrada a pits del rival:

  1. La ETIQUETA ya no excluye los PitIn/PitOut del rival: una ganancia de
     posicion sobre un rival en ciclo de pit cuenta como overtake=1.
  2. rival_PitIn se evalua en la vuelta L (el ingreso ocurre al final de esa
     vuelta con poca perdida) y rival_PitOut en L+1 (donde se consume ~20s),
     la misma vuelta que usa delta_t_lap.
  3. La muestra se filtra por el gap PREDICHO delta_total en [-30, 5] en vez
     del gap real <= 1.0s: entran los casos en que el rival de adelante pierde
     mucho tiempo (PitOut) aunque el perseguidor venga lejos.
  4. El termino pitout_prox = rival_PitOut * exp(-|delta_total|) esta definido
     pero EXCLUIDO del ajuste (comentado, decision del autor): la informacion
     de pit entra via el re-etiquetado + la pendiente por GP, no via dummy.

El modelo final por GP sigue siendo lineal en delta_total (const + slope), asi
que es drop-in para el simulador (mismo fast path de 2 parametros).

Salidas (NADA del arbol original se toca):
  - trained_models/simulation_pitlogit_20260807/<GP>/overtake_logit.pkl (+ threshold)
    en un espejo por symlinks del arbol original -> usar via F1_SIMULATION_DIR.
  - results/overtake_pitlogit_20260807/ (tabla de probabilidades, heatmap, taus).

Uso:
    python src/models/logit_overtakes_pit.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle

from reg import FastLinearPredictor
# Clase a nivel de modulo para que los pickles por GP sean despickleables
# desde el env (una clase local a una funcion no lo es).
from models.overtake_pit_view import PerGPLogitView

NEW_TREE = os.path.join(str(config.MODELS_DIR), 'simulation_pitlogit_20260807')
OUT_DIR = os.path.join(str(config.ROOT_DIR), 'results', 'overtake_pitlogit_20260807')


def load_data():
    """
    Load session CSVs, apply TyreLife adjustment, load FastLinearPredictors,
    and prepare the concatenated + filtered DataFrame.

    Returns:
        tuple: (laps_2021_2022_2023, selected_GP, reg_clear_dd)
    """
    # Cargamos los dataframes
    laps_2021 = pd.read_csv(str(config.DATA_DIR / 'session_2021_V2.csv'))
    laps_2022 = pd.read_csv(str(config.DATA_DIR / 'session_2022_V2.csv'))
    laps_2023 = pd.read_csv(str(config.DATA_DIR / 'session_2023_V2.csv'))

    adjust_2021(laps_2021)                       # factor de 2021: src/tyrelife_2021.py

    selected_GP = {
            'Bahrain Grand Prix': ['C1', 'C2', 'C3'],
            'Belgian Grand Prix': ['C2', 'C3', 'C4'],
            'Dutch Grand Prix': ['C1', 'C2', 'C3'],
            'Emilia Romagna Grand Prix': ['C3', 'C4', 'C5'],
            'Hungarian Grand Prix': ['C3', 'C4', 'C5'],
            'Miami Grand Prix': ['C2', 'C3', 'C4'],
            'Saudi Arabian Grand Prix': ['C2', 'C3', 'C4'],
            'Singapore Grand Prix': ['C3', 'C4', 'C5'],
            'United States Grand Prix': ['C2', 'C3', 'C4']
            }

    # Cargar FastLinearPredictor en lugar de modelos statsmodels
    # (siempre desde el arbol ORIGINAL: el nuevo arbol es un espejo en
    # construccion durante esta corrida)
    orig_sim = str(config.MODELS_DIR / 'simulation')
    reg_clear_dd = {}
    for gp in selected_GP.keys():
        with open(os.path.join(orig_sim, f'{gp}_fast_predictor.pkl'), "rb") as f:
            fast_pred = pickle.load(f)
            reg_clear_dd[gp] = fast_pred

    # Concatenamos las vuelta del 2021 al 2023
    laps_2021_2022_2023 = pd.concat([laps_2021, laps_2022, laps_2023])
    laps_2021_2022_2023 = filter_laps(laps_2021_2022_2023)

    # Guardar Interval_front original ANTES de transformar (necesario para logit)
    laps_2021_2022_2023['interval_front_real'] = laps_2021_2022_2023['Interval_front'].copy()

    # Calcular DRS (1 si esta a menos de 1 segundo y no es lap 1 o 2)
    laps_2021_2022_2023['DRS'] = (
        (laps_2021_2022_2023['Interval_front'] < 1.0) &
        (laps_2021_2022_2023['LapNumber'] > 2)
    ).astype(int)

    # Transformar intervalos a escala exponencial (para el modelo de regresion)
    laps_2021_2022_2023['Interval_front'] = np.exp(-laps_2021_2022_2023['Interval_front'])
    laps_2021_2022_2023['Interval_behind'] = np.exp(-laps_2021_2022_2023['Interval_behind'])

    # Variables adicionales para el modelo
    laps_2021_2022_2023['LapNumber_2'] = laps_2021_2022_2023['LapNumber'] ** 2
    first_lap = laps_2021_2022_2023['LapNumber'] == 1.0
    laps_2021_2022_2023['FirstLap_pos'] = laps_2021_2022_2023['Position'] * first_lap

    return laps_2021_2022_2023, selected_GP, reg_clear_dd


def filter_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina ciertos Grandes Premios y filas con TrackStatus erroneo.
    """
    # Eliminamos los GP donde se uso Compound WET o INTERMEDIATE
    laps_filter = laps[~laps['Compound'].isin(['WET', 'INTERMEDIATE'])].copy()

    # Keep only compounds C1-C5 (exclude C0)
    laps_filter = laps_filter[laps_filter['Compound_Detail'].isin(['C1', 'C2', 'C3', 'C4', 'C5'])].copy()

    # Compute the maximum lap number per (Year, GP, Driver)
    max_laps = laps_filter.groupby(['Year', 'GP', 'Driver'])['LapNumber'].transform('max')

    # Crear la columna 'LapsLeft'
    laps_filter['LapsLeft'] = max_laps - laps_filter['LapNumber']

    return laps_filter


def optimize_f1_analysis(laps_2021_2022_2023, selected_GP, reg_clear_dd):
    """
    Optimized function for analyzing F1 laps.
    """

    # 1. PREPARACION DE DATOS CON ENCODING EFICIENTE
    print("Preparing data and creating dummy variables...")
    all_laps_df = laps_2021_2022_2023.copy()

    # Guardar columnas originales antes de crear dummies (necesarias para merge
    # y, CRITICO, para FastLinearPredictor: _build_design_vector resuelve los
    # terminos C(Compound_Detail) y TyreLife:C(Compound_Detail) leyendo la
    # clave semantica 'Compound_Detail'. Si no se restaura, TODOS los efectos
    # de compuesto y degradacion valen 0 en LapTime_pred y delta_total pierde
    # la senal fisica que gobierna los adelantamientos.
    all_laps_df['Year_original'] = all_laps_df['Year'].copy()
    all_laps_df['Driver_original'] = all_laps_df['Driver'].copy()
    all_laps_df['Compound_Detail_original'] = all_laps_df['Compound_Detail'].copy()

    # Crear todas las variables dummy de una vez
    dummy_columns = ["Driver", "Compound_Detail", "Year"]
    all_laps_df = pd.get_dummies(
        all_laps_df,
        columns=dummy_columns,
        prefix=dummy_columns,
        dtype=int
    )

    # Restaurar columnas originales con sus nombres
    all_laps_df['Year'] = all_laps_df['Year_original']
    all_laps_df['Driver'] = all_laps_df['Driver_original']
    all_laps_df['Compound_Detail'] = all_laps_df['Compound_Detail_original']
    all_laps_df = all_laps_df.drop(columns=['Year_original', 'Driver_original', 'Compound_Detail_original'])

    # Build LapNumber_square (PitIn and PitOut already exist in the data)
    all_laps_df['LapNumber_square'] = all_laps_df['LapNumber'] ** 2

    # Variable FirstLap_pos (ya se calculo antes, pero aseguramos que existe)
    if 'FirstLap_pos' not in all_laps_df.columns:
        first_lap_mask = (all_laps_df['LapNumber'] == 1.0)
        all_laps_df['FirstLap_pos'] = all_laps_df['Position'].where(first_lap_mask, 0)

    # Crear variables TyreLife_Compound de forma vectorizada
    compound_cols = [col for col in all_laps_df.columns if col.startswith("Compound_Detail_")]
    for col in compound_cols:
        compound_name = col.split("_", 2)[-1]
        all_laps_df[f"TyreLife_{compound_name}"] = all_laps_df["TyreLife"] * all_laps_df[col]


    # 2. PREDICCIONES OPTIMIZADAS
    print("Calculando predicciones...")
    relevant_data = all_laps_df[all_laps_df['GP'].isin(selected_GP)].copy()
    relevant_data['LapTime_pred'] = np.nan

    for gp in selected_GP:
        if gp not in reg_clear_dd:
            continue

        gp_mask = relevant_data['GP'] == gp
        gp_data = relevant_data[gp_mask]

        if gp_data.empty:
            continue

        try:
            fast_pred = reg_clear_dd[gp]
            data_list = gp_data.to_dict('records')
            predictions = fast_pred.predict_batch(data_list)
            relevant_data.loc[gp_mask, 'LapTime_pred'] = predictions
        except Exception as e:
            print(f"Error al predecir para GP {gp}: {e}")
            continue


    # 3. PREPARACION DEL DATASET FINAL
    print("Preparando dataset final...")
    logit_df = laps_2021_2022_2023[laps_2021_2022_2023['GP'].isin(selected_GP)].copy()

    pred_df = relevant_data[['Year', 'GP', 'Driver', 'LapNumber', 'LapTime_pred']].copy()
    logit_df = logit_df.merge(pred_df, on=['Year', 'GP', 'Driver', 'LapNumber'], how='left')

    logit_df = logit_df.sort_values(['Year', 'GP', 'Driver', 'LapNumber']).reset_index(drop=True)


    # 4. CALCULO DE METRICAS OPTIMIZADO
    print("Calculando metricas de posicion...")

    last_laps = logit_df.groupby(['Year', 'GP', 'Driver'])['LapNumber'].max().reset_index()
    last_laps.rename(columns={'LapNumber': 'LastLap'}, inplace=True)
    logit_df = logit_df.merge(last_laps, on=['Year', 'GP', 'Driver'])

    logit_df['next_position'] = logit_df.groupby(['Year', 'GP', 'Driver'])['Position'].shift(-1)
    logit_df['position_gain/loss'] = logit_df['Position'] - logit_df['next_position']
    logit_df['position_gain/loss'] = logit_df['position_gain/loss'].fillna(0)


    # 5. DETECCION DE ADELANTAMIENTOS OPTIMIZADA
    print("Detectando adelantamientos...")
    logit_df['overtake'] = 0

    overtake_candidates = logit_df[
        (logit_df['LapNumber'] > 1) &
        (logit_df['position_gain/loss'] > 0) &
        (logit_df['TrackStatus'] == 1.0) &
        (logit_df['next_position'].notna())
    ].copy()

    if not overtake_candidates.empty:
        logit_indexed = logit_df.set_index(['Year', 'GP', 'LapNumber', 'Position'])

        for idx, row in overtake_candidates.iterrows():
            year, gp, lap = row['Year'], row['GP'], row['LapNumber']
            pos_act, next_pos = int(row['Position']), int(row['next_position'])

            try:
                # Buscar rival que fue adelantado
                rival_key = (year, gp, lap, pos_act - 1)
                if rival_key not in logit_indexed.index:
                    continue

                rival = logit_indexed.loc[rival_key]
                if isinstance(rival, pd.DataFrame):
                    rival = rival.iloc[0]

                # Verificar condiciones de adelantamiento valido
                # No se excluyen los casos de PitIn/PitOut del rival: esas ganancias
                # de posicion tambien cuentan como overtake=1, y es rival_PitIn /
                # rival_PitOut (covariables del logit) lo que le permite al modelo
                # distinguir ese regimen del pase normal en pista.
                if (rival['next_position'] >= next_pos and
                    rival['LastLap'] != lap):

                    logit_df.at[idx, 'overtake'] = 1

            except (KeyError, IndexError):
                continue


    # 6. CALCULO DE DELTA TIEMPO OPTIMIZADO
    print("Calculando deltas de tiempo...")
    logit_df['delta_t_lap'] = np.nan
    logit_df['rival_PitIn'] = np.nan
    logit_df['rival_PitOut'] = np.nan

    for (year, gp), group in logit_df.groupby(['Year', 'GP']):
        if group.empty:
            continue

        lap_times_dict = group.set_index(['Driver', 'LapNumber'])['LapTime_pred'].to_dict()
        position_dict = group.set_index(['LapNumber', 'Position'])['Driver'].to_dict()
        pitin_dict = group.set_index(['Driver', 'LapNumber'])['PitIn'].to_dict()
        pitout_dict = group.set_index(['Driver', 'LapNumber'])['PitOut'].to_dict()

        non_leaders = group[group['Position'] > 1].copy()

        for idx, row in non_leaders.iterrows():
            driver, lap, pos = row['Driver'], row['LapNumber'], int(row['Position'])

            rival_key = (lap, pos - 1)
            if rival_key not in position_dict:
                continue

            rival_driver = position_dict[rival_key]
            next_lap = lap + 1

            # PitIn/PitOut del rival de adelante (con quien se evalua el adelantamiento).
            # PitIn se evalua en la vuelta actual (L): el ingreso a boxes ocurre al
            # final de esa vuelta con poca perdida de tiempo. PitOut se evalua en la
            # vuelta siguiente (L+1), que es donde realmente se consume el tiempo de
            # la parada (~20s) y donde el rival cae posiciones; ademas next_lap es la
            # misma vuelta que usa delta_t_lap para comparar los tiempos predichos.
            logit_df.at[idx, 'rival_PitIn'] = pitin_dict.get((rival_driver, lap), np.nan)
            logit_df.at[idx, 'rival_PitOut'] = pitout_dict.get((rival_driver, next_lap), np.nan)

            my_time = lap_times_dict.get((driver, next_lap))
            rival_time = lap_times_dict.get((rival_driver, next_lap))

            if pd.notna(my_time) and pd.notna(rival_time):
                logit_df.at[idx, 'delta_t_lap'] = my_time - rival_time

    print("Procesamiento completado!")
    return logit_df


def enhanced_logit_modeling_with_evaluation(logit_df, min_samples=30, min_positives=5,
                                          show_plots=False, save_plots=True,
                                          plot_dir=None):
    """
    Ajuste pooled (const y delta_total por GP) sobre la muestra re-etiquetada,
    con evaluacion por GP. Ver docstring del modulo.
    """
    plot_dir = plot_dir or os.path.join(OUT_DIR, 'plots/')
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    # 1. LIMPIEZA DE DATOS
    print("Preparing data...")
    logit_df_clean = logit_df.dropna(subset=['interval_front_real', 'delta_t_lap', 'DRS', 'overtake',
                                              'rival_PitOut']).copy()

    logit_df_clean['delta_total'] = logit_df_clean['interval_front_real'] + logit_df_clean['delta_t_lap']

    # Filtrar por el gap PREDICHO (delta_total) en vez del gap real actual:
    # esto captura tambien los casos en que el auto viene lejos pero se predice
    # que el rival de adelante pierde mucho tiempo (p.ej. PitOut), quedando el
    # auto que persigue hasta 20s "adelante" (delta_total muy negativo) o hasta
    # 1.5s detras del rival.
    logit_df_clean = logit_df_clean[
        (logit_df_clean['delta_total'] >= -30) & (logit_df_clean['delta_total'] <= 5.0)
    ]

    gps = logit_df_clean['GP'].unique()

    results = {
        'models': {},
        'metrics': {},
        'failed_gps': [],
        'gp_stats': {},
        'pooled_model': None,
    }

    # 3. AJUSTE DEL MODELO POOLED (todos los GP juntos)
    # rival_PitOut entra como rival_PitOut * exp(-|delta_total|): la idea es
    # que el rival este en PitOut solo deberia importar cuando el delta
    # pronosticado es chico en magnitud (el resultado esta indefinido); si
    # |delta_total| es grande, el signo de delta_total ya determina casi todo
    # el resultado y el termino se apaga solo. Coeficiente unico (pooled),
    # compartido entre todos los GP. El intercepto y la pendiente de
    # delta_total SI quedan especificos por GP.
    # NOTA (decision del autor): pitout_prox queda EXCLUIDO del ajuste.
    print("Ajustando modelo logit pooled (const y delta_total por GP)...")

    logit_df_clean['pitout_prox'] = (
        logit_df_clean['rival_PitOut'] * np.exp(-logit_df_clean['delta_total'].abs())
    )

    gp_dummies = pd.get_dummies(logit_df_clean['GP'], prefix='GP', dtype=float)
    gp_slope = gp_dummies.multiply(logit_df_clean['delta_total'], axis=0)
    gp_slope.columns = [f"{c}_delta_total" for c in gp_dummies.columns]

    # X_pooled = pd.concat([gp_dummies, gp_slope, logit_df_clean[['pitout_prox']]], axis=1)
    X_pooled = pd.concat([gp_dummies, gp_slope], axis=1)
    y_pooled = logit_df_clean['overtake']

    pooled_model = sm.Logit(y_pooled, X_pooled).fit(
        disp=False, maxiter=1000, method='bfgs', warn_convergence=False
    )
    print("STATSMODELS (pooled): Convergencia exitosa" if pooled_model.mle_retvals['converged']
          else "STATSMODELS (pooled): No convergio")
    results['pooled_model'] = pooled_model

    print(f"\n{'='*80}")
    print("MODELADO Y EVALUACION POR GP")
    print(f"{'='*80}")

    for gp in gps:
        gp_df = logit_df_clean[logit_df_clean['GP'] == gp].copy()

        print(f"\n{'='*25} {gp} {'='*25}")

        n_total = len(gp_df)
        n_overtakes = gp_df['overtake'].sum()
        overtake_rate = n_overtakes / n_total if n_total > 0 else 0

        print(f"Muestras: {n_total} | Overtakes: {n_overtakes} | Tasa: {overtake_rate:.1%}")

        results['gp_stats'][gp] = {
            'total_samples': n_total,
            'overtakes': n_overtakes,
            'overtake_rate': overtake_rate,
            'delta_total_mean': gp_df['delta_total'].mean(),
            'delta_total_std': gp_df['delta_total'].std(),
        }

        if n_total < min_samples:
            print(f"SALTADO: Muy pocas muestras ({n_total} < {min_samples})")
            results['failed_gps'].append((gp, "insufficient_samples"))
            continue

        if n_overtakes < min_positives:
            print(f"SALTADO: Muy pocos overtakes ({n_overtakes} < {min_positives})")
            results['failed_gps'].append((gp, "insufficient_positives"))
            continue

        # Preparar variables (mismas columnas que uso el modelo pooled)
        X = gp_df[['delta_total', 'pitout_prox']].copy()
        y = gp_df['overtake'].copy()

        # Extraer la vista de este GP del modelo pooled: const y delta_total
        # propios de este GP.
        gp_dummy_col = f"GP_{gp}"
        gp_slope_col = f"GP_{gp}_delta_total"
        params = pd.Series({
            'const': pooled_model.params[gp_dummy_col],
            'delta_total': pooled_model.params[gp_slope_col],
        })
        pvalues = pd.Series({
            'const': pooled_model.pvalues[gp_dummy_col],
            'delta_total': pooled_model.pvalues[gp_slope_col],
        })
        model = PerGPLogitView(params, pvalues)
        method_used = "statsmodels_pooled"

        # 4. PREDICCIONES Y EVALUACION
        try:
            X_pred = sm.add_constant(X)
            y_pred_prob = model.predict(X_pred)

            # Metodo de umbral adaptativo para balancear predicciones
            n_real_overtakes = int(y.sum())

            if n_real_overtakes > 0 and n_real_overtakes < len(y_pred_prob):
                sorted_probs = np.sort(y_pred_prob)[::-1]  # Descendente
                threshold = sorted_probs[n_real_overtakes - 1]
                threshold = max(threshold - 1e-10, 0.001)
            else:
                threshold = 0.5

            print(f"Umbral adaptativo: {threshold:.4f}")

            y_pred = (y_pred_prob >= threshold).astype(int)

            # 5. METRICAS DE EVALUACION
            cm = confusion_matrix(y, y_pred)
            tn, fp, fn, tp = cm.ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            try:
                auc = roc_auc_score(y, y_pred_prob) if len(np.unique(y)) > 1 else np.nan
            except Exception:
                auc = np.nan

            metrics = {
                'method': method_used,
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'confusion_matrix': cm,
                'predicted_overtakes': int(y_pred.sum()),
                'actual_overtakes': n_real_overtakes
            }

            results['models'][gp] = model
            results['metrics'][gp] = metrics

            print(f"Metricas:")
            print(f"   Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
            print(f"   Accuracy: {accuracy:.3f} | AUC: {auc:.3f}")
            print(f"   Overtakes reales: {n_real_overtakes} | Predichos: {int(y_pred.sum())}")

            print(f"Coeficientes ({method_used}):")
            for param, coef in model.params.items():
                pval = model.pvalues[param]
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                print(f"   {param}: {coef:.4f} (p={pval:.4f}) {sig}")

            # 7. VISUALIZACION
            if save_plots:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                           xticklabels=['No Overtake', 'Overtake'],
                           yticklabels=['No Overtake', 'Overtake'],
                           ax=axes[0])
                axes[0].set_title(f'Matriz de Confusion - {gp}')
                axes[0].set_xlabel('Predicho')
                axes[0].set_ylabel('Real')

                axes[1].hist(y_pred_prob[y == 0], bins=20, alpha=0.7, label='No Overtake', color='lightblue')
                axes[1].hist(y_pred_prob[y == 1], bins=20, alpha=0.7, label='Overtake', color='orange')
                axes[1].axvline(threshold, color='red', linestyle='--', label=f'Umbral: {threshold:.3f}')
                axes[1].set_xlabel('Probabilidad Predicha')
                axes[1].set_ylabel('Frecuencia')
                axes[1].set_title(f'Distribucion de Probabilidades - {gp}')
                axes[1].legend()

                plt.tight_layout()
                plt.savefig(f"{plot_dir}evaluation_{gp.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
                plt.close(fig)

        except Exception as e:
            print(f"ERROR EN EVALUACION: {str(e)[:50]}...")
            results['failed_gps'].append((gp, f"evaluation_failed: {e}"))

    # 8. RESUMEN FINAL
    print(f"\n{'='*80}")
    print("RESUMEN FINAL")
    print(f"{'='*80}")
    print(f"Modelos exitosos: {len(results['models'])}")
    print(f"Modelos fallidos: {len(results['failed_gps'])}")

    if results['models']:
        avg_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
            values = [m[metric] for m in results['metrics'].values() if not np.isnan(m[metric])]
            if values:
                avg_metrics[metric] = np.mean(values)

        print(f"\nMetricas Promedio:")
        for metric, value in avg_metrics.items():
            print(f"   {metric.capitalize()}: {value:.3f}")

    if results['failed_gps']:
        print(f"\nGPs Fallidos:")
        for gp, reason in results['failed_gps']:
            print(f"   {gp}: {reason}")

    return results


def overtake_probability_table(results, delta_min=-20, delta_max=20):
    """
    Tabla GP x estado_rival x delta_total con la probabilidad de overtake
    predicha por la vista por GP del modelo pooled.
    """
    deltas = list(range(delta_min, delta_max + 1))
    deltas_arr = np.array(deltas)
    cases = {
        'normal': 0,
        'rival_pitout': 1,
    }

    rows = []
    for gp, model in results['models'].items():
        params = model.params
        const = params.get('const', 0.0)
        b_delta = params.get('delta_total', 0.0)
        b_prox = params.get('pitout_prox', 0.0)

        for case_name, pitout in cases.items():
            pitout_prox = pitout * np.exp(-np.abs(deltas_arr))
            linear = const + b_delta * deltas_arr + b_prox * pitout_prox
            probs = 1 / (1 + np.exp(-linear))
            row = {'GP': gp, 'rival_status': case_name}
            row.update(zip(deltas, probs))
            rows.append(row)

    table = pd.DataFrame(rows)
    return table


def plot_overtake_probability_heatmap(table, deltas=None, save_path=None):
    """Heatmap (GP x estado_rival) x delta_total con la probabilidad predicha."""
    if deltas is None:
        deltas = [-20, -10, -5, -1, 0, 1, 5, 10, 20]
        deltas = [d for d in deltas if d in table.columns]

    status_label = {'normal': 'Normal', 'rival_pitout': 'Rival en PitOut'}
    plot_df = table.copy()
    plot_df['row_label'] = (
        plot_df['GP'] + ' - ' + plot_df['rival_status'].map(status_label)
    )
    plot_df = plot_df.set_index('row_label')[deltas] * 100

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(plot_df) + 1.5))
    sns.heatmap(
        plot_df,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Probabilidad de overtake (%)'},
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )
    ax.set_xlabel('Delta de tiempo pronosticado para la vuelta L+1 (s)')
    ax.set_ylabel('')
    ax.set_title('Probabilidad de adelantamiento predicha por GP')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def build_mirror_tree(selected_gps):
    """Espejo por symlinks del arbol de simulacion original en NEW_TREE.
    Los overtake_logit.pkl / overtake_threshold.pkl de los GPs seleccionados
    NO se enlazan (los escribe main con el modelo nuevo). Nada del arbol
    original se modifica."""
    orig = str(config.MODELS_DIR / 'simulation')
    os.makedirs(NEW_TREE, exist_ok=True)
    for entry in sorted(os.listdir(orig)):
        src = os.path.join(orig, entry)
        dst = os.path.join(NEW_TREE, entry)
        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            skip = {'overtake_logit.pkl', 'overtake_threshold.pkl'} if entry in selected_gps else set()
            for f in sorted(os.listdir(src)):
                if f in skip:
                    continue
                d = os.path.join(dst, f)
                if not os.path.lexists(d):
                    os.symlink(os.path.relpath(os.path.join(src, f), dst), d)  # relative: portable
        else:
            if not os.path.lexists(dst):
                os.symlink(os.path.relpath(src, NEW_TREE), dst)  # relative: portable
    print(f"Espejo listo: {NEW_TREE}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    laps_2021_2022_2023, selected_GP, reg_clear_dd = load_data()
    logit_df = optimize_f1_analysis(laps_2021_2022_2023, selected_GP, reg_clear_dd)

    # Filter a few exceptions
    logit_df = logit_df[logit_df['LapNumber'] > 3]
    logit_df = logit_df[logit_df['LapNumber'] < logit_df['LastLap']]

    # Eliminar filas con NaNs o valores invalidos
    logit_df = logit_df.dropna(subset=['interval_front_real', 'delta_t_lap', 'DRS', 'overtake',
                                        'rival_PitOut'])
    logit_df['delta_total'] = logit_df['interval_front_real'] + logit_df['delta_t_lap']
    logit_df = logit_df[(logit_df['delta_total'] >= -30) & (logit_df['delta_total'] <= 30.0)]

    results = enhanced_logit_modeling_with_evaluation(logit_df, show_plots=False, save_plots=True)

    # Tabla GP x delta_total con probabilidad de overtake predicha
    prob_table = overtake_probability_table(results)
    prob_table.to_csv(os.path.join(OUT_DIR, 'overtake_prob_delta.csv'), index=False)
    plot_overtake_probability_heatmap(
        prob_table, save_path=os.path.join(OUT_DIR, 'overtake_prob_heatmap.png'))

    # --- Publicacion para el simulador (arbol espejo datado, nada se borra) ---
    build_mirror_tree(set(selected_GP.keys()))
    for gp, model in results['models'].items():
        gp_dir = os.path.join(NEW_TREE, gp)
        os.makedirs(gp_dir, exist_ok=True)
        with open(os.path.join(gp_dir, 'overtake_logit.pkl'), 'wb') as f:
            pickle.dump(model, f)
        with open(os.path.join(gp_dir, 'overtake_threshold.pkl'), 'wb') as f:
            pickle.dump(results['metrics'][gp]['threshold'], f)
        print(f"  guardado {gp}: const={model.params['const']:.3f} "
              f"slope={model.params['delta_total']:.4f} "
              f"tau(0)={1/(1+np.exp(-model.params['const'])):.3f}")

    # Curvas tau en el regimen de pit para validacion rapida
    print("\n=== tau(delta_total) por GP (modelo NUEVO) ===")
    for gp, model in results['models'].items():
        c, s = model.params['const'], model.params['delta_total']
        taus = {d: 1/(1+np.exp(-(c + s*d))) for d in [1, 0, -1, -5, -9.5, -24]}
        print(f"  {gp:28s} " + "  ".join(f"tau({d:+.1f})={t:.3f}" for d, t in taus.items()))
    print(f"\nPickles del simulador en: {NEW_TREE}\nUsar via F1_SIMULATION_DIR={NEW_TREE}")
