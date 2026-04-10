import config

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

from reg import FastLinearPredictor


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

    tyrelife_adjust = {'C0': 0.65,
                       'C1': 1.0,
                       'C2': 0.90,
                       'C3': 0.98,
                       'C4': 0.92,
                       'C5': 0.92}
    laps_2021['TyreLife'] = laps_2021['TyreLife'] * laps_2021['Compound_Detail'].map(tyrelife_adjust)

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
    reg_clear_dd = {}
    for gp in selected_GP.keys():
        with open(str(config.SIMULATION_DIR / f'{gp}_fast_predictor.pkl'), "rb") as f:
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

    Args:
    - laps (pd.DataFrame): DataFrame con datos de vueltas de carreras.

    Returns:
    - pd.DataFrame: DataFrame limpio.
    """
    # Eliminamos los GP donde se uso Compound WET o INTERMEDIATE
    laps_filter = laps[~laps['Compound'].isin(['WET', 'INTERMEDIATE'])].copy()

    # Filtramos solo compuestos C1-C5 (excluir C0)
    laps_filter = laps_filter[laps_filter['Compound_Detail'].isin(['C1', 'C2', 'C3', 'C4', 'C5'])].copy()

    # Calcular el numero maximo de vueltas por Year, GP y Driver
    max_laps = laps_filter.groupby(['Year', 'GP', 'Driver'])['LapNumber'].transform('max')

    # Crear la columna 'LapsLeft'
    laps_filter['LapsLeft'] = max_laps - laps_filter['LapNumber']

    # TrackStatus VSC or SC
    # laps_filter = laps_filter[laps_filter['TrackStatus'].isin([4., 6., 7.])]

    return laps_filter


def optimize_f1_analysis(laps_2021_2022_2023, selected_GP, reg_clear_dd):
    """
    Funcion optimizada para el analisis de vueltas de F1.

    Args:
        laps_2021_2022_2023: DataFrame con datos de vueltas
        selected_GP: Lista de GPs seleccionados
        reg_clear_dd: Diccionario con modelos de regresion

    Returns:
        logit_df: DataFrame procesado con predicciones y metricas
    """

    # 1. PREPARACION DE DATOS CON ENCODING EFICIENTE
    print("Preparando datos y creando variables dummy...")
    all_laps_df = laps_2021_2022_2023.copy()

    # Guardar columnas originales antes de crear dummies (necesarias para merge)
    all_laps_df['Year_original'] = all_laps_df['Year'].copy()
    all_laps_df['Driver_original'] = all_laps_df['Driver'].copy()

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
    all_laps_df = all_laps_df.drop(columns=['Year_original', 'Driver_original'])

    # Crear variable LapNumber_square (PitIn y PitOut ya existen en los datos)
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
    # Filtrar datos relevantes desde el inicio
    relevant_data = all_laps_df[all_laps_df['GP'].isin(selected_GP)].copy()
    relevant_data['LapTime_pred'] = np.nan

    # Agrupar por GP para procesar en lotes
    for gp in selected_GP:
        if gp not in reg_clear_dd:
            continue

        gp_mask = relevant_data['GP'] == gp
        gp_data = relevant_data[gp_mask]

        if gp_data.empty:
            continue

        # Hacer predicciones usando FastLinearPredictor
        try:
            fast_pred = reg_clear_dd[gp]
            # Convertir DataFrame a lista de diccionarios para predict_batch
            data_list = gp_data.to_dict('records')
            predictions = fast_pred.predict_batch(data_list)
            relevant_data.loc[gp_mask, 'LapTime_pred'] = predictions
        except Exception as e:
            print(f"Error al predecir para GP {gp}: {e}")
            continue


    # 3. PREPARACION DEL DATASET FINAL
    print("Preparando dataset final...")
    # Usar datos originales filtrados y agregar predicciones
    logit_df = laps_2021_2022_2023[laps_2021_2022_2023['GP'].isin(selected_GP)].copy()

    # Merge correcto usando columnas clave para asegurar alineacion
    pred_df = relevant_data[['Year', 'GP', 'Driver', 'LapNumber', 'LapTime_pred']].copy()
    logit_df = logit_df.merge(pred_df, on=['Year', 'GP', 'Driver', 'LapNumber'], how='left')

    # Ordenar una sola vez
    logit_df = logit_df.sort_values(['Year', 'GP', 'Driver', 'LapNumber']).reset_index(drop=True)


    # 4. CALCULO DE METRICAS OPTIMIZADO
    print("Calculando metricas de posicion...")

    # Calcular ultima vuelta por piloto de forma eficiente
    last_laps = logit_df.groupby(['Year', 'GP', 'Driver'])['LapNumber'].max().reset_index()
    last_laps.rename(columns={'LapNumber': 'LastLap'}, inplace=True)
    logit_df = logit_df.merge(last_laps, on=['Year', 'GP', 'Driver'])

    # Calcular posiciones siguientes y cambios de forma vectorizada
    logit_df['next_position'] = logit_df.groupby(['Year', 'GP', 'Driver'])['Position'].shift(-1)
    logit_df['position_gain/loss'] = logit_df['Position'] - logit_df['next_position']
    logit_df['position_gain/loss'] = logit_df['position_gain/loss'].fillna(0)


    # 5. DETECCION DE ADELANTAMIENTOS OPTIMIZADA
    print("Detectando adelantamientos...")
    logit_df['overtake'] = 0

    # Pre-filtrar candidatos a adelantamiento
    overtake_candidates = logit_df[
        (logit_df['LapNumber'] > 1) &
        (logit_df['position_gain/loss'] > 0) &
        (logit_df['TrackStatus'] == 1.0) &
        (logit_df['next_position'].notna())
    ].copy()

    if not overtake_candidates.empty:
        # Crear indice para busquedas rapidas
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
                if (rival['next_position'] >= next_pos and
                    rival['PitIn'] != 1.0 and
                    rival['PitOut'] != 1.0 and
                    rival['LastLap'] != lap):

                    logit_df.at[idx, 'overtake'] = 1

            except (KeyError, IndexError):
                continue


    # 6. CALCULO DE DELTA TIEMPO OPTIMIZADO
    print("Calculando deltas de tiempo...")
    logit_df['delta_t_lap'] = np.nan

    # Procesar por grupos para optimizar memoria
    for (year, gp), group in logit_df.groupby(['Year', 'GP']):
        if group.empty:
            continue

        # Crear indices rapidos para el grupo
        lap_times_dict = group.set_index(['Driver', 'LapNumber'])['LapTime_pred'].to_dict()
        position_dict = group.set_index(['LapNumber', 'Position'])['Driver'].to_dict()

        # Filtrar solo posiciones que no son primera
        non_leaders = group[group['Position'] > 1].copy()

        for idx, row in non_leaders.iterrows():
            driver, lap, pos = row['Driver'], row['LapNumber'], int(row['Position'])

            # Buscar piloto adelante
            rival_key = (lap, pos - 1)
            if rival_key not in position_dict:
                continue

            rival_driver = position_dict[rival_key]
            next_lap = lap + 1

            # Obtener tiempos de vuelta siguiente
            my_time = lap_times_dict.get((driver, next_lap))
            rival_time = lap_times_dict.get((rival_driver, next_lap))

            if pd.notna(my_time) and pd.notna(rival_time):
                logit_df.at[idx, 'delta_t_lap'] = my_time - rival_time

    print("Procesamiento completado!")
    return logit_df


def enhanced_logit_modeling_with_evaluation(logit_df, min_samples=30, min_positives=5,
                                          show_plots=True, save_plots=False, plot_dir='./plots/'):
    """
    Funcion mejorada que combina modelado robusto con evaluacion completa.

    Args:
        logit_df: DataFrame con los datos
        min_samples: Minimo numero de muestras por GP
        min_positives: Minimo numero de casos positivos por GP
        show_plots: Si mostrar graficos
        save_plots: Si guardar graficos
        plot_dir: Directorio para guardar graficos

    Returns:
        results: Diccionario con modelos, metricas y estadisticas
    """

    # Preparar directorio para plots si es necesario
    if save_plots:
        import os
        os.makedirs(plot_dir, exist_ok=True)

    # 1. LIMPIEZA DE DATOS
    print("Preparando datos...")
    logit_df_clean = logit_df.dropna(subset=['interval_front_real', 'delta_t_lap', 'DRS', 'overtake']).copy()
    logit_df_clean = logit_df_clean[logit_df_clean['interval_front_real'] <= 1.0]
    logit_df_clean['delta_total'] = logit_df_clean['interval_front_real'] + logit_df_clean['delta_t_lap']

    gps = logit_df_clean['GP'].unique()

    # 2. ESTRUCTURA PARA RESULTADOS
    results = {
        'models': {},
        'metrics': {},
        'failed_gps': [],
        'gp_stats': {}
    }

    print(f"\n{'='*80}")
    print("MODELADO Y EVALUACION POR GP")
    print(f"{'='*80}")

    for gp in gps:
        gp_df = logit_df_clean[logit_df_clean['GP'] == gp].copy()

        print(f"\n{'='*25} {gp} {'='*25}")

        # Estadisticas basicas
        n_total = len(gp_df)
        n_overtakes = gp_df['overtake'].sum()
        overtake_rate = n_overtakes / n_total if n_total > 0 else 0

        print(f"Muestras: {n_total} | Overtakes: {n_overtakes} | Tasa: {overtake_rate:.1%}")

        # Guardar estadisticas
        results['gp_stats'][gp] = {
            'total_samples': n_total,
            'overtakes': n_overtakes,
            'overtake_rate': overtake_rate,
            'delta_total_mean': gp_df['delta_total'].mean(),
            'delta_total_std': gp_df['delta_total'].std(),
        }

        # Verificar datos suficientes
        if n_total < min_samples:
            print(f"SALTADO: Muy pocas muestras ({n_total} < {min_samples})")
            results['failed_gps'].append((gp, "insufficient_samples"))
            continue

        if n_overtakes < min_positives:
            print(f"SALTADO: Muy pocos overtakes ({n_overtakes} < {min_positives})")
            results['failed_gps'].append((gp, "insufficient_positives"))
            continue

        # Preparar variables
        X = gp_df[['delta_total']].copy()
        y = gp_df['overtake'].copy()

        # 3. AJUSTE DEL MODELO
        model_success = False
        model = None
        method_used = None

        # METODO 1: Statsmodels
        try:
            # X['const'] = 1.0
            X_sm = sm.add_constant(X)
            model_sm = sm.Logit(y, X_sm).fit(
                disp=False,
                maxiter=1000,
                method='bfgs',
                warn_convergence=False
            )

            if model_sm.mle_retvals['converged']:
                model = model_sm
                method_used = "statsmodels"
                model_success = True
                print("STATSMODELS: Convergencia exitosa")
            else:
                print("STATSMODELS: No convergio, intentando Sklearn...")

        except Exception as e:
            print(f"ERROR STATSMODELS: {str(e)[:50]}...")


        # 4. PREDICCIONES Y EVALUACION
        try:
            # Preparar datos para prediccion
            if method_used == "statsmodels":
                X_pred = sm.add_constant(X)
            else:
                X_pred = X

            y_pred_prob = model.predict(X_pred)

            # Metodo de umbral adaptativo para balancear predicciones
            n_real_overtakes = int(y.sum())

            if n_real_overtakes > 0 and n_real_overtakes < len(y_pred_prob):
                # Ordenar probabilidades y encontrar umbral
                sorted_probs = np.sort(y_pred_prob)[::-1]  # Descendente
                threshold = sorted_probs[n_real_overtakes - 1]

                # Ajustar umbral ligeramente para evitar empates
                threshold = max(threshold - 1e-10, 0.001)
            else:
                threshold = 0.5

            print(f"Umbral adaptativo: {threshold:.4f}")

            # Predicciones binarias
            y_pred = (y_pred_prob >= threshold).astype(int)

            # 5. METRICAS DE EVALUACION
            cm = confusion_matrix(y, y_pred)

            # Calcular metricas
            tn, fp, fn, tp = cm.ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            # AUC si hay variabilidad en las clases
            try:
                auc = roc_auc_score(y, y_pred_prob) if len(np.unique(y)) > 1 else np.nan
            except:
                auc = np.nan

            # Guardar metricas
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

            # 6. MOSTRAR RESULTADOS
            print(f"Metricas:")
            print(f"   Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
            print(f"   Accuracy: {accuracy:.3f} | AUC: {auc:.3f}")
            print(f"   Overtakes reales: {n_real_overtakes} | Predichos: {int(y_pred.sum())}")

            # Mostrar coeficientes
            print(f"Coeficientes ({method_used}):")
            for param, coef in model.params.items():
                if method_used == "statsmodels" and hasattr(model, 'pvalues'):
                    pval = model.pvalues[param]
                    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                    print(f"   {param}: {coef:.4f} (p={pval:.4f}) {sig}")
                else:
                    print(f"   {param}: {coef:.4f}")

            # 7. VISUALIZACION
            if show_plots or save_plots:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                # Matriz de confusion
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                           xticklabels=['No Overtake', 'Overtake'],
                           yticklabels=['No Overtake', 'Overtake'],
                           ax=axes[0])
                axes[0].set_title(f'Matriz de Confusion - {gp}')
                axes[0].set_xlabel('Predicho')
                axes[0].set_ylabel('Real')

                # Distribucion de probabilidades
                axes[1].hist(y_pred_prob[y == 0], bins=20, alpha=0.7, label='No Overtake', color='lightblue')
                axes[1].hist(y_pred_prob[y == 1], bins=20, alpha=0.7, label='Overtake', color='orange')
                axes[1].axvline(threshold, color='red', linestyle='--', label=f'Umbral: {threshold:.3f}')
                axes[1].set_xlabel('Probabilidad Predicha')
                axes[1].set_ylabel('Frecuencia')
                axes[1].set_title(f'Distribucion de Probabilidades - {gp}')
                axes[1].legend()

                plt.tight_layout()

                if save_plots:
                    plt.savefig(f"{plot_dir}evaluation_{gp.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')

                if show_plots:
                    plt.show()
                else:
                    plt.close()

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
        # Metricas promedio
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

# FUNCION SIMPLIFICADA PARA USO DIRECTO
def quick_logit_analysis(logit_df):
    """Funcion simplificada para analisis rapido."""
    return enhanced_logit_modeling_with_evaluation(
        logit_df,
        show_plots=True,
        save_plots=False
    )


if __name__ == "__main__":
    laps_2021_2022_2023, selected_GP, reg_clear_dd = load_data()
    logit_df = optimize_f1_analysis(laps_2021_2022_2023, selected_GP, reg_clear_dd)

    # Filtramos algunas excepciones
    logit_df = logit_df[logit_df['LapNumber'] > 3]
    logit_df = logit_df[logit_df['LapNumber'] < logit_df['LastLap']]

    # Eliminar filas con NaNs o valores invalidos
    logit_df = logit_df.dropna(subset=['interval_front_real', 'delta_t_lap', 'DRS', 'overtake'])
    logit_df = logit_df[logit_df['interval_front_real'] <= 1.0]
    logit_df['delta_total'] = logit_df['interval_front_real'] + logit_df['delta_t_lap']

    results = quick_logit_analysis(logit_df)
    logit_dd = results['models']
    metrics = results['metrics']
