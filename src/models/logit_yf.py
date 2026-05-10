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


def load_data():
    """
    Load session CSVs for 2021-2024, apply TyreLife adjustment, filter,
    and prepare the concatenated DataFrame for yellow-flag pit-stop modeling.

    Returns:
        tuple: (pitstop_df, laps_2021_2022_2023_yf)
    """
    # Cargamos los dataframes
    laps_2021 = pd.read_csv(str(config.DATA_DIR / 'session_2021_V2.csv'))
    laps_2022 = pd.read_csv(str(config.DATA_DIR / 'session_2022_V2.csv'))
    laps_2023 = pd.read_csv(str(config.DATA_DIR / 'session_2023_V2.csv'))
    laps_2024 = pd.read_csv(str(config.DATA_DIR / 'session_2024_V2.csv'))

    tyrelife_adjust = {'C0': 0.65,
                       'C1': 1.0,
                       'C2': 0.90,
                       'C3': 0.98,
                       'C4': 0.92,
                       'C5': 0.92}
    laps_2021['TyreLife'] = laps_2021['TyreLife'] * laps_2021['Compound_Detail'].map(tyrelife_adjust)

    # Concatenamos las vuelta del 2021 al 2023
    laps_2021_2022_2023 = pd.concat([laps_2021, laps_2022, laps_2023])

    # Filter the laps
    laps_2021_2022_2023_yf = filter_laps(laps_2021_2022_2023).reset_index(drop=True)
    laps_2021_2022_2023_yf['TrackStatus'] = laps_2021_2022_2023_yf['TrackStatus'].replace(7.0, 6.0)

    laps_2021_2022_2023_yf["TrackStatus"] = laps_2021_2022_2023_yf["TrackStatus"].apply(reduce_trackstatus)
    laps_2021_2022_2023_yf = laps_2021_2022_2023_yf[laps_2021_2022_2023_yf["TrackStatus"].isin([4, 6])]

    # Ordenar para asegurar correcta deteccion de procesos
    df_sc = laps_2021_2022_2023_yf.sort_values(by=['Year', 'GP', 'Driver', 'LapNumber'])

    # Detectar nuevos procesos
    df_sc['proceso_id'] = (
        (df_sc['TrackStatus'] != df_sc['TrackStatus'].shift()) |
        (df_sc['LapNumber'] != df_sc['LapNumber'].shift() + 1) |
        (df_sc['Driver'] != df_sc['Driver'].shift())
    ).cumsum()

    # Agrupar por proceso
    filter_df = df_sc.groupby('proceso_id').agg({
        'Driver': 'first',
        'LapNumber': 'first',
        'TrackStatus': 'first',
        'PitIn': 'max',
        'Compound_Detail': 'first',
        'TyreLife': 'max',
        'LapsLeft': 'max',
        'Position': 'first'
    }).reset_index(drop=True)

    filter_df['SC'] = (filter_df['TrackStatus'] == 4.).astype(int)

    pitstop_df = filter_df.copy()

    return pitstop_df, laps_2021_2022_2023_yf


def filter_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina ciertos Grandes Premios y filas con TrackStatus erroneo.

    Args:
    - laps (pd.DataFrame): DataFrame with race lap data.

    Returns:
    - pd.DataFrame: DataFrame limpio.
    """
    # Eliminamos los GP donde se uso Compound WET o INTERMEDIATE
    laps_filter = laps[~laps['Compound'].isin(['WET', 'INTERMEDIATE'])].copy()

    # Keep only compounds C1-C5 (exclude C0)
    laps_filter = laps_filter[laps_filter['Compound_Detail'].isin(['C1', 'C2', 'C3', 'C4', 'C5'])].copy()

    # Compute the maximum lap number per (Year, GP, Driver)
    max_laps = laps_filter.groupby(['Year', 'GP', 'Driver'])['LapNumber'].transform('max')

    # Crear la columna 'LapsLeft'
    laps_filter['LapsLeft'] = max_laps - laps_filter['LapNumber']

    # TrackStatus VSC or SC
    # laps_filter = laps_filter[laps_filter['TrackStatus'].isin([4., 6., 7.])]

    return laps_filter


def reduce_trackstatus(ts):
    digits = set(map(int, str(ts)))
    if 4 in digits:
        return 4          # Safety Car
    elif 6 in digits or 7 in digits:
        return 6          # VSC
    elif 2 in digits:
        return 2          # Yellow
    else:
        return 1          # Track clear


def simple_pitstop_prediction(pitstop_df, random_state=42,
                             show_plots=True, save_plots=False, plot_dir='./plots/'):
    """
    Funcion para crear un modelo simple de prediccion de pit stops.
    USA: SC, TyreLife, LapsLeft, TyreLife*Compound_Detail, LapsLeft*Compound_Detail
    (Eliminada Compound_Detail sola, ahora solo interacciones)
    Todo el dataset se usa para entrenamiento (sin division train/test)

    Args:
        pitstop_df: DataFrame with the data de pit stops
        random_state: Semilla para reproducibilidad
        show_plots: Si mostrar graficos
        save_plots: Si guardar graficos
        plot_dir: Directorio para guardar graficos

    Returns:
        results: Diccionario con modelo, metricas y estadisticas
    """

    # Preparar directorio para plots si es necesario
    if save_plots:
        import os as _os
        _os.makedirs(plot_dir, exist_ok=True)

    # 1. LIMPIEZA Y PREPARACION DE DATOS
    print("=" * 80)
    print("MODELO SIMPLE DE PREDICCION DE PIT STOPS (CON INTERACCIONES)")
    print("=" * 80)
    print("Preparing data...")

    # Variables requeridas
    required_vars = ['PitIn', 'SC', 'Compound_Detail', 'TyreLife', 'LapsLeft', 'Position']

    # Verificar que existan las variables
    missing_vars = [var for var in required_vars if var not in pitstop_df.columns]
    if missing_vars:
        raise ValueError(f"Variables faltantes en el DataFrame: {missing_vars}")

    # Clean data
    df_clean = pitstop_df.dropna(subset=required_vars).copy()

    # Filters for valid data
    df_clean = df_clean[
        (df_clean['TyreLife'] >= 0) &
        (df_clean['TyreLife'] <= 100) &
        (df_clean['LapsLeft'] >= 0) &
        (df_clean['LapsLeft'] <= 100)
    ].copy()

    print(f"Datos despues de limpieza: {len(df_clean):,} observaciones")

    # 2. INGENIERIA DE CARACTERISTICAS - INTERACCIONES CON COMPOUND
    print("Preparando variables con interacciones...")

    # Crear interacciones (LapsLeft + TyreLife) * Compound para cada compuesto C1-C5
    compounds = ['C1', 'C2', 'C3', 'C4', 'C5']
    interactions = pd.DataFrame()
    for comp in compounds:
        interaction_name = f'LapsLeft_TyreLife_x_{comp}'
        is_compound = (df_clean['Compound_Detail'] == comp).astype(int)
        interactions[interaction_name] = (df_clean['LapsLeft'] + df_clean['TyreLife']) * is_compound

    print(f"Interacciones (LapsLeft + TyreLife) * Compound creadas: {list(interactions.columns)}")

    # Combinar todas las variables
    df_model = pd.concat([
        df_clean[['PitIn', 'SC', 'Position', 'LapsLeft']],
        interactions
    ], axis=1)

    # 3. ESTADISTICAS DESCRIPTIVAS
    n_total = len(df_clean)
    n_pitstops = df_clean['PitIn'].sum()
    pitstop_rate = n_pitstops / n_total

    print(f"\nESTADISTICAS GENERALES:")
    print(f"   Total observaciones: {n_total:,}")
    print(f"   Pit stops: {n_pitstops:,}")
    print(f"   Tasa general de pit stops: {pitstop_rate:.1%}")
    print(f"   SC rate: {df_clean['SC'].mean():.1%}")
    print(f"   Avg tire life: {df_clean['TyreLife'].mean():.1f} laps")
    print(f"   Vueltas restantes promedio: {df_clean['LapsLeft'].mean():.1f}")

    # Distribucion por compuesto
    compound_stats = df_clean.groupby('Compound_Detail').agg({
        'PitIn': ['count', 'sum', 'mean']
    }).round(3)
    compound_stats.columns = ['Total', 'PitStops', 'Rate']
    print(f"\nDISTRIBUCION POR COMPUESTO:")
    print(compound_stats)

    # Estadisticas de interacciones
    print(f"\nESTADISTICAS DE INTERACCIONES:")
    print("(LapsLeft + TyreLife) * Compound (promedios):")
    for col in interactions.columns:
        mean_val = interactions[col].mean()
        print(f"   {col}: {mean_val:.2f}")

    # 4. PREPARAR VARIABLES PARA MODELADO
    feature_cols = [col for col in df_model.columns if col != 'PitIn']

    X = df_model[feature_cols].copy()
    y = df_model['PitIn'].copy()

    print(f"\nVariables en el modelo: {len(feature_cols)}")
    print("Variables principales:")
    main_vars = [col for col in feature_cols if not ('_x_' in col)]
    for i, col in enumerate(main_vars):
        print(f"   {i+1}. {col}")

    print("Interacciones (LapsLeft + TyreLife) * Compound:")
    interaction_vars = [col for col in feature_cols if 'LapsLeft_TyreLife_x_' in col]
    for i, col in enumerate(interaction_vars):
        print(f"   {len(main_vars)+i+1}. {col}")

    print(f"\nDatos para entrenamiento:")
    print(f"   Total observaciones: {len(X):,}")
    print(f"   Pit stops: {y.sum():,} ({y.mean():.1%})")

    # 5. AJUSTE DEL MODELO
    print(f"\nAJUSTE DEL MODELO:")
    print("-" * 40)

    model_success = False
    model = None
    method_used = None

    # METODO 1: Statsmodels (preferido para interpretabilidad)
    try:
        print("Intentando con Statsmodels...")
        X_sm = sm.add_constant(X)
        model_sm = sm.Logit(y, X_sm).fit(
            disp=False,
            maxiter=2000,
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
        print(f"ERROR STATSMODELS: {str(e)[:80]}...")

    # METODO 2: Sklearn como respaldo
    if not model_success:
        try:
            print("Intentando con Sklearn...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model_sklearn = LogisticRegression(
                random_state=random_state,
                max_iter=3000,
                solver='liblinear',
                C=1.0,
                class_weight='balanced'
            ).fit(X_scaled, y)

            # Wrapper para compatibilidad
            class SklearnLogitWrapper:
                def __init__(self, sklearn_model, scaler, feature_names):
                    self.sklearn_model = sklearn_model
                    self.scaler = scaler
                    self.feature_names = feature_names
                    self.method = "sklearn"
                    self._create_params()

                def _create_params(self):
                    coefs = self.sklearn_model.coef_[0]
                    intercept = self.sklearn_model.intercept_[0]
                    self.params = pd.Series({
                        'const': intercept,
                        **{name: coef for name, coef in zip(self.feature_names, coefs)}
                    })

                def predict(self, X_input):
                    if isinstance(X_input, pd.DataFrame):
                        X_vals = X_input.drop('const', axis=1, errors='ignore')
                    else:
                        X_vals = X_input
                    X_scaled = self.scaler.transform(X_vals)
                    return self.sklearn_model.predict_proba(X_scaled)[:, 1]

            model = SklearnLogitWrapper(model_sklearn, scaler, X.columns.tolist())
            method_used = "sklearn"
            model_success = True
            print("SKLEARN: Ajuste exitoso")

        except Exception as e:
            print(f"ERROR SKLEARN: {str(e)[:80]}...")
            raise Exception("No se pudo ajustar el modelo con ningun metodo")

    # 6. PREDICCIONES Y EVALUACION
    print(f"\nEVALUACION DEL MODELO:")
    print("-" * 40)

    # Predicciones
    if method_used == "statsmodels":
        X_pred = sm.add_constant(X)
    else:
        X_pred = X

    y_pred_prob = model.predict(X_pred)

    # Encontrar umbral que iguala predicciones positivas con casos reales positivos
    n_positive_real = int(y.sum())  # Convertir a int de Python

    # Ordenar probabilidades de mayor a menor para encontrar el umbral
    sorted_probs = np.sort(y_pred_prob)[::-1]  # De mayor a menor

    # El umbral sera la probabilidad en la posicion n_positive_real
    if n_positive_real > 0 and n_positive_real <= len(sorted_probs):
        optimal_threshold = sorted_probs[n_positive_real - 1]
    else:
        optimal_threshold = 0.5  # Fallback

    print(f"Umbral para igualar predicciones: {optimal_threshold:.3f}")
    print(f"Casos positivos reales: {n_positive_real}")

    # Verificar que el umbral produce el resultado esperado
    y_pred_temp = (y_pred_prob >= optimal_threshold).astype(int)
    n_positive_pred = y_pred_temp.sum()
    print(f"Casos positivos predichos: {n_positive_pred}")

    # Predicciones finales
    y_pred = (y_pred_prob >= optimal_threshold).astype(int)

    # Calcular metricas
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score_val = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    try:
        auc = roc_auc_score(y, y_pred_prob) if len(np.unique(y)) > 1 else np.nan
    except:
        auc = np.nan

    # Imprimir metricas solicitadas
    print(f"\nMETRICAS DEL MODELO (Umbral balanceado):")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1_score_val:.4f}")
    print(f"   AUC:       {auc:.4f}")
    print(f"   Pit stops reales: {y.sum()}")
    print(f"   Pit stops predichos: {y_pred.sum()}")
    print(f"   Diferencia: {abs(y.sum() - y_pred.sum())}")

    # Verificar simetria de la matriz de confusion
    print(f"\nMATRIZ DE CONFUSION:")
    print(f"   TN: {tn}, FP: {fp}")
    print(f"   FN: {fn}, TP: {tp}")
    print(f"   Suma diagonal secundaria: FP + FN = {fp + fn}")
    if fp + fn > 0:
        print(f"   Balance FP/FN: {fp/(fp+fn):.1%} FP, {fn/(fp+fn):.1%} FN")

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score_val,
        'auc': auc,
        'confusion_matrix': cm,
        'predicted_positives': int(y_pred.sum()),
        'actual_positives': int(y.sum())
    }

    # 7. ANALISIS DE COEFICIENTES
    print(f"\nCOEFICIENTES DEL MODELO ({method_used.upper()}):")
    print("-" * 50)

    # Ordenar coeficientes por magnitud
    coef_df = pd.DataFrame({
        'variable': model.params.index,
        'coeficiente': model.params.values,
        'abs_coef': np.abs(model.params.values)
    })

    if method_used == "statsmodels" and hasattr(model, 'pvalues'):
        coef_df['p_value'] = model.pvalues.values
        coef_df['significativo'] = coef_df['p_value'] < 0.05

    coef_df = coef_df.sort_values('abs_coef', ascending=False)

    print("Variables mas influyentes:")
    for _, row in coef_df.iterrows():
        var = row['variable']
        coef = row['coeficiente']

        direction = "aumenta" if coef > 0 else "disminuye" if coef < 0 else "neutro"

        # Classify variable type
        if var == 'const':
            var_type = "[INTERCEPTO]"
        elif var in ['SC', 'Position', 'LapsLeft']:
            var_type = "[PRINCIPAL]"
        elif 'LapsLeft_TyreLife_x_' in var:
            var_type = "[INTERAC]"
        else:
            var_type = "[OTRO]"

        if 'p_value' in row:
            pval = row['p_value']
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"   {var_type} {var}: {coef:.4f} {direction} prob. pit stop (p={pval:.4f}) {sig}")
        else:
            print(f"   {var_type} {var}: {coef:.4f} {direction} prob. pit stop")

    # 8. VISUALIZACIONES (SOLO MATRIZ DE CONFUSION Y DISTRIBUCION DE PROBABILIDADES)
    if show_plots or save_plots:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Analisis del Modelo con Interacciones Compound', fontsize=16)

        # 1. Matriz de confusion
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   cbar=False, xticklabels=['No Pit Stop', 'Pit Stop'],
                   yticklabels=['No Pit Stop', 'Pit Stop'], ax=axes[0])
        axes[0].set_title(f'Matriz de Confusion (Umbral Balanceado)\nFP+FN = {fp+fn}')
        axes[0].set_xlabel('Predicho')
        axes[0].set_ylabel('Real')

        # Anadir texto adicional sobre el balance
        axes[0].text(0.02, 0.98, f'Predichos: {y_pred.sum()}\nReales: {y.sum()}',
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 2. Distribucion de probabilidades
        axes[1].hist(y_pred_prob[y == 0], bins=30, alpha=0.7, label='No Pit Stop', color='lightblue')
        axes[1].hist(y_pred_prob[y == 1], bins=30, alpha=0.7, label='Pit Stop', color='orange')
        axes[1].axvline(optimal_threshold, color='red', linestyle='--',
                       label=f'Umbral balanceado: {optimal_threshold:.3f}')
        axes[1].set_xlabel('Probabilidad Predicha')
        axes[1].set_ylabel('Frecuencia')
        axes[1].set_title('Distribucion de Probabilidades')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            plt.savefig(f"{plot_dir}simple_pitstop_model_interactions.png", dpi=300, bbox_inches='tight')

        if show_plots:
            plt.show()
        else:
            plt.close()

    # 9. PREPARAR RESULTADOS
    results = {
        'model': model,
        'method': method_used,
        'optimal_threshold': optimal_threshold,
        'compounds': compounds,
        'interactions': list(interactions.columns),
        'feature_columns': feature_cols,
        'metrics': metrics,
        'coefficients': coef_df,
        'sample_info': {
            'total': len(df_clean),
            'pitstops_total': int(df_clean['PitIn'].sum()),
            'pitstop_rate': pitstop_rate
        }
    }

    print(f"\nMODELO CON INTERACCIONES COMPLETADO EXITOSAMENTE")
    print(f"Rendimiento: F1={f1_score_val:.3f}, AUC={auc:.3f}")

    return results

# FUNCION PARA HACER PREDICCIONES CON EL MODELO CON INTERACCIONES
def predict_pitstop_simple(new_data, results):
    """
    Predice probabilidades de pit stop usando el modelo con interacciones.

    Args:
        new_data: DataFrame con las variables predictoras (SC, Position, LapsLeft, TyreLife, Compound_Detail)
        results: Resultados del modelo con interacciones

    Returns:
        Array con probabilidades de pit stop
    """
    model = results['model']
    feature_cols = results['feature_columns']
    compounds = results['compounds']

    # Prepare data
    new_data_prep = new_data.copy()

    # Crear interacciones (LapsLeft + TyreLife) * Compound para cada compuesto
    interactions_new = pd.DataFrame()
    for comp in compounds:
        interaction_name = f'LapsLeft_TyreLife_x_{comp}'
        is_compound = (new_data_prep['Compound_Detail'] == comp).astype(int)
        interactions_new[interaction_name] = (new_data_prep['LapsLeft'] + new_data_prep['TyreLife']) * is_compound

    # Combinar todas las variables
    new_data_final = pd.concat([
        new_data_prep[['SC', 'Position', 'LapsLeft']],
        interactions_new
    ], axis=1)

    # Seleccionar solo las variables del modelo
    X_new = new_data_final[feature_cols]

    # Hacer prediccion
    if hasattr(model, 'method') and model.method == "sklearn":
        probabilities = model.predict(X_new)
    else:
        X_new_sm = sm.add_constant(X_new)
        probabilities = model.predict(X_new_sm)

    return probabilities

# FUNCION SIMPLIFICADA PARA USO DIRECTO
def quick_simple_pitstop_model(pitstop_df):
    """Funcion simplificada para crear modelo con interacciones de pit stops."""
    return simple_pitstop_prediction(
        pitstop_df,
        show_plots=False,
        save_plots=False
    )


if __name__ == "__main__":
    pitstop_df, _laps_yf = load_data()

    results = quick_simple_pitstop_model(pitstop_df)

    # Ver metricas
    print("Metricas del modelo:", results['metrics'])

    print(results['model'].summary())

    filepath = str(config.SIMULATION_DIR / 'pit_stops_YF_rivals_reg_Pickle.pkl')
    pickle.dump(results['model'], open(filepath, 'wb'))
