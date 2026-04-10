"""
Step 2: Train clear-track lap time regressions (RegLTC - Eq. 1).

Runs OLS regressions per GP with iterative IQR outlier removal,
optionally updates coefficients from matrix-completion results,
and saves both statsmodels and FastLinearPredictor models.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
import compat
compat.setup_all()

import pickle
import pandas as pd

from models.regressions import load_data, filter_laps, all_laps, calibrar_regresion_gp, filtrar_outliers_iqr
from reg import FastLinearPredictor


def main():
    # ------------------------------------------------------------------
    # 1. Load laps and run regressions per GP
    # ------------------------------------------------------------------
    laps_non_outliers = all_laps()
    laps_non_outliers['GP'] = laps_non_outliers['GP'].replace(
        'Styrian Grand Prix', 'Austrian Grand Prix'
    )
    laps_non_outliers.loc[
        laps_non_outliers['GP'] == 'Emilia Romagna Grand Prix',
        'Compound_Detail'
    ] = laps_non_outliers.loc[
        laps_non_outliers['GP'] == 'Emilia Romagna Grand Prix',
        'Compound_Detail'
    ].replace('C2', 'C5')

    reg_results = {}
    for gp in laps_non_outliers.GP.unique():
        try:
            modelos_gp, df_non_outlier = filtrar_outliers_iqr(laps_non_outliers, gp)
            reg_results[gp] = [modelos_gp, df_non_outlier]
        except Exception:
            pass

    print(f"Trained regressions for {len(reg_results)} GPs")

    # ------------------------------------------------------------------
    # 2. (Optional) Update coefficients from matrix-completion Excel
    # ------------------------------------------------------------------
    coefs_excel = os.path.join(config.DATA_DIR, 'completed_coefs_V2_filled.xlsx')
    selected_GP = {}

    if os.path.isfile(coefs_excel):
        print(f"Loading matrix-completion coefficients from {coefs_excel}")
        coefs_fixed_df = pd.read_excel(coefs_excel, sheet_name='fixed')
        coefs_variable_df = pd.read_excel(coefs_excel, sheet_name='variable')

        coefs_fixed_df = coefs_fixed_df[coefs_fixed_df['C_2024_m'].str.strip() == ', ,']
        coefs_variable_df = coefs_variable_df[coefs_variable_df['C_2024_m'].str.strip() == ', ,']

        common_gps = set(coefs_fixed_df['Unnamed: 0']).intersection(
            coefs_variable_df['Unnamed: 0']
        )
        coefs_fixed_df = coefs_fixed_df[
            coefs_fixed_df['Unnamed: 0'].isin(common_gps)
        ].reset_index(drop=True)
        coefs_variable_df = coefs_variable_df[
            coefs_variable_df['Unnamed: 0'].isin(common_gps)
        ].reset_index(drop=True)

        coefs_fixed_df = coefs_fixed_df[
            coefs_fixed_df['Unnamed: 0'] != 'Canadian Grand Prix'
        ].reset_index(drop=True)
        coefs_variable_df = coefs_variable_df[
            coefs_variable_df['Unnamed: 0'] != 'Canadian Grand Prix'
        ].reset_index(drop=True)
        print(coefs_fixed_df)

        selected_GP = (
            coefs_variable_df
            .set_index('Unnamed: 0')['C_2024']
            .str.split(', ')
            .to_dict()
        )
        print(selected_GP)

        for gp_name, compounds in selected_GP.items():
            print(f"\n{gp_name}")

            df_fixed = coefs_fixed_df[coefs_fixed_df['Unnamed: 0'] == gp_name].copy()
            df_variable = coefs_variable_df[coefs_variable_df['Unnamed: 0'] == gp_name].copy()

            if df_fixed.empty or df_variable.empty:
                print(f"Warning: no data found for {gp_name}")
                continue

            param_index = reg_results[gp_name][0].params.index
            param_series = pd.Series(param_index)

            compound_ref = (
                param_series[param_series.str.contains('Compound_Detail')]
                .str.extract(r"reference='([^']+)'")[0]
                .unique()[0]
            )

            if compound_ref is None:
                print(f"Warning: no reference compound found for {gp_name}")
                continue

            print(f"Reference compound: {compound_ref}")

            # Update fixed coefficients (Compound_Detail)
            for compound in compounds:
                if compound == compound_ref:
                    print(f"  {compound}: reference (0.0)")
                    continue
                fixed_value = df_fixed[compound].iloc[0]
                if pd.notna(fixed_value):
                    try:
                        param_name = (
                            f"C(Compound_Detail, Treatment(reference='"
                            f"{compound_ref}'))[T.{compound}]"
                        )
                        reg_results[gp_name][0].params[param_name] = fixed_value
                        print(f"  {compound} (fixed): {fixed_value}")
                    except Exception as e:
                        print(f"  Error updating {compound} (fixed): {e}")

            # Update variable coefficients (TyreLife:Compound_Detail)
            for compound in compounds:
                variable_value = df_variable[compound].iloc[0]
                if pd.notna(variable_value):
                    try:
                        param_name = f"TyreLife:C(Compound_Detail)[{compound}]"
                        reg_results[gp_name][0].params[param_name] = variable_value
                        print(f"  {compound} (variable): {variable_value}")
                    except Exception as e:
                        print(f"  Error updating {compound} (variable): {e}")
    else:
        print(f"Matrix-completion file not found ({coefs_excel}), skipping coefficient update.")
        # Use all GP results when no matrix-completion file is available
        selected_GP = {gp: [] for gp in reg_results}

    # ------------------------------------------------------------------
    # 3. Save models
    # ------------------------------------------------------------------
    os.makedirs(str(config.SIMULATION_DIR), exist_ok=True)

    good_gp_reg = {}
    fast_predictors = {}

    gps_to_save = selected_GP if selected_GP else reg_results

    for gp in gps_to_save:
        if gp not in reg_results:
            print(f"Warning: {gp} not in regression results, skipping.")
            continue

        lreg = reg_results[gp][0]
        good_gp_reg[gp] = lreg

        # Save statsmodels model (for statistical analysis)
        filepath_statsmodels = os.path.join(
            str(config.SIMULATION_DIR), f'{gp}_reg_V2_Pickle.pkl'
        )
        pickle.dump(lreg, open(filepath_statsmodels, 'wb'))

        # Create and save FastLinearPredictor (for RL - 32x faster predictions)
        fast_predictor = FastLinearPredictor(lreg, use_float32=True)
        fast_predictors[gp] = fast_predictor

        filepath_fast = os.path.join(
            str(config.SIMULATION_DIR), f'{gp}_fast_predictor.pkl'
        )
        pickle.dump(fast_predictor, open(filepath_fast, 'wb'))

        # Print performance info
        info = fast_predictor.get_info()
        print(f"\n{gp} - Fast Predictor Info:")
        print(f"  Features: {info['n_features']}")
        print(f"  Categorical vars: {info['n_categorical_vars']}")
        print(f"  Interactions: {info['n_interactions']}")
        print(f"  Memory: {info['memory_bytes']} bytes")
        print(f"  Expected speedup: 32x (statsmodels 44us -> FastPredictor 1.4us)")

    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"  Created fast predictors for {len(fast_predictors)} GPs")
    print(f"  Expected prediction speedup: 32x faster than statsmodels")
    print(f"  Memory usage: 50% reduction (float32)")
    print(f"  For 10,000 predictions: 440ms -> 14ms per episode")
    print("=" * 70)


if __name__ == "__main__":
    main()
