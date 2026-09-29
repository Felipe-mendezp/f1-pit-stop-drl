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
import numpy as np
import pandas as pd

from models.regressions import load_data, filter_laps, all_laps, calibrar_regresion_gp, filtrar_outliers_iqr
from reg import FastLinearPredictor


def apply_coef_override(fast_predictor: FastLinearPredictor, param_name: str, value: float):
    """
    Set (or ADD) a coefficient on a FastLinearPredictor.

    Compounds that never appear in a GP's 2021-2023 data have no dummy in
    the fitted model; the matrix-completion value must then be added as a
    NEW feature (with zero rows/cols in xtx_inv_, i.e. treated as
    noise-free for the prediction-std computation).
    """
    if param_name in fast_predictor.feature_names_:
        idx = fast_predictor.feature_names_.index(param_name)
        fast_predictor.coef_[idx] = fast_predictor.dtype_(value)
        return
    fast_predictor.feature_names_.append(param_name)
    fast_predictor.coef_ = np.append(
        fast_predictor.coef_, fast_predictor.dtype_(value)
    ).astype(fast_predictor.dtype_)
    n = fast_predictor.xtx_inv_.shape[0]
    extended = np.zeros((n + 1, n + 1), dtype=fast_predictor.xtx_inv_.dtype)
    extended[:n, :n] = fast_predictor.xtx_inv_
    fast_predictor.xtx_inv_ = extended
    # Rebuild categorical/interaction lookup tables with the new feature
    fast_predictor._parse_categorical_info()


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
    # 1b. Export compound coefficients for the matrix-completion step.
    # This regenerates data/compounds_coefs_V3.xlsx (the input of
    # scripts/run_matrix_completion.py) from the fitted regressions, so the
    # full pipeline is reproducible end-to-end:
    #   train_regressions -> run_matrix_completion -> train_regressions
    # ------------------------------------------------------------------
    coef_rows = []
    for gp, (modelos_gp, _df) in reg_results.items():
        params = modelos_gp.params
        pvalues = modelos_gp.pvalues
        for name in params.index:
            is_fixed = ('Compound_Detail' in name) and ('TyreLife' not in name)
            is_variable = name.startswith('TyreLife') and 'Compound_Detail' in name
            if is_fixed or is_variable:
                coef_rows.append({
                    'GP': gp,
                    'Variable': name,
                    'Coef': float(params[name]),
                    'p-value': float(pvalues[name]),
                })
    coefs_export = pd.DataFrame(coef_rows)
    coefs_export_path = os.path.join(config.DATA_DIR, 'compounds_coefs_V3.xlsx')
    coefs_export.to_excel(coefs_export_path)
    print(f"Exported {len(coefs_export)} compound coefficients to {coefs_export_path}")

    # ------------------------------------------------------------------
    # 2. (Optional) Update coefficients from matrix-completion Excel
    # ------------------------------------------------------------------
    coefs_excel = os.path.join(config.DATA_DIR, 'completed_coefs_V2_filled.xlsx')
    selected_GP = {}
    coef_overrides = {}  # {gp: {param_name: completed_value}}

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

            # Record the completed coefficients as OVERRIDES to be applied on
            # the FastLinearPredictor after construction. NOTE: assigning into
            # statsmodels' result.params Series does NOT reliably persist
            # under pandas >= 3.0 copy-on-write (the write lands on a
            # temporary), so we never mutate the statsmodels result here.
            gp_overrides = coef_overrides.setdefault(gp_name, {})

            # Fixed coefficients (Compound_Detail)
            for compound in compounds:
                if compound == compound_ref:
                    print(f"  {compound}: reference (0.0)")
                    continue
                fixed_value = df_fixed[compound].iloc[0]
                if pd.notna(fixed_value):
                    param_name = (
                        f"C(Compound_Detail, Treatment(reference='"
                        f"{compound_ref}'))[T.{compound}]"
                    )
                    gp_overrides[param_name] = float(fixed_value)
                    print(f"  {compound} (fixed): {fixed_value}")

            # Variable coefficients (TyreLife:Compound_Detail)
            for compound in compounds:
                variable_value = df_variable[compound].iloc[0]
                if pd.notna(variable_value):
                    param_name = f"TyreLife:C(Compound_Detail)[{compound}]"
                    gp_overrides[param_name] = float(variable_value)
                    print(f"  {compound} (variable): {variable_value}")
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

        # Create FastLinearPredictor (for RL - 32x faster predictions) and
        # apply the matrix-completion overrides directly on its coefficient
        # array (plain numpy: immune to pandas copy-on-write semantics).
        fast_predictor = FastLinearPredictor(lreg, use_float32=True)
        for param_name, value in coef_overrides.get(gp, {}).items():
            apply_coef_override(fast_predictor, param_name, value)
        fast_predictors[gp] = fast_predictor

        filepath_fast = os.path.join(
            str(config.SIMULATION_DIR), f'{gp}_fast_predictor.pkl'
        )
        pickle.dump(fast_predictor, open(filepath_fast, 'wb'))

        # Save to the per-GP layout that ModelLoader actually loads
        # (trained_models/simulation/<GP>/laptime_clear.pkl)
        gp_dir = os.path.join(str(config.SIMULATION_DIR), gp)
        os.makedirs(gp_dir, exist_ok=True)
        pickle.dump(fast_predictor, open(os.path.join(gp_dir, 'laptime_clear.pkl'), 'wb'))

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
