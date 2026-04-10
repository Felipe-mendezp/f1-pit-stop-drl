# python code that performs a matrix completion for SC VSC lap times
import os
import numpy as np
import pandas as pd
# from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from models.regressions import filtrar_outliers_iqr
from fancyimpute import SoftImpute

from config import DATA_DIR

# function that reads the lap time data and preprocess it
def read_and_preprocess_data():

    def filter_laps(laps: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina ciertos Grandes Premios y filas con compound WET - INT.

        Args:
        - laps (pd.DataFrame): DataFrame con datos de vueltas de carreras.

        Returns:
        - pd.DataFrame: DataFrame limpio.
        """
        # Eliminamos los GP donde se uso Compound WET o INTERMEDIATE
        try:
            laps = laps[~laps['Compound'].isin(['WET', 'INTERMEDIATE'])]
        except:
            pass
        laps = laps.dropna()

        return laps

    def pre_process_laps(laps):
        laps.columns
        # 'TrackStatus' 1 = GREEN, 2 = YELLOW FLAG, 4 = SC, 5 = RED FLAG, 6 = VSC, 7 = VSC ENDING
        # drop rows that have a 5 in TrackStatus, such as 5., 45.
        laps = laps[~laps['TrackStatus'].astype(str).str.contains('5')]
        laps['TrackStatus_2'] = laps['TrackStatus'].copy()
        # convert every that contains 7 to 1
        laps['TrackStatus_2'] = laps['TrackStatus_2'].replace({267.: 1., 67.: 1., 71.: 1., 7.: 1.})
        # convert every that contains 24 to 4
        laps['TrackStatus_2'] = laps['TrackStatus_2'].replace({24.: 4.})
        # convert every that contains 26 to 6
        laps['TrackStatus_2'] = laps['TrackStatus_2'].replace({26.: 6., 126.:6., 671.:6., 1267.:6., 167.:6., 2671.:6., 2167.:6., 16.:6.})
        # convert every that contains 64 to 4
        laps['TrackStatus_2'] = laps['TrackStatus_2'].replace({124.: 4., 41.: 4., 14.: 4., 1264. : 4., 164.: 4., 214.:4, 64.: 4.})
        # convert every that contains 264 to 4
        laps['TrackStatus_2'] = laps['TrackStatus_2'].replace({264.: 4.})
        # convert every that contains 264 to 4
        laps['TrackStatus_2'] = laps['TrackStatus_2'].replace({12.: 2., 21.:2.})
        # remove rows with 2
        laps = laps[laps['TrackStatus_2'] != 2.]

        # binary SC variable, VSC
        laps['SC'] = (laps['TrackStatus_2'] == 4.).astype(int)
        laps['VSC'] = (laps['TrackStatus_2'] == 6.).astype(int)

        # replace spaces of column names with underscores
        laps.columns = laps.columns.str.replace(' ', '_')
        return laps

    laps_2021 = pd.read_csv(os.path.join(DATA_DIR, 'session_2021_V2.csv'))
    laps_2022 = pd.read_csv(os.path.join(DATA_DIR, 'session_2022_V2.csv'))
    laps_2023 = pd.read_csv(os.path.join(DATA_DIR, 'session_2023_V2.csv'))


    tyrelife_adjust = {'C0': 0.65,
                   'C1': 1.0,
                   'C2': 0.90,
                   'C3': 0.98,
                   'C4': 0.92,
                   'C5': 0.92}
    laps_2021['TyreLife'] = laps_2021['TyreLife'] * laps_2021['Compound_Detail'].map(tyrelife_adjust)

    laps_2021_2022_2023 = pd.concat([laps_2021, laps_2022, laps_2023])

    # remove laps with intermediate or wet tires and NaN values
    laps_2021_2022_2023 = filter_laps(laps_2021_2022_2023)

    laps_2021_2022_2023['Interval_front'] = np.exp(-laps_2021_2022_2023['Interval_front'])

    laps_2021_2022_2023 = pre_process_laps(laps_2021_2022_2023) # preprocess the laps trackstatus
    print(laps_2021_2022_2023["TrackStatus_2"].unique())

    # build a new column with a one of this is the first lap of the VSC or SC for each driver
    # ensure correct ordering within each race for each driver
    laps_2021_2022_2023 = laps_2021_2022_2023.sort_values(["Year", "GP", "Driver", "LapNumber"]).copy()

    g = laps_2021_2022_2023.groupby(["Year", "GP", "Driver"], sort=False)

    # start of a VSC segment: VSC==1 and previous lap's VSC==0 (or missing)
    laps_2021_2022_2023["VSC_firstlap"] = ((laps_2021_2022_2023["VSC"].eq(1)) & (g["VSC"].shift(1).fillna(0).eq(0))).astype(int)

    # start of a SC segment: SC==1 and previous lap's SC==0 (or missing)
    laps_2021_2022_2023["SC_firstlap"]  = ((laps_2021_2022_2023["SC"].eq(1))  & (g["SC"].shift(1).fillna(0).eq(0))).astype(int)

    # save to csv
    # laps_2021_2022_2023.to_csv('laps_2021_2022_2023_processed.csv', index=False)

    # filter rows with Trackstatus_2 different than 1
    # laps_2021_2022_2023_regular = laps_2021_2022_2023[laps_2021_2022_2023['TrackStatus_2'] == 1.]
    laps_2021_2022_2023_YF = laps_2021_2022_2023[laps_2021_2022_2023['TrackStatus_2'] != 1.]

    return laps_2021_2022_2023_YF

# function that performs the linear regression for regular laps
def perform_linear_regression_regular(laps, printear: bool = True, GP = None):
    # filter by GP if GP is not None
    if GP is not None:
        df_laps = laps[laps['GP'] == GP].copy()
    else:
        df_laps = laps.copy()

    formula = 'LapTime ~ Interval_front + PitIn + PitOut'
    # check if the GP had a SC event
    if df_laps['SC'].sum() == 1:
        formula += ' + SC'
        # check if the GP had a SC lap that is not first lap
        if df_laps['SC_firstlap'].sum() < df_laps['SC'].sum():
            formula += ' + SC_firstlap'
    # check if the GP had a VSC event
    if df_laps['VSC'].sum() == 1:
        formula += ' + VSC'
        # check if the GP had a VSC lap that is not first lap
        if df_laps['VSC_firstlap'].sum() < df_laps['VSC'].sum():
            formula += ' + VSC_firstlap'
    formula = 'LapTime ~ Interval_front + SC + VSC + SC_firstlap + VSC_firstlap + PitIn + PitOut'
    model = smf.ols(formula=formula, data=df_laps)
    results = model.fit()
    if printear:
        print(results.summary())
    return results

# function that performs the linear regression for regular laps
def perform_linear_regression_YF(laps, printear: bool = True, gp = None):
    # filter by GP if GP is not None
    if gp is not None:
        df_laps = laps[laps['GP'] == gp].copy()
    else:
        df_laps = laps.copy()

    # print the maximum value of PitIn and PitOut
    # print("Max PitIn:", df_laps['PitIn'].max())
    # print("Max PitOut:", df_laps['PitOut'].max())

    formula = 'LapTime ~ 0'
    # check if there is a PitIn equal to 1
    if df_laps['PitIn'].max() > 0:
        formula += ' + PitIn'
    # check if there is a PitOut equal to 1
    if df_laps['PitOut'].max() > 0:
        formula += ' + PitOut'
    # statistics of column Interval_front
    # print(df_laps['Interval_front'].describe())
    # check if the GP had a SC event
    if df_laps['SC'].sum() > 0:
        formula += ' + SC'
        # check if the GP had a SC lap that is not first lap
        if df_laps['SC_firstlap'].sum() < df_laps['SC'].sum():
            # formula += ' + SC_firstlap:Interval_front'
            formula += ' + SC_firstlap'
    # check if the GP had a VSC event
    if df_laps['VSC'].sum() > 0:
        formula += ' + VSC'
        # check if the GP had a VSC lap that is not first lap
        if df_laps['VSC_firstlap'].sum() < df_laps['VSC'].sum():
            # formula += ' + VSC_firstlap:Interval_front'
            formula += ' + VSC_firstlap'
    # formula = 'LapTime ~ Interval_front + SC + VSC + SC_firstlap + VSC_firstlap + PitIn + PitOut'
    model = smf.ols(formula=formula, data=df_laps)
    results = model.fit()

    # esto es solo para ver si es full rank la matrix X
    X = results.model.exog             # design matrix
    # names = results.model.exog_names   # column names
    # print(X.shape)                 # (n_obs, n_features)

    rank = np.linalg.matrix_rank(X)
    n_cols = X.shape[1]

    # print("Rank(X):", rank)
    # print("Number of columns:", n_cols)

    if rank < n_cols:
        print("The columns of X are linearly dependent (perfect multicollinearity).")
    else:
        print("X has full column rank (no *exact* linear dependence).")

    if printear:
        print(results.summary())
    return results

# function that performs all linear regressions for all GPs of 2024
def perform_all_linear_regressions_YF(laps):
    g = laps.groupby('GP')
    resultados = {}
    for gp_name, gp_data in g:
        print(f'Performing linear regression for GP: {gp_name}')
        resultados[gp_name] = perform_linear_regression_YF(gp_data, printear=True, GP=None)
    return resultados
# function that performs the matrix completion


# results = perform_linear_regression_YF(laps, printear = True, GP = "")

# perform all regular linear regressions for the GPs in 2024
def perform_all_linear_regressons(printear: bool = True):
    # get the list of unique GP names of year 2024
    laps_2024 = pd.read_csv(os.path.join(DATA_DIR, 'session_2024_V2.csv'))
    GP_names_2024 = laps_2024[laps_2024['Year'] == 2024]['GP'].unique()

    df_laps_yf = read_and_preprocess_data()
    # save to csv
    df_laps_yf.to_csv('laps_2021_2022_2023_YF_processed.csv', index=False)

    results = {}
    for gp in GP_names_2024:
        # continue if the gp is not in df_laps_yf['GP'].unique()
        #gp = GP_names_2024[0]
        #gp = 'Belgian Grand Prix'
        # check the maximum value of PitIn and PitOut for that gp
        # print("Max PitIn:", df_laps_yf[df_laps_yf['GP'] == gp]['PitIn'].max())
        # print("Max PitOut:", df_laps_yf[df_laps_yf['GP'] == gp]['PitOut'].max())
        if gp not in df_laps_yf['GP'].unique():
            continue  # skip to the next gp
        print(f'Performing linear regression for GP: {gp}')
        if printear:
            print(f'reg_int\treg_PIn\treg_POu\tyf_SC\tyf_VSC\tyf_SC_f\tyf_VSCf\tyf_PIn\tyf_POu')
        try:
            reg_modelos_gp, reg_df_non_outlier = filtrar_outliers_iqr(df_laps_yf, gp_name=gp)
            reg_intercept = reg_modelos_gp.params['Intercept']
            reg_PitIn     = reg_modelos_gp.params.get('PitIn', np.nan)
            reg_PitOut    = reg_modelos_gp.params.get('PitOut', np.nan)
        except ValueError:
            reg_intercept, reg_PitIn, reg_PitOut = np.nan, np.nan, np.nan

        # see if there is TrackStatus_2 different than 1 in reg_df_non_outlier
        yf_modelos_gp = perform_linear_regression_YF(df_laps_yf, printear = False, gp = gp)
        #hay_SC = df_laps_yf[df_laps_yf['GP'] == gp]['TrackStatus_2'].isin([4.]).any()
        # yf_modelos_gp.params.get('SC')
        # yf_modelos_gp.params
        partial_dict = {'reg_intercept' : reg_intercept,
                        'reg_PitIn'     : reg_PitIn,
                        'reg_PitOut'    : reg_PitOut,
                        'yf_SC'         : yf_modelos_gp.params.get('SC', np.nan),
                        'yf_VSC'        : yf_modelos_gp.params.get('VSC', np.nan),
                        'yf_SC_firstlap': yf_modelos_gp.params.get('SC_firstlap', np.nan),
                        'yf_VSC_firstlap': yf_modelos_gp.params.get('VSC_firstlap', np.nan),
                        'yf_PitIn'      : yf_modelos_gp.params.get('PitIn', np.nan),
                        'yf_PitOut'     : yf_modelos_gp.params.get('PitOut', np.nan)}
        if printear:
            print(f"{partial_dict['reg_intercept']:.4f}\t{partial_dict['reg_PitIn']:.4f}\t{partial_dict['reg_PitOut']:.4f}\t{partial_dict['yf_SC']:.4f}\t{partial_dict['yf_VSC']:.4f}\t{partial_dict['yf_SC_firstlap']:.4f}\t{partial_dict['yf_VSC_firstlap']:.4f}\t{partial_dict['yf_PitIn']:.4f}\t{partial_dict['yf_PitOut']:.4f}")
        results[gp] = partial_dict

    print(results)
    # pasar el diccionario a un dataframe
    results_df = pd.DataFrame.from_dict(results, orient='index')
    print(results_df)
    # save to csv
    results_df.to_csv('regression_results_reg_YF_2024.csv')
    return results_df

# performs a matrix completion in the output df of permor_all_linear_regressons
def matrix_completion_regressions_YF(df_reg_results, max_rank: int = 1):
    # read the .csv file regression_results_reg_YF_2024.csv
    df_reg_results = pd.read_csv('regression_results_reg_YF_2024.csv', index_col=0)

    imputer_fixed = SoftImpute(max_rank=max_rank, init_fill_method="zero", max_iters=100, convergence_threshold=1e-5, verbose=False)
    X_fixed_completed = imputer_fixed.fit_transform(df_reg_results)

    df_reg_results_completed = pd.DataFrame(X_fixed_completed, columns=df_reg_results.columns, index=df_reg_results.index)

    # save to csv
    df_reg_results_completed.to_csv(f'regression_results_reg_YF_2024_completed_rank{max_rank}.csv', index=True)


if __name__ == "__main__":
    resultados_todos_gp_YF = perform_all_linear_regressons()
    # resultados_todos_gp_YF.to_csv('regression_results_YF_2024.csv')
    matrix_completion_regressions_YF(resultados_todos_gp_YF, max_rank=2)
