## test matrix completion
import os
from fancyimpute import SoftImpute
import pandas as pd
import numpy as np

from config import DATA_DIR

np.random.seed(123)  # controla inicializaciones aleatorias aguas abajo

def fun_example():
    # matriz con NaN en las posiciones faltantes
    X_incomplete = np.array([
        [1, 2, np.nan, 4],
        [5, np.nan, 7, 8],
        [9, 10, 11, np.nan]
    ])

    # aplicar SoftImpute
    X_completed = SoftImpute().fit_transform(X_incomplete)

    print(X_completed)

def fun_real_data():
    # read selected_coefs.xlsx
    df = pd.read_excel('selected_coefs.xlsx', index_col=0)

    # cols_fixed = ["[C"+str(i)+"]" for i in range(6)]
    # cols_variable = ["TyreLife*[C"+str(i)+"]" for i in range(6)]

    cols_fixed = ["[C"+str(i)+"]" for i in range(1, 6)]
    cols_variable = ["TyreLife*[C"+str(i)+"]" for i in range(1, 6)]

    df_fixed = df[cols_fixed]
    df_variable = df[cols_variable]

    # print first row of df_fixed
    print(df_fixed.iloc[0])


    # aplicar SoftImpute with a single latent factor
    X_fixed_completed = SoftImpute(max_rank=1).fit_transform(df_fixed)
    X_variable_completed = SoftImpute(max_rank=1).fit_transform(df_variable)
    # X_fixed_completed = SoftImpute().fit_transform(df_fixed)
    # X_variable_completed = SoftImpute().fit_transform(df_variable)

    # create a dataframe as df_fixed with the completed values
    df_fixed_completed = pd.DataFrame(X_fixed_completed, columns=cols_fixed, index=df_fixed.index)
    df_variable_completed = pd.DataFrame(X_variable_completed, columns=cols_variable, index=df_variable.index)

    print(df_fixed_completed)
    print(df_variable_completed)

    # table of 2024 compounds
    map_compound_2024 = {
        'Bahrain Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Saudi Arabian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Australian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Japanese Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Chinese Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Miami Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Emilia Romagna Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Monaco Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Canadian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Spanish Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Austrian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'British Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Hungarian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Belgian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Dutch Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Italian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Azerbaijan Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Singapore Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'United States Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Mexico City Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Sao Paulo Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Las Vegas Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Abu Dhabi Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    }

    # columnas deseadas
    compounds = [f"C{i}" for i in range(6)]

    # armar registros
    rows = []
    for gp, comp_map in map_compound_2024.items():
        row = {"GP": gp}
        for c in compounds:
            row[c] = "x" if c in comp_map.values() else ""
        rows.append(row)

    df = pd.DataFrame(rows, columns=["GP"] + compounds)
    # sort by GP
    df = df.sort_values(by="GP").reset_index(drop=True)

    # guardar a Excel
    # df.to_excel("compounds_2024.xlsx", index=False)
    # print(df.head())

    # save to excel
    with pd.ExcelWriter('completed_coefs3.xlsx') as writer:
        df_fixed_completed.to_excel(writer, sheet_name='fixed')
        df_variable_completed.to_excel(writer, sheet_name='variable')


def check_compatibility(dict_compound_coef, new_compound, new_coef, fixed = True):
    """
    dict_compound_coef  : dict with compound as key and coef as value
    new_compound        : compound to check
    new_coef            : coef to check
    fixed               : if True, check for fixed coef compatibility, else for variable coef compatibility

    return              : True if compatible, False otherwise
    """
    # check that the new compound is not in the dictionary, o/w error
    if new_compound in dict_compound_coef:
        raise ValueError(f"Compound {new_compound} is already in the dictionary")
    # if the dictionary has no values, then return True
    if len(dict_compound_coef) == 0:
        return True

    # iterate on the keys of the dictionary
    for compound, coef in dict_compound_coef.items():
        if fixed:
            if new_compound < compound: # new compound is harder than compound
                # new_coef must be greater than coef, o/w is not compatible
                if new_coef <= coef:
                    return False
            else: # new compound is softer than compound
                # new_coef must be less than coef, o/w is not compatible
                if new_coef >= coef:
                    return False
        else:
            if new_compound < compound: # new compound is harder than compound
                # new_coef must be less than coef, o/w is not compatible
                if new_coef >= coef:
                    return False
            else: # new compound is softer than compound
                # new_coef must be greater than coef, o/w is not compatible
                if new_coef <= coef:
                    return False
    return True

def df_to_dict(df):
    '''
    Converts a dataframe with columns of tire indexes and rows of Grand Prix names,
    into a dictionary with Grand Prix names as keys and dictionaries of tire indexes and their values as values.
    '''
    result_dict = {}
    for index, row in df.iterrows():
        gp_name = index
        tire_dict = {}
        for col in df.columns:
            if not pd.isna(row[col]):
                tire_dict[col] = row[col]
        result_dict[gp_name] = tire_dict
    return result_dict

def dict_to_df(dict):
    '''
    Converts a dictionary with Grand Prix names as keys and dictionaries of tire indexes and their values as values,
    into a dataframe with columns of tire indexes and rows of Grand Prix names.
    '''
    df = pd.DataFrame.from_dict(dict, orient='index')
    # sort by row index
    df = df.sort_index()
    # sort columns alphabetically
    df = df.reindex(sorted(df.columns), axis=1)
    return df

# function that substracts one component, i.e., "C1" -> "C0"
def subtract_component(compound):
    if compound.startswith("C"):
        index = int(compound[1:])
        if index > 0:
            return "C" + str(index - 1)
    return None

# function that adds one component, i.e., "C1" -> "C2"
def add_component(compound):
    if compound.startswith("C"):
        index = int(compound[1:])
        if index < 5:
            return "C" + str(index + 1)
    return None

def print_diagnostic(title, data, map_compound_2024=None):
    """Imprime diagnostico de cuantos GPs y que datos tienen."""
    print(f"\n{'='*60}")
    print(f"DIAGNOSTICO: {title}")
    print(f"{'='*60}")

    if isinstance(data, pd.DataFrame):
        print(f"Total GPs: {len(data)}")
        print(f"GPs: {list(data.index if hasattr(data, 'index') else data['GP'].unique())[:10]}...")
        print(f"Columnas: {list(data.columns)}")
        print(f"NaN por columna:\n{data.isna().sum()}")
        print(f"Filas sin ningun NaN: {(~data.isna().any(axis=1)).sum()}")
        if map_compound_2024:
            gps_2024 = set(map_compound_2024.keys())
            gps_in_data = set(data.index) if hasattr(data, 'index') else set(data['GP'].unique())
            print(f"GPs de 2024 presentes: {len(gps_2024 & gps_in_data)}/{len(gps_2024)}")
            print(f"GPs de 2024 faltantes: {gps_2024 - gps_in_data}")
    elif isinstance(data, dict):
        print(f"Total GPs: {len(data)}")
        for gp, compounds in data.items():
            print(f"  {gp}: {list(compounds.keys())}")
    print(f"{'='*60}\n")


def fun_real_data_v2():
    # columnas de coeficientes variables
    #cols_C0_C5 = ["C"+str(i)+"" for i in range(0, 6)]
    #cols_C1_C5 = ["C"+str(i)+"" for i in range(1, 6)]
    cols_components = ["C"+str(i)+"" for i in range(0, 6)]

    # table of 2024 compounds
    # NOTA: Chinese Grand Prix excluido porque no tiene datos historicos (2021-2023)
    map_compound_2024 = {
        'Bahrain Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Saudi Arabian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Australian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Japanese Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        # 'Chinese Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},  # Excluido: sin datos historicos
        'Miami Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Emilia Romagna Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Monaco Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Canadian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Spanish Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Austrian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'British Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Hungarian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Belgian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Dutch Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
        'Italian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Azerbaijan Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Singapore Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'United States Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
        'Mexico City Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Sao Paulo Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Las Vegas Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
        'Abu Dhabi Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    }

    # read compounds_coefs_V2.xlsx
    df = pd.read_excel(os.path.join(DATA_DIR, "compounds_coefs_V3.xlsx"), index_col=0)
    print(df.head())
    print(df.columns.tolist())

    # DIAGNOSTICO 1: Archivo de entrada
    print_diagnostic("1. Archivo compounds_coefs_V2.xlsx (entrada)",
                     pd.DataFrame({'GP': df['GP'].unique()}), map_compound_2024)

    # erase column "std err"
    # df = df.drop(columns=['std err'])

    # keep only rows that start with "C" in column Variable
    df_fixed = df[df['Variable'].str.startswith('C')].copy()

    # keep only rows that start with "TyreLife*" in column Variable
    df_variable = df[df['Variable'].str.startswith('T')].copy()
    df_variable["Variable"] = df_variable["Variable"].str[-3:-1]

    # arreglar nombre de columna Variable (compuesto) y agregar fila con compuesto de referencia

    # 1) Extraer por-GP el codigo de referencia (despues de reference='...')
    df_fixed["ref_code"] = df_fixed["Variable"].str.extract(r"reference='(C\d+)'", expand=True)
    gp_ref = (df_fixed[["GP", "ref_code"]].dropna().drop_duplicates(subset=["GP"]))  # asumimos 1 ref_code por GP

    # 2) Construir filas nuevas (una por GP) con Variable=ref_code, Coef=0, p-value=-1
    new_rows = (gp_ref.rename(columns={"ref_code": "Variable"}).assign(Coef=0, **{"p-value": -1}))

    # 3) En las filas antiguas: dejar Variable como (antepenultimo + penultimo) caracter.
    #    Preferimos extraer "C#" con regex; si no, fallback con slicing [-3:-1].
    old = df_fixed.drop(columns=["ref_code"]).copy()
    # Intento con regex [T.C#]
    extracted = old["Variable"].str.extract(r"\[T\.(C\d+)\]", expand=True)[0]
    # Fallback: antepenultimo + penultimo (siempre que el string tenga largo suficiente)
    fallback = old["Variable"].str[-3:-1]
    old["Variable"] = extracted.fillna(fallback)

    # 4) Concatenar y (opcional) evitar duplicados por (GP, Variable)
    df_fixed = pd.concat([old, new_rows], ignore_index=True)
    df_fixed = df_fixed.drop_duplicates(subset=["GP", "Variable"], keep="first")

    # df_fixed es el resultado final
    # pasar a .csv
    # df_fixed.to_csv("compounds_coefs_V2_expanded.csv", index=False)

    # pasar a wide los valores de Coef (Fixed)
    df_wide_fixed = df_fixed.pivot_table(
        index="GP",
        columns="Variable",
        values="Coef",
        aggfunc="first"   # o 'mean' si hubiera duplicados GP-Variable
    ).reset_index()

    # make GP the index, and erase column
    df_wide_fixed = df_wide_fixed.set_index("GP")
    df_wide_fixed = df_wide_fixed.drop(columns=["GP"], errors="ignore")

    # (Opcional) ordenar columnas por el orden natural C0, C1, C2, ...
    df_wide_fixed = df_wide_fixed[cols_components]

    # remover GP que no esten en 2024
    df_wide_fixed = df_wide_fixed[df_wide_fixed.index.isin(map_compound_2024.keys())]

    # DIAGNOSTICO 2: df_wide_fixed despues de filtrar por GPs 2024
    print_diagnostic("2. df_wide_fixed (despues de filtrar GPs 2024)", df_wide_fixed, map_compound_2024)

    # aplicar un offset en las filas en que el C3 no sea 0, para que lo sea
    dict_offset_fixed = {}
    for index, row in df_wide_fixed.iterrows():
        #index = "Japanese Grand Prix"
        GP = index
        print(f"Processing GP: {GP}")
        offset = row["C3"]
        if not pd.isna(row["C3"]):
            dict_offset_fixed[GP] = offset
        else:
            # search for the column with value 0 in the row
            col_0 = row[row == 0].index.tolist()[0]
            print(f"\tColumn with value 0: {col_0}")
            # take the average of the values of the column col_0 excluding NaN and excluding rows where column "C3" is NaN
            mean_coef_col_0 = df_wide_fixed[col_0][df_wide_fixed["C3"].notna()].mean()
            # mean_coef_col_0 = df_wide_fixed[col_0].mean()
            offset = mean_coef_col_0
            dict_offset_fixed[GP] = offset
            print(f"\tUsing mean of column {col_0} as offset: {mean_coef_col_0}")
        # restar offset a todos los valores de la fila
        for col in cols_components:
            if not pd.isna(row[col]):
                df_wide_fixed.at[GP, col] = row[col] - offset

    print(f"\ndf_wide_fixed")
    print(df_wide_fixed)

    # pasar a wide los valores de Coef (variables)
    df_wide_variable = df_variable.pivot_table(
        index="GP",
        columns="Variable",
        values="Coef",
        aggfunc="first"   # o 'mean' si hubiera duplicados GP-Variable
    ).reset_index()

    # make GP the index, and erase column
    df_wide_variable = df_wide_variable.set_index("GP")
    df_wide_variable = df_wide_variable.drop(columns=["GP"], errors="ignore")

    # (Opcional) ordenar columnas por el orden natural C0, C1, C2, ...
    df_wide_variable = df_wide_variable[cols_components]

    # remover GP que no esten en 2024
    df_wide_variable = df_wide_variable[df_wide_variable.index.isin(map_compound_2024.keys())]

    # DIAGNOSTICO 3: df_wide_variable antes de filtrar negativos
    print_diagnostic("3. df_wide_variable (antes de filtrar negativos)", df_wide_variable, map_compound_2024)

    # replace negative values with NaN
    n_negative = (df_wide_variable < 0).sum().sum()
    print(f"\n*** Valores negativos encontrados (seran NaN): {n_negative} ***")
    df_wide_variable = df_wide_variable.where(df_wide_variable >= 0, np.nan)

    # DIAGNOSTICO 4: df_wide_variable despues de filtrar negativos
    print_diagnostic("4. df_wide_variable (despues de filtrar negativos)", df_wide_variable, map_compound_2024)

    print(f"\ndf_wide_variable:")
    print(df_wide_variable)

    # construir dataframe con valores pivotes, con NaN en todas las posiciones excepto en las que haya 0 para coeficientes fijos
    df_pivotes_fixed = df_wide_fixed.copy()
    # fill all values with nan except values that are equal to 0
    df_pivotes_fixed = df_pivotes_fixed.where(df_pivotes_fixed == 0, np.nan)
    print(df_pivotes_fixed)

    df_pivotes_variable = df_wide_variable.copy()
    # fill all values with nan except values that are equal to 0
    df_pivotes_variable = df_pivotes_variable.where(df_pivotes_variable == 0, np.nan)
    print(df_pivotes_variable)

    # anadir coeficientes compatibles a df_pivotes_fixed, recorriendo p-valores en orden creciente
    for index, row in df_pivotes_fixed.iterrows():
        # index = "Austrian Grand Prix"
        # nombre del GP
        GP = index
        print(f"Processing GP: {GP}")
        # buscar los p-valores correspondientes en el dataframe original df_fixed
        df_pvalues = df_fixed[df_fixed["GP"] == GP].sort_values(by="p-value")
        print(f"P-values for GP {GP}:\n{df_pvalues}")
        dict_compound_coef = {}
        for index_pval, row_pval in df_pvalues.iterrows():
            new_compound = row_pval["Variable"]
            new_coef = row_pval["Coef"] - dict_offset_fixed.get(GP, 0) # aca de aplica el offset
            if new_compound in cols_components:
                is_compatible = check_compatibility(dict_compound_coef, new_compound, new_coef, fixed = True)
                if is_compatible:
                    dict_compound_coef[new_compound] = new_coef
                    print(f"\tAdded compound {new_compound} with coef {new_coef}")
                else:
                    print(f"\tCompound {new_compound} with coef {new_coef} is not compatible")

        # rellenar los valores de df_pivotes con los coeficientes compatibles
        for compound, coef in dict_compound_coef.items():
            df_pivotes_fixed.at[GP, compound] = coef

    print(f"\ndf_pivotes_fixed:")
    print(df_pivotes_fixed)
    dict_fixed = df_to_dict(df_pivotes_fixed)

    # DIAGNOSTICO 5: dict_fixed despues de filtrar por compatibilidad
    print_diagnostic("5. dict_fixed (despues de check_compatibility)", dict_fixed, map_compound_2024)

    # Contar cuantos compuestos tiene cada GP vs cuantos necesita para 2024
    print("\n*** Analisis de compuestos faltantes (FIXED) ***")
    for gp, comp_map in map_compound_2024.items():
        needed = set(comp_map.values())
        have = set(dict_fixed.get(gp, {}).keys())
        missing = needed - have
        if missing:
            print(f"  {gp}: tiene {have}, necesita {needed}, FALTAN {missing}")

    # anadir coeficientes compatibles a df_pivotes_variable, recorriendo p-valores en orden creciente
    print("\nanadir coeficientes compatibles a df_pivotes_variable, recorriendo p-valores en orden creciente\n")
    for index, row in df_pivotes_variable.iterrows():
        GP = index
        print(f"Processing GP: {GP}")
        # buscar los p-valores correspondientes en el dataframe original df_fixed
        df_pvalues = df_variable[df_variable["GP"] == GP].sort_values(by="p-value")
        print(f"P-values for GP {GP}:\n{df_pvalues}")
        dict_compound_coef = {}
        for index_pval, row_pval in df_pvalues.iterrows():
            new_compound = row_pval["Variable"]
            new_coef = df_wide_variable.at[GP, new_compound]
            if pd.isna(new_coef):
                continue
            if new_compound in cols_components:
                is_compatible = check_compatibility(dict_compound_coef, new_compound, new_coef, fixed = False)
                # print("\tCompatibility:", is_compatible)
                # print("\tdict_compound_coef:", dict_compound_coef)
                # print("\tnew_compound:", new_compound)
                # print("\tnew_coef:", new_coef)
                if is_compatible:
                    dict_compound_coef[new_compound] = new_coef
                    print(f"\tAdded compound {new_compound} with coef {new_coef}")
                else:
                    print(f"\tCompound {new_compound} with coef {new_coef} is not compatible")

        # rellenar los valores de df_pivotes con los coeficientes compatibles
        for compound, coef in dict_compound_coef.items():
            df_pivotes_variable.at[GP, compound] = coef

    print(f"\ndf_pivotes_variable:")
    print(df_pivotes_variable)

    dict_variable = df_to_dict(df_pivotes_variable)

    # DIAGNOSTICO 6: dict_variable despues de filtrar por compatibilidad
    print_diagnostic("6. dict_variable (despues de check_compatibility)", dict_variable, map_compound_2024)

    # Contar cuantos compuestos tiene cada GP vs cuantos necesita para 2024
    print("\n*** Analisis de compuestos faltantes (VARIABLE) ***")
    for gp, comp_map in map_compound_2024.items():
        needed = set(comp_map.values())
        have = set(dict_variable.get(gp, {}).keys())
        missing = needed - have
        if missing:
            print(f"  {gp}: tiene {have}, necesita {needed}, FALTAN {missing}")


    # aplicar SoftImpute with a single latent factor
    imputer_fixed = SoftImpute(max_rank=1, init_fill_method="zero", max_iters=100, convergence_threshold=1e-5, verbose=False)
    X_fixed_completed = imputer_fixed.fit_transform(df_pivotes_fixed)

    imputer_var = SoftImpute(max_rank=1, init_fill_method="zero", max_iters=100, convergence_threshold=1e-5, verbose=False)
    X_variable_completed = imputer_var.fit_transform(df_pivotes_variable)

    # create a dataframe as df_fixed with the completed values
    df_fixed_completed = pd.DataFrame(X_fixed_completed, columns=cols_components, index=df_pivotes_fixed.index)
    df_variable_completed = pd.DataFrame(X_variable_completed, columns=cols_components, index=df_pivotes_variable.index)

    # remove the offsets from df_fixed_completed
    for index, row in df_fixed_completed.iterrows():
        GP = index
        offset = dict_offset_fixed.get(GP, 0)
        for col in cols_components:
            df_fixed_completed.at[GP, col] = row[col] + offset

    # print the first 3 decimals only
    print(df_fixed_completed.round(3))
    print(df_variable_completed.round(3))

    # iterate on GP of 2024
    for gp, comp_map in map_compound_2024.items():
        # gp = 'Abu Dhabi Grand Prix'
        # comp_map = map_compound_2024[gp]
        print(f"GP: {gp}")
        if not gp in dict_fixed:
            continue
        # for fixed compounds
        gp_2024_compounds = comp_map.values()
        gp_2024_compounds_to_fill = [c for c in gp_2024_compounds if not c in list(dict_fixed.get(gp, {}).keys())]
        # fill compounds with matrix completion values that are compatible
        for c in gp_2024_compounds_to_fill:
            # c = gp_2024_compounds_to_fill[0]
            # check if c is consistent with dict_fixed[gp]
            is_compatible_fixed = check_compatibility(dict_fixed.get(gp, {}), c, df_fixed_completed.at[gp, c], fixed = True)
            if is_compatible_fixed:
                dict_fixed.setdefault(gp, {})[c] = df_fixed_completed.at[gp, c]
                print(f"\t\tFilled fixed coef for compound {c} with value {df_fixed_completed.at[gp, c]}")

        # fill compounds with average of neighbours of matrix completion if its compatible
        gp_2024_compounds_to_fill = [c for c in gp_2024_compounds if not c in list(dict_fixed.get(gp, {}).keys())]
        for c in gp_2024_compounds_to_fill:
            if c != "C0" and c != "C5":
                c_minus_1 = subtract_component(c)
                c_plus_1 = add_component(c)
                if c_minus_1 in dict_fixed.get(gp, {}) and c_plus_1 in dict_fixed.get(gp, {}):
                    avg_coef = (dict_fixed[gp][c_minus_1] +  dict_fixed[gp][c_plus_1]) / 2
                    is_compatible_fixed = check_compatibility(dict_fixed.get(gp, {}), c, avg_coef, fixed = True)
                    if is_compatible_fixed:
                        dict_fixed.setdefault(gp, {})[c] = avg_coef
                        print(f"\t\tFilled fixed coef for compound {c} with average value {avg_coef}")

        # for variable compounds
        gp_2024_compounds = comp_map.values()
        gp_2024_compounds_to_fill = [c for c in gp_2024_compounds if not c in list(dict_variable.get(gp, {}).keys())]
        # fill compounds with matrix completion values that are compatible
        for c in gp_2024_compounds_to_fill:
            # c = gp_2024_compounds_to_fill[0]
            # check if c is consistent with dict_fixed[gp]
            is_compatible_variable = check_compatibility(dict_variable.get(gp, {}), c, df_variable_completed.at[gp, c], fixed = False)
            if is_compatible_variable:
                dict_variable.setdefault(gp, {})[c] = df_variable_completed.at[gp, c]
                print(f"\t\tFilled variable coef for compound {c} with value {df_variable_completed.at[gp, c]}")

        # fill compounds with average of neighbours of matrix completion if its compatible
        gp_2024_compounds_to_fill = [c for c in gp_2024_compounds if not c in list(dict_variable.get(gp, {}).keys())]
        for c in gp_2024_compounds_to_fill:
            if c != "C0" and c != "C5":
                c_minus_1 = subtract_component(c)
                c_plus_1 = add_component(c)
                if c_minus_1 in dict_variable.get(gp, {}) and c_plus_1 in dict_variable.get(gp, {}):
                    avg_coef = (dict_variable[gp][c_minus_1] +  dict_variable[gp][c_plus_1]) / 2
                    is_compatible_variable = check_compatibility(dict_variable.get(gp, {}), c, avg_coef, fixed = False)
                    if is_compatible_variable:
                        dict_variable.setdefault(gp, {})[c] = avg_coef
                        print(f"\t\tFilled variable coef for compound {c} with average value {avg_coef}")

    df_fixed_final = dict_to_df(dict_fixed)
    df_variable_final = dict_to_df(dict_variable)
    print(df_fixed_final.round(3))
    print(df_variable_final.round(3))

    # DIAGNOSTICO 7: Despues del matrix completion + compatibilidad post-imputacion
    print_diagnostic("7. dict_fixed FINAL (despues de matrix completion)", dict_fixed, map_compound_2024)
    print_diagnostic("7b. dict_variable FINAL (despues de matrix completion)", dict_variable, map_compound_2024)

    # Analisis detallado de que falta
    print("\n*** RESUMEN FINAL: GPs con 3 compuestos completos ***")
    gp_fixed_complete = []
    gp_variable_complete = []
    for gp, comp_map in map_compound_2024.items():
        needed = set(comp_map.values())
        have_fixed = set(dict_fixed.get(gp, {}).keys())
        have_variable = set(dict_variable.get(gp, {}).keys())

        fixed_ok = needed.issubset(have_fixed)
        variable_ok = needed.issubset(have_variable)

        if fixed_ok:
            gp_fixed_complete.append(gp)
        if variable_ok:
            gp_variable_complete.append(gp)

        status_fixed = "OK" if fixed_ok else f"MISSING (faltan {needed - have_fixed})"
        status_variable = "OK" if variable_ok else f"MISSING (faltan {needed - have_variable})"
        print(f"  {gp}: FIXED {status_fixed}, VARIABLE {status_variable}")

    print(f"\nGPs con FIXED completo: {len(gp_fixed_complete)}/23")
    print(f"GPs con VARIABLE completo: {len(gp_variable_complete)}/23")
    print(f"GPs con AMBOS completos: {len(set(gp_fixed_complete) & set(gp_variable_complete))}/23")

    # simplemente ver las compuestos de 2024 y cuales faltan
    df_fixed_final["C_2024"] = "" # compuestos de 2024
    df_fixed_final["C_2024_m"] = "" # compuestos de 2024 que faltan en df_fixed_final
    # iterae on rows of df_fixed_final
    for index, row in df_fixed_final.iterrows():
        gp = index
        if gp in map_compound_2024:
            comp_map = map_compound_2024[gp]
            df_fixed_final.at[gp, "C_2024"] = comp_map['HARD'] + ", " + comp_map['MEDIUM'] + ", " + comp_map['SOFT']
            df_fixed_final.at[gp, "C_2024_m"] = (comp_map['HARD'] if pd.isna(row[comp_map['HARD']]) else "") + ", " + (comp_map['MEDIUM'] if pd.isna(row[comp_map['MEDIUM']]) else "") + ", " + (comp_map['SOFT'] if pd.isna(row[comp_map['SOFT']]) else "")
    print(df_fixed_final.round(3))

    df_variable_final["C_2024"] = "" # compuestos de 2024
    df_variable_final["C_2024_m"] = "" # compuestos de 2024 que faltan en df_variable_final
    # iterae on rows of df_variable_final
    for index, row in df_variable_final.iterrows():
        gp = index
        if gp in map_compound_2024:
            comp_map = map_compound_2024[gp]
            df_variable_final.at[gp, "C_2024"] = comp_map['HARD'] + ", " + comp_map['MEDIUM'] + ", " + comp_map['SOFT']
            df_variable_final.at[gp, "C_2024_m"] = (", ").join([comp_map['HARD'] if pd.isna(row[comp_map['HARD']]) else "", comp_map['MEDIUM'] if pd.isna(row[comp_map['MEDIUM']]) else "", comp_map['SOFT'] if pd.isna(row[comp_map['SOFT']]) else ""])
    print(df_variable_final.round(3))

    # save to excel
    with pd.ExcelWriter(os.path.join(DATA_DIR, 'completed_coefs_V2_filled.xlsx')) as writer:
        df_fixed_final.to_excel(writer, sheet_name='fixed')
        df_variable_final.to_excel(writer, sheet_name='variable')

    # si quisieramos filtrar los circuitos de 2024 con 3 neumaticos, seria asi
    gp_with_3 = []
    for gp, comp_map in map_compound_2024.items():
        if gp in dict_fixed and gp in dict_variable:
            if all(c in dict_fixed[gp] for c in comp_map.values()) and all(c in dict_variable[gp] for c in comp_map.values()):
                gp_with_3.append(gp)
    print(f"\ngp_with_3 = {gp_with_3}")
    print(f"\nNumber of GPs with 3 compounds filled: {len(gp_with_3)}")


# create main
if __name__ == "__main__":
    # fun_example()
    # fun_real_data()
    fun_real_data_v2()
