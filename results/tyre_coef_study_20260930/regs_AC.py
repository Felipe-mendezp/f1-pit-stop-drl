"""(A): OLS Ec.1 excluyendo ediciones (Year,GP) con >WET_MAX de vueltas WET/INT.
Uso: python regs_AC.py <outdir> <wet_max>   (wet_max=1.01 -> baseline sin exclusion)"""
import sys, os
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'src'))
import config, compat; compat.setup_all()
import pandas as pd, numpy as np
import models.regressions as R

out, WET_MAX = sys.argv[1], float(sys.argv[2])
orig_filter = R.filter_laps
def filter_laps_A(laps):
    share = laps.groupby(['Year', 'GP'])['Compound'].transform(lambda x: x.isin(['WET', 'INTERMEDIATE']).mean())
    ex = laps.loc[share > WET_MAX, ['Year', 'GP']].drop_duplicates()
    if len(ex): print('(A) ediciones excluidas:\n' + ex.to_string(index=False))
    ex.to_csv(os.path.join(out, 'excluded_editions.csv'), index=False)
    return orig_filter(laps[share <= WET_MAX])
R.filter_laps = filter_laps_A

laps = R.all_laps()
laps['GP'] = laps['GP'].replace('Styrian Grand Prix', 'Austrian Grand Prix')
m = laps['GP'] == 'Emilia Romagna Grand Prix'
laps.loc[m, 'Compound_Detail'] = laps.loc[m, 'Compound_Detail'].replace('C2', 'C5')  # igual que train_regressions.py

rows, nlaps, full = [], [], []
for gp in laps.GP.unique():
    try:
        mod, d = R.filtrar_outliers_iqr(laps, gp)
    except Exception as e:
        print('skip', gp, e); continue
    nlaps += [{'GP': gp, 'Compound_Detail': c, 'n': n} for c, n in d.Compound_Detail.value_counts().items()]
    for name in mod.params.index:
        if 'Compound_Detail' in name:
            rows.append({'GP': gp, 'Variable': name, 'Coef': float(mod.params[name]), 'p-value': float(mod.pvalues[name])})
    full.append({'GP': gp, 'LapNumber': float(mod.params['LapNumber']), 'LapNumber_2': float(mod.params['LapNumber_2']), 'maxlap': float(d.LapNumber.max()), **{'se:' + n: float(mod.bse[n]) for n in mod.params.index if 'Compound_Detail' in n}})
pd.DataFrame(rows).to_excel(os.path.join(out, 'compounds_coefs_V3.xlsx'))
pd.DataFrame(nlaps).to_csv(os.path.join(out, 'n_laps_compound.csv'), index=False)
pd.DataFrame(full).drop_duplicates('GP').to_csv(os.path.join(out, 'race_params.csv'), index=False)
print('ok', len(rows))
