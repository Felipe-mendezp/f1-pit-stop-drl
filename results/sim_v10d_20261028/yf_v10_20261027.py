"""2026-10-27 — Modelo de tiempos bajo SC/VSC del simulador v10: el de scripts/fit_yf_model_v4_20261007.py (mismo modelo y
muestra), sin las vueltas de paso por el pit lane sin parada (common_v10_20261027.pass_flags), tambien en la perdida verde
Gin/Gout. Control: CONTROL=1 (sin quitar nada) debe reproducir data/yf_model_v4_20261007.csv. No escribe en data/.
    ~/.venvs/f1-drl-v2/bin/python results/sim_v10_20261027/yf_v10_20261027.py
Salida: results/sim_v10_20261027/yf_model_v10_20261027.csv (mismas columnas que v4) y yf_model_v10_20261027_fit.txt
"""
import os, sys, warnings; warnings.filterwarnings('ignore')
ROOT = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(ROOT, 'scripts'), HERE]
import numpy as np, pandas as pd, statsmodels.formula.api as smf
import fit_yf_model_v4_20261007 as F
import common_v10_20261027 as CV

CONTROL = os.environ.get('CONTROL') == '1'
OUT_CSV = os.path.join(HERE, 'yf_model_v10_20261027.csv'); OUT_TXT = OUT_CSV.replace('.csv', '_fit.txt')


def main():
    d = F.load_laps()
    if not CONTROL:
        m = d.merge(CV.pass_flags((2021, 2022, 2023)), on=CV.KEY, how='left')
        drop = (m.in_pass.fillna(False) | m.out_pass.fillna(False)).values
        print('vueltas de paso quitadas:', int(drop.sum()), 'de', len(d))
        d = d[~drop]
    G = F.green_components(d)
    Lg = d.groupby('GP')['Lref'].median().rename('Lref')
    y = d[d.pure.isin(['SC', 'VSC'])].copy()
    y['VSC'] = (y.pure == 'VSC').astype(int)
    y['SCl'] = ((y.pure == 'SC') & (y.ev_idx == 2)).astype(int)
    y['SCf'] = ((y.pure == 'SC') & (y.ev_idx >= 3)).astype(int)
    y = y[(y.VSC + y.SCl + y.SCf) == 1]
    y = y[~(((y.PitIn + y.PitOut) > 0) & (y.SCf == 1))]
    y = y.join(G[['Gin', 'Gout']], on='GP').dropna(subset=['Gin', 'Gout'])
    y['ratio'] = y.LapTime / y.Lref
    y = y[(y.ratio > 0.9) & (y.ratio < 2.2)]
    y['PI'] = y.PitIn * y.Gin / y.Lref; y['PO'] = y.PitOut * y.Gout / y.Lref
    y['SC'] = y.SCl + y.SCf
    y['rl'] = y.Year.astype(str) + '|' + y.GP + '|' + y.LapNumber.astype(int).astype(str)
    m = smf.mixedlm('ratio ~ 0 + VSC + SCl + SCf + PI + PO', y, groups=y['GP'], re_formula='~0+SC',
                    vc_formula={'rl': '0+C(rl)'}).fit(method='lbfgs')
    fe = m.fe_params; u = {gp: float(r.iloc[0]) for gp, r in m.random_effects.items()}
    rows = []
    for gp in sorted(set(G.dropna().index) & set(Lg.index)):
        L, gi, go, ug = Lg[gp], G.loc[gp, 'Gin'], G.loc[gp, 'Gout'], u.get(gp, 0.0)
        rows.append({'GP': gp, 'Lref': L, 'Gin': gi, 'Gout': go,
                     'n_green_in': int(G.loc[gp, 'n_green_in']), 'n_green_out': int(G.loc[gp, 'n_green_out']),
                     'n_sc_laps': int(y[(y.GP == gp) & (y.SC == 1)].shape[0]),
                     'rho_VSC': fe['VSC'], 'rho_SC_libre': fe['SCl'] + ug, 'rho_SC_fila': fe['SCf'] + ug,
                     'u_SC': ug, 'k_in': fe['PI'], 'k_out': fe['PO'],
                     'PitIn': fe['PI'] * gi, 'PitOut': fe['PO'] * go,
                     'SC': (fe['SCf'] + ug) * L, 'SC_firstlap': (fe['SCl'] - fe['SCf']) * L,
                     'VSC': fe['VSC'] * L, 'VSC_firstlap': 0.0})
    out = pd.DataFrame(rows)
    summ = (f'Muestra: {len(y)} vueltas; VSC {int(y.VSC.sum())}, SC libre {int(y.SCl.sum())}, SC fila {int(y.SCf.sum())}, '
            f'PitIn {int(y.PitIn.sum())}, PitOut {int(y.PitOut.sum())}\n' + str(m.summary().tables[1].iloc[:7].round(4)) +
            f'\nsd efecto GP en SC: {np.sqrt(m.cov_re.iloc[0, 0]):.4f}\nconverged: {m.converged}\n')
    print(summ)
    v4 = pd.read_csv(os.path.join(ROOT, 'data', 'yf_model_v4_20261007.csv')).set_index('GP')
    o = out.set_index('GP')
    cols = ['PitIn', 'PitOut', 'SC', 'SC_firstlap', 'VSC']
    diff = (o[cols] - v4.loc[o.index, cols])
    if CONTROL:
        print('CONTROL max |dif| con v4:', float(diff.abs().max().max())); return
    out.to_csv(OUT_CSV, index=False, float_format='%.6f')
    with open(OUT_TXT, 'w') as f:
        f.write(summ)
    paper = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
             'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']
    t = pd.DataFrame({'loss_v4': v4.loc[paper, 'PitIn'] + v4.loc[paper, 'PitOut'],
                      'loss_v10': o.loc[paper, 'PitIn'] + o.loc[paper, 'PitOut'],
                      'SC_v4': v4.loc[paper, 'SC'], 'SC_v10': o.loc[paper, 'SC']}).round(2)
    print(t.to_string())


if __name__ == '__main__':
    main()
