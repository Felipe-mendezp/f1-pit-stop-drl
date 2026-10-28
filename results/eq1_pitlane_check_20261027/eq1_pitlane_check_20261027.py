"""2026-10-27 — Ec. 1 (pista limpia) por GP del paper: ¿cambia la estimación si se quitan las vueltas con PitIn/PitOut
sin cambio de juego (pasos por el pit lane sin parar, p. ej. penalizaciones)? Mismo pipeline que
results/sim_v7_20261015/laptime_v7_20261015.py, paso 1 (hist_laps_A + filtrar_outliers_iqr). No escribe nada fuera de
esta carpeta. Sin parada: PitIn = 1 y la vuelta siguiente tiene el mismo compuesto y TyreLife + 1 (PitOut: idem con la
anterior), calculado en data/session_202{1,2,3}_V2.csv sin reescalar.
    F1_TYRELIFE_2021=v7 ~/.venvs/f1-drl-v2/bin/python results/eq1_pitlane_check_20261027/eq1_pitlane_check_20261027.py
"""
import io, os, sys, contextlib, importlib.util, warnings; warnings.filterwarnings('ignore')
ROOT = os.getcwd(); sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts')]
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
import models.regressions as R
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('p0', os.path.join(ROOT, 'results', 'tyre_coef_study_20260930', 'pooled_20261004.py'))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
GPS = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
       'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']


def pass_flags():
    out = []
    for y in (2021, 2022, 2023):
        s = pd.read_csv(os.path.join(ROOT, 'data', f'session_{y}_V2.csv')).sort_values(['GP', 'Driver', 'LapNumber'])
        g = s.groupby(['GP', 'Driver']); n, p = g.shift(-1), g.shift(1)
        s['in_pass'] = (s.PitIn == 1) & (n.LapNumber == s.LapNumber + 1) & (n.Compound_Detail == s.Compound_Detail) & (n.TyreLife == s.TyreLife + 1)
        s['out_pass'] = (s.PitOut == 1) & (p.LapNumber == s.LapNumber - 1) & (p.Compound_Detail == s.Compound_Detail) & (s.TyreLife == p.TyreLife + 1)
        s['Year'] = y
        out.append(s[['Year', 'GP', 'Driver', 'LapNumber', 'in_pass', 'out_pass']])
    f = pd.concat(out); f['GP'] = f.GP.replace('Styrian Grand Prix', 'Austrian Grand Prix')
    return f


def main():
    dh = P.hist_laps_A()
    dh = dh.merge(pass_flags(), on=['Year', 'GP', 'Driver', 'LapNumber'], how='left')
    dh[['in_pass', 'out_pass']] = dh[['in_pass', 'out_pass']].fillna(False)
    rows, cnt = [], []
    for gp in GPS:
        g = dh[(dh.GP == gp) & (dh.TrackStatus == 1.0)]
        with contextlib.redirect_stdout(io.StringIO()):
            m0, d0 = R.filtrar_outliers_iqr(dh, gp)
            m1, _ = R.filtrar_outliers_iqr(dh[~(dh.in_pass | dh.out_pass)], gp)
        kept = d0 if isinstance(d0, pd.DataFrame) else None
        cnt.append(dict(GP=gp, PitIn=int(g.PitIn.sum()), in_pass=int(g.in_pass.sum()), PitOut=int(g.PitOut.sum()),
                        out_pass=int(g.out_pass.sum()),
                        pass_kept_by_iqr=int((kept.in_pass | kept.out_pass).sum()) if kept is not None and 'in_pass' in kept else np.nan,
                        n0=int(m0.nobs), n1=int(m1.nobs)))
        for k in m0.params.index:
            rows.append(dict(GP=gp, coef=k, orig=m0.params[k], se=m0.bse[k], new=m1.params.get(k, np.nan)))
    C = pd.DataFrame(cnt); T = pd.DataFrame(rows); T['diff'] = T.new - T.orig; T['diff_se'] = T['diff'] / T.se
    C.to_csv(os.path.join(HERE, 'counts_20261027.csv'), index=False); T.to_csv(os.path.join(HERE, 'coefs_20261027.csv'), index=False)
    print(C.to_string(index=False))
    key = T[T.coef.str.contains('PitIn|PitOut|Compound_Detail|TyreLife')]
    print('\nPitIn / PitOut:'); print(T[T.coef.isin(['PitIn', 'PitOut'])][['GP', 'coef', 'orig', 'new', 'se']].round(3).to_string(index=False))
    print('\nmax |diff|/se por GP (todos los coeficientes; neumaticos):')
    print(pd.concat([T.groupby('GP').diff_se.apply(lambda x: x.abs().max()).rename('todos'),
                     key[~key.coef.isin(['PitIn', 'PitOut'])].groupby('GP').diff_se.apply(lambda x: x.abs().max()).rename('neumaticos')], axis=1).round(2).to_string())


if __name__ == '__main__':
    main()
