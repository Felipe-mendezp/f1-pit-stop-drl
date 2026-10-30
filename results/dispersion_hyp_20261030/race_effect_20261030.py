"""2026-10-30 — ¿Hay un efecto piloto-carrera persistente en los residuos de la Ec. 1? Ajusta la Ec. 1 de cada GP como
results/sim_v10eGQ_20261028/laptime_v10_20261027.py (sin los overrides de neumatico) y descompone los residuos:
var(media del residuo por piloto-carrera) frente a lo esperado si fueran iid (sigma^2 / n_vueltas).
    F1_TYRELIFE_2021=v10eGQ python results/dispersion_hyp_20261030/race_effect_20261030.py
"""
import io, os, sys, contextlib, importlib.util
ROOT = os.getcwd(); SV = os.path.join(ROOT, 'results', 'sim_v10eGQ_20261028')
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts'), SV]
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
import models.regressions as R
import common_v10_20261027 as C
C.patch_regressions(R)
spec = importlib.util.spec_from_file_location('p0', os.path.join(ROOT, 'results', 'tyre_coef_study_20260930', 'pooled_20261004.py'))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
GPS = ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Miami Grand Prix', 'Hungarian Grand Prix',
       'Belgian Grand Prix', 'Dutch Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']
dh = P.hist_laps_A(); rows = []; allr = []
for gp in GPS:
    with contextlib.redirect_stdout(io.StringIO()):
        lreg, _ = R.filtrar_outliers_iqr(dh, gp)
    df = lreg.model.data.frame.copy(); df['res'] = lreg.resid
    yr = 'Year' if 'Year' in df else None
    key = ['Driver'] + ([yr] if yr else [])
    g = df.groupby(key).res; n = g.size(); m = g.mean()
    s2 = df.res.var(); keep = n >= 10
    # componente entre piloto-carrera: var(medias) - E[s2_within/n]
    within = g.var()[keep]; vb = m[keep].var() - (within / n[keep]).mean()
    # autocorrelacion de orden 1 dentro del piloto-carrera (persistencia de vuelta a vuelta)
    srt = df.sort_values(key + (['LapNumber'] if 'LapNumber' in df else []))
    lag = srt.groupby(key).res.shift(1); ac = np.corrcoef(srt.res[lag.notna()], lag[lag.notna()])[0, 1]
    mm = m[keep].rename('mean').reset_index(); mm['n'] = n[keep].values; mm['w'] = (within / n[keep]).values; mm['GP'] = gp
    allr.append(mm)
    rows.append(dict(GP=gp[:-11], cols=','.join(key), n_laps=len(df), n_dr=int(keep.sum()), sigma=np.sqrt(s2),
                     sd_race_effect=np.sqrt(max(vb, 0)), sd_mean_iid=np.sqrt((within / n[keep]).mean()), ac1=ac,
                     sd_total_effect_s=np.sqrt(max(vb, 0)) * n[keep].mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(lreg.model.formula if hasattr(lreg.model, 'formula') else '')

# efecto piloto-carrera neto del efecto piloto-temporada (media del piloto en el anio sobre los 8 GP)
A = pd.concat(allr); A['season'] = A.groupby(['Driver', 'Year'])['mean'].transform('mean')
A['k'] = A.groupby(['Driver', 'Year'])['mean'].transform('size'); A = A[A.k >= 3]
A['dev'] = (A['mean'] - A.season) * np.sqrt(A.k / (A.k - 1))      # correccion por estimar la media de temporada
vr = A.dev.var() - A.w.mean(); vs = A.groupby(['Driver', 'Year']).season.first().var()
print(f"n piloto-carrera {len(A)} | sd efecto carrera neto de temporada {np.sqrt(max(vr,0)):.3f} s/vuelta | "
      f"sd efecto temporada {np.sqrt(vs):.3f} | sd iid de la media {np.sqrt(A.w.mean()):.3f}")
print(A.groupby('GP').dev.std().round(3).to_string())
A.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'race_effect_means_20261030.csv'), index=False)
