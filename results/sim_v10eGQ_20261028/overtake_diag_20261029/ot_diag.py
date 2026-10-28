"""Diagnostico del logit de adelantamiento v10eGQ: muestra exacta de la estimacion, tasas empiricas por tramo de delta."""
import os, sys, io, contextlib, pickle, warnings; warnings.filterwarnings('ignore')
ROOT = os.getcwd(); CH = os.path.join(ROOT, 'results', 'sim_v10eGQ_20261028')
sys.path[:0] = [ROOT, os.path.join(ROOT, 'src'), CH]
import common_v10_20261027  # noqa: ganchos de compuestos y edad
import config, compat; compat.setup_all()
import numpy as np, pandas as pd
import models.logit_overtakes_pit as O

with contextlib.redirect_stdout(io.StringIO()):
    laps, selected, _ = O.load_data()
    selected = {g: c for g, c in selected.items() if g != 'Emilia Romagna Grand Prix'}
    reg = {}
    for gp in selected:
        with open(os.path.join(CH, 'artifacts', gp, 'laptime_clear_hist.pkl'), 'rb') as fh:
            reg[gp] = pickle.load(fh)
    df = O.optimize_f1_analysis(laps, selected, reg)
full = df.copy()
df = df[(df.LapNumber > 3) & (df.LapNumber < df.LastLap)]
df = df.dropna(subset=['interval_front_real', 'delta_t_lap', 'DRS', 'overtake', 'rival_PitOut'])
df['delta_total'] = df.interval_front_real + df.delta_t_lap
df = df[(df.delta_total >= -30) & (df.delta_total <= 5)]          # filtro efectivo (el de [-30, 30] y luego el de [-30, 5])
# tiempo real de la vuelta L+1 de ambos y estado de pista de ambos
key = full.set_index(['Year', 'GP', 'Driver', 'LapNumber'])
lt = key.LapTime.to_dict(); ts = key.TrackStatus.to_dict(); po = key.PitOut.to_dict(); pi = key.PitIn.to_dict()
pos = full.set_index(['Year', 'GP', 'LapNumber', 'Position']).Driver.to_dict()
riv = [pos.get((y, g, l, p - 1)) for y, g, l, p in zip(df.Year, df.GP, df.LapNumber, df.Position)]
df['rival'] = riv
df['real_dt'] = [lt.get((y, g, d, l + 1), np.nan) - lt.get((y, g, r, l + 1), np.nan) for y, g, d, r, l in zip(df.Year, df.GP, df.Driver, df.rival, df.LapNumber)]
df['delta_real'] = df.interval_front_real + df.real_dt
df['ts_L'] = df.TrackStatus.astype(str)
df['ts_L1'] = [str(ts.get((y, g, d, l + 1))) for y, g, d, l in zip(df.Year, df.GP, df.Driver, df.LapNumber)]
df['green'] = (df.TrackStatus == 1.0)
df['own_pit'] = [(pi.get((y, g, d, l), 0) == 1) or (po.get((y, g, d, l + 1), 0) == 1) for y, g, d, l in zip(df.Year, df.GP, df.Driver, df.LapNumber)]
df['gain'] = df['position_gain/loss'] > 0
print('muestra:', len(df), '| overtakes:', int(df.overtake.sum()))
print('vueltas L no verdes en la muestra:', int((~df.green).sum()), f'({(~df.green).mean():.1%}); ganan posicion al rival pero overtake=0 por no verde:',
      int(((~df.green) & df.gain).sum()))
bins = [-30, -10, -5, -2, 0, 2, 5]
df['bin'] = pd.cut(df.delta_total, bins)
pd.set_option('display.width', 250)
for gp in ['Saudi Arabian Grand Prix', 'Singapore Grand Prix', 'Hungarian Grand Prix', 'Bahrain Grand Prix']:
    g = df[df.GP == gp]
    t = g.groupby('bin', observed=False).agg(n=('overtake', 'size'), tasa=('overtake', 'mean'),
                                              no_verde=('green', lambda x: (~x).mean()),
                                              rival_pitout=('rival_PitOut', 'mean'), propia_parada=('own_pit', 'mean'))
    gg = g[g.green]
    t['tasa_solo_verde'] = gg.groupby('bin', observed=False).overtake.mean()
    t['n_verde'] = gg.groupby('bin', observed=False).size()
    print('\n==', gp); print(t.round(3).to_string())
df.to_pickle(os.path.expanduser('~/.claude/jobs/4608e3ae/tmp/ot_df.pkl'))
full.to_pickle(os.path.expanduser('~/.claude/jobs/4608e3ae/tmp/ot_full.pkl'))
