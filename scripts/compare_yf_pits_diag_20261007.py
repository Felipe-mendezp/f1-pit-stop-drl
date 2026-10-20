"""
2026-10-07 — Compara el diagnóstico de paradas bajo amarilla (diag_yf_pits_20261007.py)
entre la física publicada (results/yf_pits_diag_20261007/) y la física con SC revelado
antes de decidir (results/yf_pits_diag_20261007_reveal/, env_paper_v3_20261007.sh).

Solo lee CSV. Uso: python scripts/compare_yf_pits_diag_20261007.py
"""
import os, glob
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
RUNS = {'publicado': 'yf_pits_diag_20261007', 'reveal': 'yf_pits_diag_20261007_reveal'}


def load(kind, fig):
    out = []
    for run, d in RUNS.items():
        for f in glob.glob(os.path.join(ROOT, d, f'{kind}_fig{fig}_*.csv')):
            try:
                df = pd.read_csv(f)
            except pd.errors.EmptyDataError:   # GP sin eventos amarillos
                continue
            df['gp'] = os.path.basename(f).split(f'fig{fig}_')[1][:-4]
            df['run'] = run
            out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    pd.set_option('display.width', 200)
    for fig in (6, 7):
        ev, laps = load('events', fig), load('laps', fig)
        if ev.empty:
            continue
        print(f'\n===== Fig {fig}: eventos amarillos (promedio sobre eventos y GP) =====')
        cols = ['ag_exe_first', 'ag_exe_later', 'ag_req_masked', 'ag_pit_post2', 'riv_share_first']
        print(ev.groupby(['typ', 'policy', 'run'])[cols].mean().round(3).unstack('run').to_string())

        # Proporción de paradas ejecutadas que caen bajo amarilla
        laps['yf'] = (laps.sc + laps.vsc) > 0
        ag = laps[laps.exe_pit == 1].groupby(['policy', 'run']).yf.mean()
        riv = (laps.assign(ry=laps.riv_pits * laps.yf).groupby(['policy', 'run'])
               .agg(ry=('ry', 'sum'), r=('riv_pits', 'sum')))
        tab = pd.DataFrame({'agente_pits_bajo_YF': ag, 'rivales_pits_bajo_YF': riv.ry / riv.r})
        print(f'\n-- Fig {fig}: share de paradas bajo amarilla --')
        print(tab.round(3).unstack('run').to_string())

        # Posición final media por política (una fila por carrera)
        fin = laps.drop_duplicates(['run', 'gp', 'policy', 'sim'])
        pos = fin.groupby(['gp', 'policy', 'run']).final_position.mean().unstack('run')
        print(f'\n-- Fig {fig}: posición final media --')
        print(pos.round(2).unstack('policy').to_string())
        print(fin.groupby(['policy', 'run']).final_position.mean().round(2).unstack('run').to_string())


if __name__ == '__main__':
    main()
