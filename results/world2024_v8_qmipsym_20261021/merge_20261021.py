"""Figs. 6 y 7 v8 con la QMIP de simetria completa — 2026-10-21.

Toma los CSV de results/world2024_v8_20261019/{fig6_2024_realdnf,fig7_2024} (5 metodos DRL + QMIP), reemplaza las filas
QMIP por las de esta carpeta (reproduce/qmipsym_figs67_20261021.sh, F1_QMIP_SYMMETRY=full) y escribe merged/<fig>/<GP>.csv.
Control: las filas PPO de esta corrida deben ser identicas a las de la corrida v8 (mismas semillas y entorno).
    python results/world2024_v8_qmipsym_20261021/merge_20261021.py
"""
import os, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(os.path.dirname(HERE), 'world2024_v8_20261019')
for fig in ('fig6_2024_realdnf', 'fig7_2024'):
    os.makedirs(os.path.join(HERE, 'merged', fig), exist_ok=True)
    for f in sorted(glob.glob(os.path.join(OLD, fig, '*.csv'))):
        gp = os.path.basename(f)
        old, new = pd.read_csv(f), pd.read_csv(os.path.join(HERE, fig, gp))
        po, pn = (d[d.algo == 'PPO'].reset_index(drop=True) for d in (old, new))
        same = (len(po) == len(pn) and po.final_position.fillna(-1).eq(pn.final_position.fillna(-1)).all()
                and (po.pit_laps.fillna('').astype(str) == pn.pit_laps.fillna('').astype(str)).all()
                and np.allclose(po.total_reward, pn.total_reward, rtol=0, atol=1e-6, equal_nan=True))
        assert same, f'{fig}/{gp}: filas PPO distintas'
        qo, qn = (d[d.algo == 'QMIP'].reset_index(drop=True) for d in (old, new))
        assert len(qo) == len(qn) == 500, (fig, gp, len(qo), len(qn))
        out = pd.concat([old[old.algo != 'QMIP'], qn], ignore_index=True)[old.columns]
        out.to_csv(os.path.join(HERE, 'merged', fig, gp), index=False)
        ch = (qo.final_position != qn.final_position)
        print(f'{fig:18s} {gp[:-4]:20s} QMIP pos. media {qo.final_position.mean():6.3f} -> {qn.final_position.mean():6.3f}'
              f'  carreras con otra posicion: {ch.sum():3d}/500  otras paradas: '
              f'{(qo.pit_laps.fillna('').astype(str) != qn.pit_laps.fillna('').astype(str)).sum():3d}/500')
