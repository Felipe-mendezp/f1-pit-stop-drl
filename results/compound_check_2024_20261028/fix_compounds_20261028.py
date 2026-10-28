"""2026-10-28 — Corrige en memoria el compuesto de las vueltas de carrera 2024 en que FastF1 difiere de OpenF1 y Lapchive
fuera de la vuelta de PitIn (solo Belgica 2024: ALO, RIC, STR, TSU, ultima tanda = HARD = C2). Ver README.md.
    patch_compounds(R)   # R = models.regressions; envuelve R.load_data (componer DESPUES de patch_regressions)
"""
import os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
KEY = ['GP', 'Driver', 'LapNumber']


def fixes():
    b = pd.read_csv(os.path.join(HERE, 'mismatch_laps.csv'))
    b = b[(b.GP == 'Belgian Grand Prix') & (b.PitIn == 0)]
    assert set(b.Driver) == {'ALO', 'RIC', 'STR', 'TSU'} and (b.Compound_openf1 == 'HARD').all(), b.Driver.unique()
    return b[KEY].assign(Compound_fix='HARD', Detail_fix='C2')


def patch_compounds(R):
    if getattr(R, '_compound_patched', False):
        return
    orig, F = R.load_data, fixes()

    def load_data():
        res = list(orig())
        d = res[3].copy()
        m = d[KEY].merge(F, on=KEY, how='left')
        hit = m.Compound_fix.notna().values
        d.loc[hit, 'Compound'] = 'HARD'
        if 'Compound_Detail' in d.columns:
            d.loc[hit, 'Compound_Detail'] = 'C2'
        print(f'[compound fix 2024] {int(hit.sum())} vueltas de Belgica corregidas', flush=True)
        res[3] = d
        return tuple(res)

    R.load_data, R._compound_patched = load_data, True
