import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import compat; compat.setup_all()

"""
2026-08-31 — Árbol de simulación con los coeficientes YF re-estimados por el
autor sobre datos 2024 ('Resultados de regresión 2024.xlsx', pestaña Sheet2).

Decisiones del autor:
  - Se usan yf_SC / yf_VSC / yf_PitIn / yf_PitOut de Sheet2 (pit-in/out YF
    ajustados; sin NaNs, no requiere imputación).
  - SC_firstlap / VSC_firstlap se OMITEN (coef 0.0; los *_lastlap del Excel
    no existen en el modelo del simulador).
  - reg_intercept / reg_PitIn / reg_PitOut son referencia (la Eq.1 de
    bandera verde NO se toca).

Salida NUEVA (nada se borra): trained_models/simulation_yf2024_20260831/
= copia de simulation_pitlogit_20260807 con laptime_yf{_fast_v2,}.pkl
reemplazados en los 9 GPs del paper. El resto (Eq.1, logits de rivales,
logit de overtakes pit-aware) queda bit-idéntico.
"""

import pickle
import shutil

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_yf_models import YF_TERMS, TARGET_GPS, build_yf_predictor

XLSX = config.DATA_DIR / 'yf_regression_2024.xlsx'  # author's 2024 YF regression results (Sheet2)
SHEET = 'Sheet2'
SRC = config.MODELS_DIR / 'simulation_pitlogit_20260807'
DST = config.MODELS_DIR / 'simulation_yf2024_20260831'

COLMAP = {  # término del modelo -> columna del Excel (None = omitir, coef 0)
    'PitIn': 'yf_PitIn',
    'PitOut': 'yf_PitOut',
    'SC': 'yf_SC',
    'SC_firstlap': None,
    'VSC': 'yf_VSC',
    'VSC_firstlap': None,
}


def main():
    assert SRC.is_dir(), f'no existe el árbol base: {SRC}'
    if DST.exists():
        raise SystemExit(f'{DST} ya existe — no se sobreescribe nada. Bórralo tú si quieres regenerar.')

    df = pd.read_excel(XLSX, sheet_name=SHEET, header=0, index_col=0)
    missing = [gp for gp in TARGET_GPS if gp not in df.index]
    assert not missing, f'GPs sin fila en {SHEET}: {missing}'

    # El árbol pitlogit es un OVERLAY de symlinks sobre trained_models/simulation
    # (solo el logit de overtakes es archivo real). symlinks=True reproduce el
    # overlay tal cual; los archivos YF nuevos se escriben como archivos REALES
    # quitando antes el symlink (escribir a través de él corrompería el base).
    print(f'Copiando overlay {SRC.name} -> {DST.name} ...')
    shutil.copytree(SRC, DST, symlinks=True)

    audit = []
    for gp in TARGET_GPS:
        row = df.loc[gp]
        coefs = {t: (0.0 if col is None else float(row[col])) for t, col in COLMAP.items()}
        assert not any(np.isnan(v) for v in coefs.values()), f'{gp}: NaN en coeficientes'
        if coefs['SC'] <= coefs['VSC']:
            # Misma convención que train_yf_models: warning, no bloqueo (los
            # valores son del autor). Caso 2024: Singapore SC < VSC.
            print(f'  WARNING {gp}: SC ({coefs["SC"]:.2f}) <= VSC ({coefs["VSC"]:.2f})')

        with open(DST / gp / 'laptime_yf_fast_v2.pkl', 'rb') as f:
            old = pickle.load(f)
        old_coefs = dict(zip(YF_TERMS, [float(c) for c in old.coef_]))

        sm_res, fast_pred = build_yf_predictor(coefs)
        for fname, obj in [('laptime_yf_fast_v2.pkl', fast_pred), ('laptime_yf.pkl', sm_res)]:
            p = DST / gp / fname
            if p.is_symlink() or p.exists():
                os.remove(p)
            with open(p, 'wb') as f:
                pickle.dump(obj, f)

        for t in YF_TERMS:
            audit.append({'GP': gp, 'term': t,
                          'old_2021_23': round(old_coefs[t], 4),
                          'new_2024': round(coefs[t], 4)})
        print(f'  {gp:28s} SC {old_coefs["SC"]:7.2f}->{coefs["SC"]:7.2f}  '
              f'VSC {old_coefs["VSC"]:7.2f}->{coefs["VSC"]:7.2f}  '
              f'PitOut {old_coefs["PitOut"]:6.2f}->{coefs["PitOut"]:6.2f}')

    pd.DataFrame(audit).to_csv(DST / 'yf2024_coef_audit.csv', index=False)
    with open(DST / 'README.md', 'w') as f:
        f.write(
            'Árbol de simulación 2026-08-31: copia de simulation_pitlogit_20260807\n'
            'con los modelos YF reemplazados por la regresión 2024 del autor\n'
            f'({os.path.basename(XLSX)}, pestaña {SHEET}): coefs yf_SC/yf_VSC/\n'
            'yf_PitIn/yf_PitOut; SC_firstlap y VSC_firstlap omitidos (0.0);\n'
            'lastlap no existe en el modelo del simulador. Eq.1 verde, logits\n'
            'de rivales y logit de overtakes pit-aware: bit-idénticos al árbol\n'
            'base. Auditoría old-vs-new en yf2024_coef_audit.csv.\n'
        )

    # Verificación de recuperación exacta releyendo del disco
    print('\nVerificación (releído de disco):')
    for gp in TARGET_GPS:
        with open(DST / gp / 'laptime_yf_fast_v2.pkl', 'rb') as f:
            m = pickle.load(f)
        got = dict(zip(YF_TERMS, [float(c) for c in m.coef_]))
        row = df.loc[gp]
        want = {t: (0.0 if col is None else float(row[col])) for t, col in COLMAP.items()}
        ok = all(abs(got[t] - want[t]) < 1e-3 for t in YF_TERMS)
        # laptime de una vuelta SC pura y una VSC pura
        sc_lap = m.predict_single({'SC': 1.0, 'SC_firstlap': 1.0})
        vsc_lap = m.predict_single({'VSC': 1.0, 'VSC_firstlap': 1.0})
        print(f'  {gp:28s} {"OK " if ok else "MISMATCH"} SC={sc_lap:7.2f} VSC={vsc_lap:7.2f} '
              f'(firstlap sin efecto: {abs(sc_lap - m.predict_single({"SC": 1.0})) < 1e-6})')
        assert ok, f'{gp}: coeficientes releídos no coinciden'
    print(f'\nLISTO: {DST}')


if __name__ == '__main__':
    main()
