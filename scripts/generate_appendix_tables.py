"""
Regenerate the LaTeX appendix tables that depend on the retrained models:

  G.1 : Belgian clear-track regression, with vs without outliers
  I.1 : Belgian yellow-flag (RegLTYF) coefficients
  K.1 : per-GP overtake logit coefficients + training counts
  L.1 : LogitStop coefficients (global)
  L.2 : CondLogitCompound coefficients (global)
  M.1 : single-driver optimality gaps (seconds) vs the QMIP optimum

Tables A.3, J.1, N.1 depend only on static data files (coef_*.csv,
gp_data_2024) and are NOT affected by the retraining.

Output: figs/appendix/*.tex  (+ M.1 also as CSV)

Usage:
    python scripts/generate_appendix_tables.py [--skip-m1]
"""

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
import compat; compat.setup_all()

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import argparse
import pickle
import re

import numpy as np
import pandas as pd

# Overridable (2026-09-23) so a regenerated M.1 can land in a new folder
# without overwriting the tables of a previous run.
OUT_DIR = os.environ.get('F1_APPENDIX_OUT_DIR', str(config.ROOT_DIR / 'figs' / 'appendix'))

GPS = ['Bahrain Grand Prix', 'Belgian Grand Prix', 'Dutch Grand Prix',
       'Emilia Romagna Grand Prix', 'Hungarian Grand Prix', 'Miami Grand Prix',
       'Saudi Arabian Grand Prix', 'Singapore Grand Prix', 'United States Grand Prix']


def stars(p):
    return '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------- G.1
def table_g1():
    from models.regressions import all_laps, calibrar_regresion_gp, filtrar_outliers_iqr
    laps = all_laps(outliers=True)
    laps['GP'] = laps['GP'].replace('Styrian Grand Prix', 'Austrian Grand Prix')

    # "With outliers" = ALL laps of the GP, including yellow-flag/SC laps
    # (no TrackStatus filter): this is the raw baseline of the paper's
    # Table G.1 (1567 obs for Belgian, R^2 ~0.65).
    raw = laps[laps['GP'] == 'Belgian Grand Prix'].copy()
    m_with, _ = calibrar_regresion_gp(raw.reset_index(drop=True))
    m_wo, df_wo = filtrar_outliers_iqr(laps, 'Belgian Grand Prix')

    def pretty(name):
        name = re.sub(r"C\(Compound_Detail.*?\[T\.(C\d)\]", r"\1 (ref. C3)", name)
        name = re.sub(r"TyreLife:C\(Compound_Detail\)\[(C\d)\]", r"TireLife $\\cdot$ \1", name)
        name = re.sub(r"C\(Driver.*?\[T\.(\w+)\]", r"\1", name)
        name = re.sub(r"C\(Year.*?\[T\.(\d+)\]", r"Year \1", name)
        name = name.replace('LapNumber_2', 'LapNumber$^2$')
        return name

    rows = []
    names = list(dict.fromkeys(list(m_with.params.index) + list(m_wo.params.index)))
    for n in names:
        cells = []
        for m in (m_with, m_wo):
            if n in m.params.index:
                cells += [f"{m.params[n]:.4f}{stars(m.pvalues[n])}", f"({m.bse[n]:.3f})"]
            else:
                cells += ['--', '']
        rows.append((pretty(n), cells))

    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r' & \multicolumn{2}{c}{With outliers} & \multicolumn{2}{c}{Without outliers} \\',
             r'Variable & Coef. & Std. Err. & Coef. & Std. Err. \\', r'\midrule']
    for name, c in rows:
        lines.append(f"{name} & {c[0]} & {c[1]} & {c[2]} & {c[3]} \\\\")
    lines += [r'\midrule',
              f"Observations & \\multicolumn{{2}}{{c}}{{{int(m_with.nobs)}}} & \\multicolumn{{2}}{{c}}{{{int(m_wo.nobs)}}} \\\\",
              f"$R^2$ & \\multicolumn{{2}}{{c}}{{{m_with.rsquared:.3f}}} & \\multicolumn{{2}}{{c}}{{{m_wo.rsquared:.3f}}} \\\\",
              r'\bottomrule', r'\end{tabular}']
    open(os.path.join(OUT_DIR, 'table_G1.tex'), 'w').write('\n'.join(lines))
    print(f"G.1 done  (R2 {m_with.rsquared:.3f} -> {m_wo.rsquared:.3f}, "
          f"obs {int(m_with.nobs)} -> {int(m_wo.nobs)})")


# ---------------------------------------------------------------- I.1
def table_i1():
    yf = load_pickle('trained_models/simulation/Belgian Grand Prix/laptime_yf_fast_v2.pkl')
    d = dict(zip(yf.feature_names_, [float(c) for c in yf.coef_]))
    order = ['PitIn', 'PitOut', 'VSC_firstlap', 'SC_firstlap', 'VSC', 'SC']
    lines = [r'\begin{tabular}{lr}', r'\toprule', r'Variable & Coef. \\', r'\midrule']
    for k in order:
        lines.append(f"{k.replace('_', r'\_')} & {d[k]:.4f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(OUT_DIR, 'table_I1.tex'), 'w').write('\n'.join(lines))
    sc_minus_vsc = d['SC'] - d['VSC']
    pit_yf = d['PitIn'] + d['PitOut']
    print(f"I.1 done  (SC-VSC = {sc_minus_vsc:.1f}s; YF pit = {pit_yf:.1f}s) "
          f"-> ACTUALIZAR texto del apendice I")


# ---------------------------------------------------------------- K.1
def table_k1():
    # counts from the training log (Samples / Overtakes per GP)
    log = open('logs/aux_models_pipeline.log').read()
    counts = {}
    for gp in GPS:
        m = re.search(rf"Training overtake model for: {re.escape(gp)}.*?"
                      rf"Samples: (\d+) \| Overtakes: (\d+)", log, re.S)
        if m:
            counts[gp] = (int(m.group(1)), int(m.group(2)))

    lines = [r'\begin{tabular}{lrrrrr}', r'\toprule',
             r' & \multicolumn{2}{c}{Estimated Coefficients} & \multicolumn{2}{c}{Training data [laps]} & Overtake \\',
             r'Grand Prix & $\gamma_0$ & $\gamma_{\mathrm{PredGap}}$ & Overtakes & Total & Ratio \\',
             r'\midrule']
    for gp in GPS:
        mdl = load_pickle(f'trained_models/simulation/{gp}/overtake_logit.pkl')
        p, pv = mdl.params, mdl.pvalues
        g0n = 'const' if 'const' in p.index else 'Intercept'
        tot, ov = counts.get(gp, (int(mdl.nobs), int(round(mdl.model.endog.sum()))))
        short = gp.replace(' Grand Prix', '')
        lines.append(
            f"{short} & {p[g0n]:.4f}{stars(pv[g0n])} & "
            f"{p['delta_total']:.4f}{stars(pv['delta_total'])} & "
            f"{ov} & {tot} & {ov/tot:.3f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(OUT_DIR, 'table_K1.tex'), 'w').write('\n'.join(lines))
    print("K.1 done")


# ---------------------------------------------------------------- L.1 / L.2
def tables_l():
    stop = load_pickle('trained_models/simulation/global_logit_stop.pkl')

    def pretty(n):
        n = re.sub(r"C\(GP\)\[T\.(.*?) Grand Prix\]", r"\1 GP", n)
        n = re.sub(r"GP\[T\.(.*?) Grand Prix\]", r"\1 GP", n)
        n = re.sub(r"TyreLife:C\(Compound_Detail\)\[(C\d)\]", r"TyreLife $\\cdot$ \1", n)
        n = n.replace('used_two_compounds', 'Used two compounds')
        n = n.replace('EarlyLaps', 'Early laps (lap $\\leq$ 10)')
        n = n.replace('LateLaps', 'Late laps (lap $\\geq N-10$)')
        return n

    def logit_table(model, fname, extra_pretty=None):
        lines = [r'\begin{tabular}{lrr}', r'\toprule',
                 r'Variable & Coef. & Std. Err. \\', r'\midrule']
        for n in model.params.index:
            disp = (extra_pretty or pretty)(n)
            lines.append(f"{disp} & {model.params[n]:.4f}{stars(model.pvalues[n])} & "
                         f"({model.bse[n]:.3f}) \\\\")
        lines += [r'\midrule',
                  f"Observations & \\multicolumn{{2}}{{c}}{{{int(model.nobs)}}} \\\\",
                  f"Pseudo $R^2$ & \\multicolumn{{2}}{{c}}{{{model.prsquared:.3f}}} \\\\",
                  r'\bottomrule', r'\end{tabular}']
        open(os.path.join(OUT_DIR, fname), 'w').write('\n'.join(lines))

    logit_table(stop, 'table_L1.tex')
    print(f"L.1 done  (obs {int(stop.nobs)}, pseudo-R2 {stop.prsquared:.3f}) "
          f"-> filas EarlyLaps/LateLaps reemplazan 'Laps left' de la tabla vieja")

    comp = load_pickle('trained_models/simulation/global_logit_compound.pkl')

    def pretty2(n):
        n = re.sub(r"C\(alt\)\[(C\d)\]:LapsLeft", r"Alt: \1 $\\cdot$ LapsLeft", n)
        n = re.sub(r"C\(alt\)\[(C\d)\]", r"Alt: \1", n)
        n = n.replace('not_change_compound', 'Alt: Same \\& only one comp. used')
        return pretty(n)

    logit_table(comp, 'table_L2.tex', extra_pretty=pretty2)
    print(f"L.2 done  (obs {int(comp.nobs)}, pseudo-R2 {comp.prsquared:.3f})")


# ---------------------------------------------------------------- M.1
def table_m1():
    from plot_fig5_stints_one_driver import (
        SELECTED_GP, LAPS_DD, MODELS, MODEL_DISPLAY, ONE_DRIVER_DIR, rollout,
    )
    from environment.f1_env_one_driver import F1EnvOneDriverV2Optimized
    from evaluate_qmip_vs_drl import (
        build_qmip_params, qmip_choose_start_compound, qmip_choose_action, qmip_set_ages,
    )
    from model_loader import ModelLoader

    def rollout_time(env, act_fn, seed=42):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, done, _, _ = env.step(act_fn(obs, env))
        return float(env.total_time), bool(env.change_compound)

    loader = ModelLoader()
    gaps = {}
    for gp, compounds in SELECTED_GP.items():
        n_laps = LAPS_DD[gp]
        predictor = load_pickle(str(config.SIMULATION_DIR / f'{gp}_fast_predictor.pkl'))

        def make_env():
            return F1EnvOneDriverV2Optimized(
                n_laps=n_laps, available_compounds=compounds, gp=gp,
                driver='VER', model_laptime=predictor)

        qp = build_qmip_params(loader, gp)
        state = {'t_prev': [], 'started': False}

        def act_qmip(obs, env):
            if not state['started']:
                a, t0 = qmip_choose_start_compound(
                    qp['betas_dict'], qp['time_stop'], env.n_laps,
                    qp['qmip_to_compound'], qp['available_compounds'],
                    set_ages=qmip_set_ages(env, qp['qmip_to_compound']))
                state['t_prev'] = [t0]; state['started'] = True
                return a
            t_cur = qp['compound_to_qmip'].get(env.compound, 'Medium')
            a = qmip_choose_action(
                qp['betas_dict'], qp['time_stop'], env.laps_left, t_cur,
                env.tyre_life, state['t_prev'], env.n_laps, 0,
                qp['qmip_to_compound'], qp['available_compounds'],
                set_ages=qmip_set_ages(env, qp['qmip_to_compound']))
            if a > 0:
                nm = qp['compound_to_qmip'].get(qp['available_compounds'][a - 1], 'Medium')
                if nm not in state['t_prev']:
                    state['t_prev'].append(nm)
            return a

        t_opt, _ = rollout_time(make_env(), act_qmip)

        gaps[gp] = {}
        for name, cls in MODELS:
            path = os.path.join(ONE_DRIVER_DIR, gp[:-11], name, 'best_model.zip')
            if not os.path.exists(path):
                # Same (gap, changed_compound) tuple shape as the happy path;
                # a bare float here crashed the DataFrame build below.
                gaps[gp][name] = (np.nan, True)
                print(f"  {gp[:20]:21s} {MODEL_DISPLAY[name]:14s} SIN MODELO ({path})")
                continue
            model = cls.load(path)
            lstm = (name == 'RecurrentPPO')
            ms = {'lstm': None, 'start': np.ones((1,), dtype=bool)}

            def act_drl(obs, env, m=model, is_lstm=lstm, st=ms):
                if is_lstm:
                    a, st['lstm'] = m.predict(obs, state=st['lstm'],
                                              episode_start=st['start'], deterministic=True)
                    st['start'] = np.zeros((1,), dtype=bool)
                else:
                    a, _ = m.predict(obs, deterministic=True)
                return int(np.asarray(a).ravel()[0])

            t_drl, changed = rollout_time(make_env(), act_drl)
            gaps[gp][name] = (t_drl - t_opt, changed)
            flag = '' if changed else '  [DSQ: sin cambio de compuesto]'
            print(f"  {gp[:20]:21s} {MODEL_DISPLAY[name]:14s} gap = {t_drl - t_opt:8.3f} s{flag}")

    algo_names = [m[0] for m in MODELS]
    df = pd.DataFrame({g: {a: gaps[g][a][0] for a in algo_names} for g in gaps}).T[algo_names]
    dsq = pd.DataFrame({g: {a: not gaps[g][a][1] for a in algo_names} for g in gaps}).T[algo_names]
    df.columns = [MODEL_DISPLAY[c] for c in df.columns]
    dsq.columns = df.columns
    df.loc['Mean'] = df.mean()
    df.to_csv(os.path.join(OUT_DIR, 'table_M1.csv'))

    lines = [r'\begin{tabular}{l' + 'r' * len(df.columns) + '}', r'\toprule',
             'Grand Prix & ' + ' & '.join(df.columns) + r' \\', r'\midrule']
    for idx, row in df.iterrows():
        short = idx.replace(' Grand Prix', '')
        if idx == 'Mean':
            lines.append(r'\midrule')
            cells = [f"{v:.3f}" for v in row]
        else:
            cells = [f"{v:.3f}" + (r'$^\dagger$' if dsq.loc[idx, c] else '')
                     for c, v in row.items()]
        lines.append(short + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule',
              r'\multicolumn{' + str(len(df.columns) + 1) + r'}{l}{\footnotesize '
              r'$^\dagger$ Strategy uses a single compound (disqualifiable under '
              r'F1 rules); its time gap is not directly comparable.} \\',
              r'\end{tabular}']
    open(os.path.join(OUT_DIR, 'table_M1.tex'), 'w').write('\n'.join(lines))
    print("M.1 done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-m1', action='store_true')
    parser.add_argument('--only-m1', action='store_true',
                        help='Regenerate only the single-driver gap table '
                             '(M.1 in this repo = Table N.1 of the supplementary)')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.only_m1:
        table_i1()
        table_k1()
        tables_l()
        table_g1()
    if not args.skip_m1:
        table_m1()
    print(f"\nTablas escritas en {OUT_DIR}")
