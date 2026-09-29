"""
Figure 3 (paper-exact): tire degradation curves with vs without outliers,
ported verbatim from the original generator Code/plot_compounds_facet.py
(which produced Code/Figures/<GP>_facet_compounds.pdf used in the paper).

Left panel  : Eq. (1) fitted on the RAW data (outliers included).
Right panel : final coefficients from data/completed_coefs_V2_filled.xlsx
              (IQR outlier removal + compatibility + matrix completion),
              i.e. exactly what the simulation environment uses.

Style: seaborn FacetGrid, no panel titles, Compound legend on the right,
colors Hard=black, Medium=#daa520, Soft=red, y = fixed + lap * variable.

Usage:
    python scripts/plot_fig3_facet_compounds.py                      # Belgian (paper Fig. 3)
    python scripts/plot_fig3_facet_compounds.py --gp "Miami Grand Prix"
    python scripts/plot_fig3_facet_compounds.py --all                # every complete GP
"""

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config
import compat; compat.setup_all()

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from models.regressions import all_laps, calibrar_regresion_gp

COLORS = {'Hard': 'black', 'Medium': '#daa520', 'Soft': 'red'}
EXCEL_PATH = str(config.DATA_DIR / 'completed_coefs_V2_filled.xlsx')
OUTPUT_DIR = str(config.ROOT_DIR / 'figs')

MAP_COMPOUND_2024 = {
    'Bahrain Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
    'Saudi Arabian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
    'Australian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'Japanese Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
    'Chinese Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
    'Miami Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
    'Emilia Romagna Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'Monaco Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'Canadian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'Spanish Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
    'Austrian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'British Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
    'Hungarian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'Belgian Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
    'Dutch Grand Prix': {'SOFT': 'C3', 'MEDIUM': 'C2', 'HARD': 'C1'},
    'Italian Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'Azerbaijan Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'Singapore Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'United States Grand Prix': {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2'},
    'Mexico City Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'São Paulo Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'Las Vegas Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
    'Abu Dhabi Grand Prix': {'SOFT': 'C5', 'MEDIUM': 'C4', 'HARD': 'C3'},
}


def get_gps_to_process():
    df_fixed = pd.read_excel(EXCEL_PATH, sheet_name='fixed')
    df_variable = pd.read_excel(EXCEL_PATH, sheet_name='variable')
    complete = df_fixed[df_fixed['C_2024_m'].str.strip() == ', ,']['Unnamed: 0'].tolist()
    selected = {}
    for gp in complete:
        if gp not in MAP_COMPOUND_2024:
            continue
        selected[gp] = [c for c in MAP_COMPOUND_2024[gp].values()]
    return selected, df_fixed, df_variable


def extract_with_outliers(laps_df, gp_name, compounds):
    data_gp = laps_df[(laps_df['GP'] == gp_name) & (laps_df['TrackStatus'] == 1.0)].copy()
    modelo, _ = calibrar_regresion_gp(data_gp)
    params = modelo.params
    fixed_c = [p.split('[T.')[1][:-1] for p in params.index
               if 'C(Compound_Detail' in p and '[T.' in p]
    ref = [c for c in compounds if c not in fixed_c][0]
    coefs_fixed = {ref: 0.0}
    for c in compounds:
        if c != ref:
            name = f"C(Compound_Detail, Treatment(reference='{ref}'))[T.{c}]"
            coefs_fixed[c] = params.get(name, np.nan)
    coefs_var = {c: params.get(f"TyreLife:C(Compound_Detail)[{c}]", np.nan)
                 for c in compounds}
    return coefs_fixed, coefs_var


def extract_without_outliers(df_fixed, df_variable, gp_name, compounds):
    rf = df_fixed[df_fixed['Unnamed: 0'] == gp_name].iloc[0]
    rv = df_variable[df_variable['Unnamed: 0'] == gp_name].iloc[0]
    return ({c: rf[c] for c in compounds}, {c: rv[c] for c in compounds})


def curves(coefs_fixed, coefs_var, labels_map, max_laps, scenario):
    rows = []
    for lap in range(1, max_laps + 1):
        for comp, label in labels_map.items():
            if comp in coefs_fixed and comp in coefs_var:
                rows.append({'Lap': lap,
                             'LapTime': coefs_fixed[comp] + lap * coefs_var[comp],
                             'Compound': label, 'Scenario': scenario})
    return pd.DataFrame(rows)


def plot_gp(gp_name, compounds, df_fixed, df_variable, laps_with, output_path):
    max_laps = int(laps_with[(laps_with['GP'] == gp_name) &
                             (laps_with['TrackStatus'] == 1.0)]['LapNumber'].max())
    labels_map = {c: h.capitalize() for h, c in MAP_COMPOUND_2024[gp_name].items()
                  if c in compounds}

    cf_w, cv_w = extract_with_outliers(laps_with, gp_name, compounds)
    cf_wo, cv_wo = extract_without_outliers(df_fixed, df_variable, gp_name, compounds)

    df = pd.concat([
        curves(cf_w, cv_w, labels_map, max_laps, 'With Outliers'),
        curves(cf_wo, cv_wo, labels_map, max_laps, 'Without Outliers'),
    ], ignore_index=True)

    g = sns.FacetGrid(df, col="Scenario", height=3, aspect=1, sharey=True)
    g.map_dataframe(sns.lineplot, x="Lap", y="LapTime", hue="Compound",
                    palette=COLORS, linewidth=2)
    g.set_axis_labels("Tire Usage [laps]", "Additional Lap Time [s]")
    g.set_titles("")
    for ax in g.axes.flatten():
        ax.set_xlim(0, max_laps)
        ax.grid(True, alpha=0.3)
    g.add_legend(title="Compound")
    sns.move_legend(g, loc="center left", bbox_to_anchor=(0.78, 0.5),
                    frameon=True, title="Compound")
    plt.savefig(output_path, format="pdf" if output_path.endswith(".pdf") else "png",
                bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gp', default='Belgian Grand Prix')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    selected, df_fixed, df_variable = get_gps_to_process()

    laps_with = all_laps(outliers=True)
    laps_with['GP'] = laps_with['GP'].replace('Styrian Grand Prix', 'Austrian Grand Prix')
    _emr = laps_with['GP'] == 'Emilia Romagna Grand Prix'
    laps_with.loc[_emr, 'Compound_Detail'] = laps_with.loc[_emr, 'Compound_Detail'].replace('C2', 'C5')

    targets = selected if args.all else {args.gp: selected.get(args.gp)}
    for gp, comps in targets.items():
        if comps is None:
            raise SystemExit(f"GP '{gp}' not complete in {EXCEL_PATH}")
        out = args.output or os.path.join(OUTPUT_DIR, f"{gp.replace(' ', '_')}_facet_compounds.pdf")
        print(f"Processing {gp} (compounds {comps})...")
        plot_gp(gp, comps, df_fixed, df_variable, laps_with, out)


if __name__ == '__main__':
    main()
