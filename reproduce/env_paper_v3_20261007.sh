# 2026-10-07. Física del paper (env_paper.sh) + corrección del timing de SC/VSC,
# para el reentrenamiento multi-semilla. NO reproduce los resultados publicados
# (para eso: source reproduce/env_paper.sh). Source this file:
#     source reproduce/env_paper_v3_20261007.sh

_here="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"
source "$_here/env_paper.sh"

# El SC/VSC de la vuelta L se revela antes de que el agente decida la vuelta L
# (src/environment/f1_env_all_drivers.py, _reveal_next_lap_yf); QMIP recibe el
# descuento de parada bajo amarilla solo en vueltas donde la parada es válida.
export F1_SC_REVEAL=1

# 2026-10-09 (D5): bajo SC/VSC la QMIP usa la misma pérdida de parada que el simulador
# (PitIn + PitOut del modelo YF del árbol, p. ej. v4) en vez de time_stop/2
# (scripts/evaluate_qmip_vs_drl.py:build_qmip_params).
export F1_QMIP_YF_LOSS=model

# Vueltas SC/VSC 2024 corregidas contra data/session_2024_V2.csv (Fig. 7 y
# apoyo a la decisión; leídas en scripts/evaluate_all_gps.py:load_gp_config).
export F1_YF_LAPS_2024="$_here/../data/yf_laps_2024_20261007.json"
unset _here
