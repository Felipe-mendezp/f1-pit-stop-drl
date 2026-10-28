# Fig. 4 y Ec. 2 con la definición v10 (28-oct-2026)

Pregunta del autor: al sacar de la Ec. 1 los pasos por el pit lane sin parada y usar el factor 2021 de 0.740, ¿cambian los
coeficientes de piloto por carrera, la corrección de la Ec. 2 y, por ende, la Fig. 4?

- **Método.** Se re-corren, sin cambios, copias de `race_coefs_20261010` → `eq2_variants_20261010` →
  `fig4_mse_by_gp_20261014` y `eq2_variants_20261014` de `../driver_coef_study_20261010/`. La única edición es la ruta
  `ROOT` de `eq2_fallbacks_20261010.py`, porque la copia queda un nivel más abajo.
  - `control/`: factor legacy y sin parche. Reproduce lo publicado **exacto** (Fig. 4, tabla y coeficientes del simulador:
    diferencia 0).
  - `v10/`: `F1_TYRELIFE_2021=v10` y `patch_regressions` de `../sim_v10_20261027/common_v10_20261027.py`.
- **Comparación:** `compare_20261028.py` → `compare_fig4_20261028.csv`, `compare_sim_drivers_20261028.csv`.

## Resultados [V]

Sí cambian, pero muy poco:
- **Coeficientes de piloto por carrera** (1486): mediana del cambio 0. El 75 % no cambia. El máximo es 0.097 s/vuelta
  (Austria 2022, carrera con pasos por el pit lane).
- **Fig. 4:**
  - "Sin corrección": cambia ≤ 0.015 (Dutch 0.638 → 0.653).
  - "Con corrección": cambia ≤ 0.010 (Qatar 0.440 → 0.450).
  - Medias: 0.194 → 0.195 (con) y 0.635 → 0.638 (sin).
  - La corrección sigue mejorando en todos los GP salvo Qatar, así que el texto de §5.1.2 sigue valiendo.
- **`tab:reg_coef_drivers`** (S):
  - 2023: (i)–(iv) iguales a 3 decimales; "mismo circuito" 0.549 → 0.550.
  - 2024: 0.228/0.199/0.215/0.194/0.340/0.628 → 0.229/0.200/0.216/0.194/0.341/0.631.
  - El orden de las especificaciones y el texto no cambian.
- **Coeficientes de piloto del simulador** (Ec. 2 para 2024, 8 GP × 19 pilotos): cambio medio 0.0009, p95 0.0025 y
  máximo 0.0048 s/vuelta. El simulador v10 usa los de v6 (los publicados). La diferencia es despreciable frente al
  error de predicción de la Ec. 2 (MSE ≈ 0.19 (s/vuelta)²).
- **`tab:regression_pitstops`** (Ec. 2 de Belgian): no se re-estimó. Usa las mismas entradas, así que se esperan cambios
  del mismo orden [H].

## Decisión y aplicación (28-oct)

El autor eligió corregirlo:
- `scripts/build_simulation_v10_20261028.py` → `trained_models/simulation_v10_20261028`: solo cambian los dummies de
  piloto, y se verifica.
- Relanzamiento del multi-piloto.
- Paper: Fig. 4, `tab:reg_coef_drivers` y `tab:regression_pitstops` en `\vten`.
- `belgian_table_20261028.py` reproduce la tabla de Belgian (control exacto; antes no tenía script).
- `fallback_mse_20261028.py` reproduce los respaldos (control exacto). Las cifras del texto, 0.02 y 0.36, no cambian.
- Mono-piloto: sus predicciones no cambian (usa VER, la referencia).

## v10d (28-oct)

`run_20261028.py v10d` (con `F1_TYRELIFE_2021=v10`) usa los datos de `results/sim_v10d_20261028/data_fix_v10d_20261028.py`
(v10c más Bélgica y Mónaco 2022). Comparación en `compare_v10d_20261028.py` → `compare_v10d_20261028.txt` [V]:
- pilotos del simulador frente a v10c: media 0.0003, máximo 0.0026 s/vuelta;
- Fig. 4: solo cambia Bélgica "sin corrección", 0.341 → 0.327.

## v10eGQ (28-oct, noche): versión del paper

Cadena `results/sim_v10eGQ_20261028/` con el rastreo corregido (factor 2021 0,78). Corridas [V]:
`F1_TYRELIFE_2021=v10eGQ python results/driver_coef_v10_20261028/run_20261028.py v10eGQ` (salidas 12:21), luego
`python results/driver_coef_v10_20261028/v10eGQ/eq2_variants_20261014.py`, `fallback_mse_20261028.py` y
`belgian_table_20261028.py` (estos dos ahora incluyen v10eGQ).
- Fig. 4: con corrección media 0,190, sin corrección 0,625; la corrección mejora en todos los GP salvo Qatar (0,451 vs 0,379).
- `tab:reg_coef_drivers`: 2023 0.136/0.153/0.166/0.135/0.249/0.538; 2024 (276 pares) 0.221/0.195/0.210/0.191/0.322/0.618.
  La especificación porcentual sigue siendo la mejor en ambas temporadas.
- Respaldos: quitar X_Q cambia como máximo 0,0196; sin carreras previas, hasta 0,342 (texto "about 0.34", antes 0.36).
- `tab:regression_pitstops` (Belgian): 929/1022 obs., R² 0.578/0.516; todos los coeficientes ***.
- Aplicado al paper en `\vten{}` el 28-oct noche (`docs/CAMBIOS_PAPER_V10_20261027.md`, ítems 25–28). Commit `9508bda`.

