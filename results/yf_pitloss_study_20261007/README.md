# Estudio exploratorio: pérdida de tiempo de pit bajo SC/VSC (2026-10-07)

Exploración, **sin cambios en el simulador ni en el manuscrito**. Datos `data/session_202{1..4}_V2.csv`,
filtro A (carreras con ≤10% de vueltas WET/INT; neumáticos de lluvia excluidos).
Ejecutar desde esta carpeta con `~/.venvs/f1-drl-v2/bin/python -W ignore <script>.py`, en orden (04→05 generan los .pkl que usan 06–08).
Cada script tiene su salida en el `.txt` del mismo nombre.

| Script | Qué hace |
|---|---|
| `yf_common_20261007.py` | carga, estado de pista por vuelta, índice de vuelta del evento, vuelta verde de referencia |
| `01_eventos` | conteo de eventos SC/VSC y paradas por vuelta del evento |
| `02_perdida_contaminada`, `03_agrupado_contaminado` | primer intento (incluye vuelta del despliegue, **contaminado**; descartado) |
| `04_componentes_limpios` | in/out-laps 100% bajo SC o VSC por piloto; exceso vs autos sin pit de la misma vuelta (3 referencias) |
| `05_referencia_estricta_cv` | referencia estricta (≥3 autos sin pit con vuelta pura), κ = exceso / componente verde del GP, CV deja-una-carrera-fuera |
| `06_comparacion_sim` | comparación con el simulador actual (`yf_regression_2024.xlsx`, Sheet2) y con 2021–23 |
| `07_vsc_y_sc_por_vuelta` | ritmo y pérdida SC por vuelta del evento; lista de eventos VSC |
| `08_unificado_sc_vsc` | modelo unificado SC temprano (vuelta 2 del evento) + VSC, bootstrap por carrera |
| `10_ritmo_y_gaps_sc` | ritmo SC/VSC sin pit (razón sobre vuelta verde, contracción por GP) y gap al de adelante bajo SC por vuelta del evento |
| `11_regresion_unica_yf` | regresión única (MixedLM, 2021–23): ritmo VSC / SC libre / SC en fila + pérdida de pit κ_in, κ_out, efecto GP en SC y efecto común por carrera-vuelta |
| `09_ajuste_2021_23_valida_2024` | mismo modelo ajustado solo con 2021–23 (verde incluido) y contraste con eventos limpios 2024 |

## Resultado (verificado)
- Ritmo: VSC 1.35× verde ≈ SC vueltas 2–3 1.38–1.39×.
- Pérdida SC según vuelta del evento (out-lap/verde): 0.72 (vuelta 2), 0.22 (vuelta 3, fila formándose), 0.89 (4+, relanzamiento).
- SC temprano vs VSC: sin diferencia significativa (Mann-Whitney p 0.13 in, 0.22 out; VSC n pequeño).
- **Unificado**: κ_in 0.38 (IC90 0.06–0.78), κ_out 0.74 (0.62–0.82). Pérdida YF media en los 8 GP del paper
  15.1 s (0.68× verde) vs simulador actual 13.5 s (0.62×); por GP −4.0 (Bélgica) a +4.6 s (Holanda, Singapur).

- **Solo 2021–23**: κ_in 0.15 (IC90 −0.09–0.73), κ_out 0.74 (0.46–0.81). Pérdida media 14.0 s (sim 13.5 s).
  Bélgica 10.8 (IC 5.9–14.7) vs sim 16.0. El contraste con 2024 no es informativo: 11 eventos limpios en 2 carreras
  (Miami, Qatar), y el simulador está ajustado con 2024 (dentro de muestra).

- **Ritmo sin pit (2021–23)**: VSC 1.35× verde (IQR 1.33–1.35, casi sin variación entre GPs); SC 1.39 (vuelta 2), 1.47 (3), 1.43 (4+).
  El sim usa valores implausibles en algunos GPs (VSC Bélgica 0.99×, EE.UU. 1.02×; SC Singapur 1.16×, Hungría 1.72×).
- **Gap bajo SC** (Interval_front, vueltas 100% SC, sin pit): mediana 1.77 s (vuelta 2), 1.63 (3), 1.68 (4), 1.37 (5), 1.21 (6);
  por carrera en vueltas ≥3: mediana 1.62 s (IQR 1.34–1.96). Verde: 2.45 s. El sim usa 1.5 s desde la 3ª vuelta.

- **Regresión única 2021–23** (1196 vueltas, 34 carreras): ρ_VSC 1.339 (±0.023), ρ_SC libre 1.439 (±0.026), ρ_SC fila 1.421 (±0.025),
  κ_in 0.38 (p 0.08), κ_out 0.63 (±0.08); sd del efecto GP en SC 0.084. Pérdida de pit en los GPs del paper 10.5–16.4 s.
  Aviso de convergencia de statsmodels (varianza en el borde).

## Adopción (2026-10-07)
Modelo aprobado por Charles y llevado al código oficial: `scripts/fit_yf_model_v4_20261007.py` (estimación,
reproduce el script 11), `scripts/build_simulation_yfv4_20261007.py` (árboles `simulation_yfv4_*`),
`scripts/test_yfv4_env_20261007.py`. Reentrenamiento v4 en el cluster: `docs/TRASPASO_SESION_20261007_v4.md`.

## Pendiente / limitaciones
- Muchas vueltas SC sin LapTime en los datos (posible selección).
- Efecto en el DRL no medido (evaluación "mundo equivocado").
