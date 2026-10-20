# Tablas de rivales y adelantamiento v7 (2026-10-20), ítems 30 y 32

Filas LaTeX de `tab:logit_pitin`, `tab:logit_compound_choice` y `tab:overtake_logit` del suplementario, generadas desde
los modelos que usa el simulador v7/v8 (`trained_models/simulation_v7_20261015`). Sin entrenar nada.

    cd Codigos/f1-pit-stop-drl-v2; source reproduce/env_paper_v3_20261007.sh
    F1_TYRELIFE_2021=v7 OUT=results/rival_overtake_tables_v7_20261020/k1_cluster.csv \
        python results/rival_overtake_tables_v7_20261020/k1_cluster_20261020.py
    python results/rival_overtake_tables_v7_20261020/make_tables_20261020.py

- LogitStop: 63,354 vueltas (2021-2023), pseudo R2 0.198, e.e. estándar. Fórmula del simulador:
  `PitIn ~ SC + VSC + used_two_compounds + EarlyLaps + LateLaps + TyreLife:C(Compound_Detail) + Position + GP`,
  EarlyLaps = vuelta <= 10, LateLaps = vuelta >= N - 10.
- CondLogitCompound (K1): 1,894 paradas (3 alternativas cada una), base C3, sin efectos de GP (no se identifican en un
  logit condicional), pseudo R2 0.219 (McFadden frente a probabilidades iguales). `k1_cluster.csv`: e.e. estándar
  (iguales a statsmodels) y agrupados por carrera (63 carreras); la tabla usa los agrupados.
- Adelantamiento: logit con intercepto y pendiente por GP, sin Emilia Romagna; todos los coeficientes con p < 0.01.
