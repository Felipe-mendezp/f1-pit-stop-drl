# Respaldo previo a la aceleración del simulador (2026-10-21)

Copias exactas (commit 46720a6) de los dos archivos que cambiaron:
- `reg.py` = `src/reg.py` (FastLinearPredictor que re-analizaba los nombres de las variables en cada predicción).
- `f1_env_all_drivers.py` = `src/environment/f1_env_all_drivers.py` (elección de compuesto de rivales con pandas).

La versión nueva da resultados idénticos bit a bit (trayectorias de 4.000 pasos en 4 GP y entrenamientos cortos de
PPO y DQN; ver `scripts/check_speedup_equivalence_20261021.py`) y es 1.7-2.0x más rápida en el simulador.
Para volver atrás: copiar estos archivos sobre los de `src/`. Dentro del código nuevo también quedan
`FastLinearPredictor._build_design_vector_legacy` y la ruta con pandas (se usa si el modelo de compuesto no es softmax).
