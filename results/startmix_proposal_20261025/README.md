# Diagnóstico de largada forzada, agentes v9r2 — 2026-10-25

`scripts/start_compound_probe_20261025.py` sobre `trained_models/rl_agents_v9r2_20261022/selected` (PPO y RPPO, 8 GP),
200 carreras por modo, argmax, grilla real 2024, rivales y SC/VSC simulados, sin máscara de legalidad, misma semilla por
carrera en todos los modos. Envoltorio v9r2 + `env_paper_v3`, `F1_SIMULATION_DIR=trained_models/simulation_v7_20261015`.
Salida `start_probe_v9r2_20261025.csv` (log `.log`).

Resultado [V]: los 16 agentes largan siempre con el mismo compuesto; 14 de 16 quedan descalificados en el 87–100 % de las
carreras con al menos un compuesto de largada forzado (el compuesto al que paran en su propia estrategia). Excepciones:
Dutch PPO (máx. 26,5 %) y US PPO (0 %). Propuesta de post-entrenamiento: `docs/PROPUESTA_POSTENTRENAMIENTO_LARGADA_20261025.md`
(raíz del proyecto).

25-oct: post-entrenamiento aprobado e implementado (`F1_START_MIX`, `scripts/finetune_startmix_20261025.py`,
`reproduce/cluster/tasks_v9r2sm.tsv`, 48 tareas); prueba de 2 tareas (Saudi PPO y RPPO) corriendo en el cluster. Salida:
`trained_models/retrained_v9r2_startmix_20261025/`. Estado y pasos en la §7 de la propuesta. El mismo script de esta
carpeta es el control E1 sobre los agentes post-entrenados (`--agents trained_models/retrained_v9r2_startmix_20261025/seedXX`).
