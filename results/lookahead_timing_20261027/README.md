# Tiempo de cómputo del look-ahead (27-oct-2026, análisis 1)

Una recomendación es una vuelta n con 4 acciones × M rollouts hasta el final de la carrera. Se midió para Saudi
(Norris), PPO post-entrenado, historia real (`F1_DS_FORCE=hist`, como la Fig. 11), con `--fast-env --legal-mask --real-2024
--sim-future --following-gap-scale 1.0`, entorno `results/real_strategy_check_20261027/env_20261027.sh` +
`F1_DS_FORCE=hist F1_DS_WORLD2024=1`. Se usó un solo proceso (`OMP_NUM_THREADS=1`) en un Apple M1 Pro y se midió con
`/usr/bin/time -p` [V]:

| Corrida | Tiempo total (s) |
|---|---|
| vuelta 7, M = 10 (≈ costo fijo: carga del modelo y prefijo) | 5,6 |
| vuelta 7, M = 100 | 16,5 |
| vuelta 7, M = 500 | 68,1 |
| vueltas 7 y 30, M = 500 | 97,8 (la vuelta 30 agrega ≈ 30 s) |

Desde la vuelta 7 quedan 43 vueltas, y cada rollout cuesta ≈ 31 ms. El costo escala con las vueltas restantes. Los rollouts
son independientes entre sí y se podrían repartir entre núcleos, pero el script actual corre en serie [H]. Los logs están
en esta carpeta. Comando:

    python scripts/gp_drivers_v5hd_20261014.py scripts/ds_rollouts_realhist_20261024.py --gp 'Saudi Arabian Grand Prix' \
      --driver Driver_NOR --model-dir Saudi_Arabian_NOR --algo PPO --M 500 --fast-env --legal-mask --real-2024 \
      --sim-future --following-gap-scale 1.0 --out-dir <tmp> --laps 7
