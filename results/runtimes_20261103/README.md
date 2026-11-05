# Tiempos de computo para el paper (3-nov-2026)

- **Entrenamiento:** `training_times_20261103.py` une `sacct_20261103.txt` con los TSV de tareas -> `training_times_20261103.csv`. Horas de reloj en NLHPC (AMD EPYC 9224).
  Multi-piloto v10g, 4 CPU: mediana 1,48 (A2C), 1,50 (PPO), 1,65 (DQN), 1,67 (TRPO), 1,86 h (RPPO). Mono-piloto v10eGQ, 2 CPU: 0,16 (A2C), 0,25 (PPO), 0,28 (TRPO), 0,37 (DQN), 0,83 h (RPPO).
  PPO multi tiene 78 de 80 tareas porque dos corrieron en el piloto 18138628.
- **Decision y QMIP:** `solve_times_20261103.py` -> `solve_times_20261103.csv`. Un proceso, Apple M4. QMIP por vuelta 1,1-3,5 ms (media por GP), al inicio 6-21 ms, carrera completa 0,06-0,21 s. PPO 0,13 ms por decision.
- **Look-ahead (Saudi, vuelta 7, 43 vueltas por correr, un proceso, Apple M4, tiempo total del comando):** M = 500: 49,8 s; M = 100: 11,3 s; M = 50: 6,8 s.
- **Look-ahead en paralelo (4-nov):** `parallel_lookahead_20261104.py` -> `parallel_lookahead_20261104.csv`. Mismo caso, Apple M4 (4 nucleos de rendimiento y 6 de eficiencia), P procesos con semillas contiguas, tiempo de reloj incluida la carga del modelo en cada proceso. M = 500: 47,9 s (P = 1), 15,9 s (P = 4), 13,3 s (P = 8). M = 100: 11,4 s (P = 1), 5,6 s (P = 4), 5,8 s (P = 8). En los seis casos la union de los procesos es identica al barrido de las Figs. 8-10 (posicion, tiempo, puntos, paradas y vueltas de parada, simulacion por simulacion). Con M = 100, P = 8 no mejora a P = 4: pesa la carga del modelo.
