# Ec. 1 sin las vueltas de paso por el pit lane (27-oct-2026)

`eq1_pitlane_check_20261027.py` repite el paso 1 de `results/sim_v7_20261015/laptime_v7_20261015.py` (por GP del paper,
vueltas verdes 2021–23, filtro A e IQR) quitando las vueltas con PitIn/PitOut sin cambio de juego. Salidas:
`counts_20261027.csv` y `coefs_20261027.csv`.

Resultado [V]: en las vueltas verdes de los 8 GP solo hay 3 pasos sin parada (Bahrain, Saudi y Hungría, una entrada y una
salida en cada uno), y el filtro IQR no los elimina. Casi ningún coeficiente cambia. El único movimiento visible es PitOut
en Saudi, de 18,67 a 18,81 (0,6 SE). FirstLap_pos en Hungría también aparece movido, pero es un coeficiente ≈ 0 con SE
≈ 0, así que es un artefacto numérico. Los coeficientes de neumático no cambian (≤ 0,1 SE). Decisión del autor: en v10 se
actualiza igual, por consistencia.
