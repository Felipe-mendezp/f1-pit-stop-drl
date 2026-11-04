# Apoyo a la decision en v10g (Figs. 8-10) — CORRIDO EL 3-NOV (19:31 a 21:22, rc=0 en las 16 corridas)

Decision del autor (3-nov): correrlo **completo y solo con PPO**, pero mas adelante (no tenia el computador
conectado). Corrido esa misma noche; el texto del paper y las figuras NO se han actualizado (a la espera del autor).

## Por que

Los tres PDF de las Figs. 8-10 en la carpeta del paper son del **24-oct**, es decir del mundo **v9r2**. Todo el resto
del manuscrito ya esta en v10eGQ/v10g (Figs. 5, 6, 7, 11, Tabla N.1, tablas de las Ecs. 1-2 y textos), asi que estas
tres figuras son lo ultimo que queda atras. El parrafo de la seccion 7 (`sec:decision_support`) cuelga de ellas con
unos ocho numeros: 4.18 de posicion media, 12.95 puntos, -0.98 +- 0.56, -1.46 +- 0.70, -2.22 +- 0.57, el 71 % de
postergacion y 0.95-0.96 de probabilidad de top-10, mas la narrativa del blando con nueve vueltas y el safety car de
la vuelta 7.

## Como correrlo

```
caffeinate -i zsh results/decision_support_v10gw24belief_20261103/run_20261103.sh 2>&1 \
    | tee results/decision_support_v10gw24belief_20261103/sweep_20261103.log
```

- 16 corridas (8 GP x `--real-2024` y `--real-yf`), solo PPO, M = 500, 8 workers. **Unas 2 h** en el Mac: el barrido
  B del 2-nov, del mismo tamano, tomo 107 min, y el de v9r2 (32 corridas, dos algoritmos) tomo 3,3 h.
- Solo el caso de las Figs. 8-10 (~35 min): agregar `--gps Saudi_Arabian_NOR`.
- Entorno ya verificado el 3-nov con una pasada en seco (`--gps NO_SUCH_GP`): todos los asserts de
  `scripts/run_ds_v6_sweep_20261015.py --version v10g` pasan (ruido de la Ec. 2, adelantamiento v10f, juegos v10eGQ,
  sin `F1_DS_FORCE`). La entrada `v10g` de `VERSIONS` se agrego ese dia: simulador `simulation_v10eGQ_20261028`,
  agentes `retrained_v10g_20261030/selected` (los seleccionados el 31-oct), salida en este folder.

## Despues de correrlo

1. Comparar con v9r2 antes de tocar el `.tex`: que recomienda PPO en la vuelta 7 de Saudi, si el caso ilustrativo
   sigue siendo "vuelta 7 bajo safety car, blando con nueve vueltas", y como quedan los ocho numeros de arriba.
   El riesgo no es la CPU sino el texto: si el agente v10g larga con otro compuesto o para en otra vuelta, hay que
   reescribir el parrafo y los tres pies, o elegir otro punto de decision.
2. Figuras: `scripts/plot_decision_support.py`, `plot_decision_stints.py` y `plot_decision_evolution.py`
   (el `--figures` del barrido ya las deja en `figs/`). Copiar al paper como `decision_support_real_lap7.pdf`,
   `decision_stints_real_lap7.pdf` y `saudi_arabian_nor_real_evolution_points.pdf`, guardando las anteriores
   como `*_old_v9r2_20261024.pdf`.
3. Resumen del barrido: `summary_*.py` de los folders anteriores (coincidencia rec = mejor, regret medio).

## Resultado y resumen (3-nov)
- `summary_20261103.py`, `candidates_20261103.py` (vueltas de parada del agente en cada GP y escenario) y `web_data_20261103.py`
  generan `summary_20261103.csv`, `candidates_20261103.csv` y `web/decisiones_v10g_20261103.html`.
- Saudi v. 7 (ii): sigue medio, 12,13 puntos, 4,18 de posicion; duro -0,54 +- 0,33; no parar -2,23 +- 0,46; blando -2,05 +- 0,40;
  edad 7; top 10 = 1,00. El blando ya no es la peor opcion, y las figuras en `figs/` no tienen el estilo del manuscrito.
