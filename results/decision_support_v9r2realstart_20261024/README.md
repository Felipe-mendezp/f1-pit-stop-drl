# Barridos A (largada real) y B (historia real), v9r2 — 2026-10-24

Pedido del autor: evaluar qué habría recomendado el DRL al piloto real (p. ej., ¿debió Norris parar bajo el SC de la
vuelta 7 en Saudi, largando con medios?). Mismo protocolo que `../decision_support_v9r2w24belief_20261023/` (PPO y RPPO,
escenarios (i) realyf y (ii) real, M = 500, proyecciones desde la vuelta n con la política y futuro simulado), salvo el
prefijo común hasta la vuelta n-1, que se fuerza con `scripts/ds_rollouts_realhist_20261024.py`:

- **A, largada real** (`../decision_support_v9r2realstart_20261024/`, `F1_DS_FORCE=start`): en la vuelta 0 el agente
  larga con el compuesto real 2024; después decide la política. Solo cambian PPO Saudi y Bahrain y RPPO Saudi, Bahrain
  y US (los demás pares tienen la misma largada y se copiaron del barrido v9r2).
- **B, historia real** (esta carpeta, `F1_DS_FORCE=hist`): largada y paradas reales hasta la vuelta n-1 (vuelta de
  entrada L -> compuesto de la tanda siguiente, como las estrategias reales de los rivales).

Lanzador `run_20261024.sh` (09:27–14:07, rc = 0 en las 42 corridas y en las figuras). Verificado [V]: en las 32 corridas de B la
historia de los meta coincide con la estrategia real de `data/session_2024_V2.csv`; en A, la largada.
La edad del juego de largada no se fuerza (el entorno toma el juego menos gastado del compuesto).

## Resultados [V] (leídos de los CSV)

**B** (`compare_real_20261024.py` -> `compare_laps_20261024.csv`, `compare_summary_20261024.csv`; dif. pareadas de puntos):
- Saudi vuelta 7 (Norris en medios, P6, no paró bajo el SC): blandos +3,2 a +3,6 puntos sobre quedarse y duros +1,5 a +2,1
  (significativos en los 4 casos). La política recomienda medios otra vez, por debajo de blandos y duros (estado poco
  visitado en el entrenamiento: el agente largaba siempre con blandos). **Matiz del 25-oct (ver abajo): la comparación
  no es limpia**, porque los brazos "no parar" y "medios" dependen de la máscara en el 99–100 % de los rollouts.
- Singapore vuelta 29 (Hülkenberg a duros): quedarse +0,7 a +1,6 puntos. US vuelta 17 (Magnussen a duros): no parar +1,5 a
  +2,2. Bahrain: parar a medios supera a los duros elegidos en las vueltas 12–15 y 32–41 (hasta +3,1).
- Recomendación = decisión real en 79 a 100 % de las vueltas (Belgian la más distinta: blandos antes de la v. 28).
- rec = mejor (posición): 73–100 %; más bajo en Bahrain (73–80 %) y Saudi RPPO (84–86 %).

**A** (`../decision_support_v9r2realstart_20261024/summary_20261024.csv`):
- Saudi: largando con medios, PPO y RPPO paran a medios en la v. 7 y de nuevo a medios al final: un solo compuesto
  (descalificados) en (i) y (ii). US RPPO igual (DSQ, coincidencia 45–60 %). Bahrain PPO (i)/(ii) y RPPO (i): 64–71 %.
- Lectura [H]: forzar una largada que el agente no elige lo lleva a estados poco entrenados, donde la política (argmax)
  es poco fiable. (Antes se decía aquí que los rollouts seguían siendo informativos en la v. 7 de Saudi; el control del
  25-oct lo matiza: ver abajo.)

## Control del 25-oct: fiabilidad de la política (correcciones de la máscara) [V]

`mask_check_20261025.py` -> `mask_check_20261025.csv` (por GP x algoritmo x escenario x barrido) y
`mask_check_laps_20261025.csv` (por vuelta y acción). En cada rollout la política juega el resto de la carrera con la
máscara de legalidad (`legal_mask`, `scripts/decision_support_rollouts.py`); `masked_hits > 0` = la política propuso al
menos una acción ilegal (volver a parar con el mismo compuesto sin cumplir la regla de dos compuestos, o no parar en la
última vuelta posible) y la máscara la reemplazó. Medias sobre los 32 casos x vueltas:

| Barrido | rec = mejor (puntos) | pérdida media | rollouts de la acción recomendada corregidos |
|---|---|---|---|
| v9r2 original | 94,5 % | 0,013 pts | 0 % |
| B, historia real | 93,4 % | 0,059 pts | 9,9 % |
| A, largada real | 90,2 % | 0,066 pts | 16,6 % |

- La máscara actúa solo en Saudi (B: 75 %, A: 99–100 % de los rollouts recomendados) y en US con RPPO (B: 31 %, A: 100 %);
  0 % en los otros GP y en todos los GP del barrido original.
- Saudi v. 7 en B, por acción (PPO (ii); RPPO y (i) igual): blandos 13,7 pts, 0 % corregidos; duros 12,1, 0 %;
  no parar 10,6, 99 %; medios 10,0, 99 %. Si no para, la política sigue con medios y casi nunca cambia sola de compuesto.
  Lectura [H]: el valor de "no parar" corresponde a una continuación mal jugada y rescatada por la máscara, así que la
  ventaja de parar (+3,2 a +3,6) probablemente está sobrestimada, en una magnitud que estos datos no permiten medir.
- US v. 17 en B: con PPO, 0 % corregidos en todas las acciones (conclusión "no parar" limpia); con RPPO, 100 % en
  "no parar" y "medios" (misma conclusión, pero apoyada en la máscara).
- Singapore v. 29 y Bahrain: 0 % en todas las acciones. En Bahrain el error es de elección (la política recomienda
  duros donde medios simulan mejor; pérdida de 0,1 a 0,4 pts por vuelta).
- Belgian, Dutch, Hungarian, Miami y Singapore en B: pérdida de 0,01 pts o menos, como el barrido original.
- Conclusión [H]: B es usable donde la largada real coincide con la del agente; Bahrain con un error moderado; Saudi y
  US RPPO no permiten conclusiones firmes. A sirve sobre todo para mostrar el límite de la política. Refuerza mantener en
  el paper el barrido original de Saudi (largada con blandos, sin correcciones de la máscara).
- Páginas A y B actualizadas el 25-oct (versión 2): columna "Máscara" en la tabla de los 8 GP y en la tabla por vuelta,
  nota en "Cómo leer esto" y tarjetas corregidas (`keys_20261024.json`; datos `mask` y `mask_rec` en `web_data_20261024.py`).

Páginas: B https://claude.ai/artifact/SJntt47FXbuASAkA8aZQbk , A https://claude.ai/artifact/YHExttBry2N4MyNxxZLkGi
(`web/decisiones_v9r2real{hist,start}_20261024.html`, datos `web_data_20261024.py`, tarjetas `keys_20261024.json`).
Rollouts (`PPO/`, `RecurrentPPO/`) y figuras por GP (`figs/`) no versionados (.gitignore).
