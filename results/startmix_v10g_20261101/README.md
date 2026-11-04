# Post-entrenamiento con largada sorteada en v10g (v10gsm) — 1 y 2 de noviembre de 2026

Agentes del apéndice `app:realhist` (§7 del manuscrito y Fig. 11): PPO v10g post-entrenado con la largada sorteada,
para que la política sepa qué hacer con el compuesto de largada real de cada piloto en 2024 y no solo con el que
elegiría por su cuenta.

- **Mundo**: v10g = v10eGQf + ruido de la Ec. 2 (`F1_RACE_EFFECT_SD=0.28`), adelantamiento v10f
  (`F1_FOLLOWING_GAP_SCALE=0.26`), edad efectiva v10eGQ, VSC solo en la primera vuelta.
- **Receta**: `--start-mix 0.5`, `--lr 1e-4`, 1,5 M de pasos desde `trained_models/retrained_v10g_20261030/selected`;
  `ent_coef` 0,05 se hereda del `.zip`.
- **Modelos**: `trained_models/retrained_v10g_startmix_20261101/seed01…seed10`, 10 semillas × 8 GP = 80 modelos
  (trabajos del cluster: tsv del 1-nov para las semillas 1 a 3, 18181301 para las 4 a 10 de Miami,
  18209955 y 18209988 para las 4 a 10 de los otros siete GP).
- **Regla de selección** (la del paper, sin usar nunca la Fig. 7): `--ok-neutral --first-after 5 --late 7
  --min-gap 8 --max-dsq 0`, 200 carreras de validación.

## Scripts y salidas

| Archivo | Qué hace |
|---|---|
| `sel_startmix_v10gsm_20261101.sh` → `e2_selection_20261102/` | Selección con las 3 primeras semillas en los 8 GP (superada por la de 10 semillas). |
| `sel_miami10_20261102.sh` → `e2_miami10_20261102/` | Selección en Miami con sus 10 semillas; pasan 3, 5, 7, 8 y 10; elegida la 8 (`pos_media` 10,995). |
| `sel_all10_20261102.sh` → `e2_all10_20261102/` | **Selección final**: 10 semillas en los ocho GP. Elegidas: Bahrein s5, Saudi s4, Miami s8, Hungría s8, Bélgica s10, Países Bajos s6, Singapur s4, EE.UU. s1 (esta última reemplazada, ver abajo). |
| `e1_probe_all10_20261102.sh` → `e1_all10_20261102/e1_all10_20261102.csv` | E1 en los cuatro compuestos sobre las ocho elegidas con 10 semillas. |
| `e1_probe_us10_20261102.sh` → `e1_all10_20261102/e1_us_seeds10_20261102.csv` | E1 en las diez réplicas de EE.UU., para elegir la mejor que cumple con C3. |
| `e3_control_all10_20261102.sh` → `e3_world2024_all10_20261102/` | **Control E3 del conjunto final** (10 semillas y EE.UU. con la semilla 2). |
| `e1_probe_20261102.sh` → `e1_20261102/e1_real_start_20261102.csv` | Control E1 con la largada real de 2024 sobre las elegidas. Con 3 semillas **EE.UU. falla** (10,5 % de DSQ con C3, la largada de Magnussen). |
| `e1_probe_us_20261102.sh` → `e1_us_seeds_20261102.csv` | E1 en las tres réplicas de EE.UU.: las semillas 2 y 3 sí cumplen. |
| `e1_probe_miami_20261102.sh` → `e1_miami_seeds_20261102.csv` | E1 **amplio** (los cuatro compuestos) en las 5 semillas de Miami que pasan la regla. Solo la 7 da 0 % en los cuatro; la elegida (8) descalifica en el 100 % de las carreras con C4. |
| `e3_control_20261102.sh` → `e3_world2024_20261102/` | Control E3 de olvido: Figs. 6 y 7 con los agentes post-entrenados. |
| `e3_compare_20261102.py` → `e3_compare_20261102.csv` | Compara `results/world2024_v10g_20261031` (orig) con E3 (post). Una carrera sin cambio de compuesto cuenta como P20. |
| `alt_miami_s7_20261102/` + `e3_miami_s7_20261102.sh` → `e3_miami_s7_20261102/` | Mide las Figs. 6 y 7 de Miami con la semilla 7, en un árbol aparte para no tocar `selected/`. |

## Hallazgo principal: E1 amplio y E3 chocan en Miami [V]

| escenario | original | Miami s8 | Miami s7 | QMIP |
|---|---|---|---|---|
| Fig. 6, promedio 8 GP | 7,725 | 7,668 (−0,057) | 7,713 (−0,013) | 8,053 |
| Fig. 7, promedio 8 GP | 8,657 | 8,723 (+0,066) | 8,795 (**+0,138**) | 9,142 |

La semilla robusta a cualquier compuesto de largada (la 7) rompe el umbral de 0,1 posiciones del control de olvido en
la Fig. 7 y queda detrás de la QMIP en Miami en los dos escenarios. El criterio vigente sigue siendo el restringido a
la largada real (decisión del autor del 26-oct), que es la única que el análisis B y la Fig. 11 usan.

Detalle de la decisión y de los pendientes en `docs/TRASPASO_SESION_20261015.md` §15.

## Decisión del 2-nov (noche): E1 pasa a ser requisito de la regla de selección [V]

Con 10 semillas, siete de las ocho elegidas cumplen E1 con su largada real de 2024 y 0 % de incumplimiento
(Bahrein C3, Saudi C3, Miami C3, Hungría C4, Bélgica C3, Países Bajos C2, Singapur C4). **EE.UU. no**: la réplica que
la regla elige por validación, la semilla 1, descalifica en el 10,5 % de las carreras cuando se le impone el C3 de
Magnussen, igual que ocurría con 3 semillas.

Las diez réplicas de EE.UU. con C3 forzado (`e1_all10_20261102/e1_us_seeds10_20261102.csv`):

| semilla | DSQ con C3 | pos con C3 | pos_media val |
|---|---|---|---|
| 1 | 10,5 % | 8,945 | 8,020 |
| 2 | **0,0 %** | 7,505 | 8,060 |
| 3 | 0,5 % | 7,605 | 8,065 |
| 4 | 0,0 % | 7,695 | 8,135 |
| 5 | 0,0 % | 7,785 | 8,185 |
| 6 | 10,0 % | 8,725 | 8,020 |
| 7 | 1,5 % | 7,570 | 8,065 |
| 8 | 14,5 % | 9,485 | 8,270 |
| 9 | 2,5 % | 7,835 | 8,165 |
| 10 | 8,0 % | 8,615 | 8,230 |

La validación con largada propia no ordena esta robustez: las semillas 1 y 6 empatan en 8,020 y las dos fallan.
**Decisión del autor:** la regla de selección exige ahora, además de la validación, cumplir E1 con la largada real, y
entre las que cumplen se sigue eligiendo por `pos_media`. En EE.UU. eso lleva a la **semilla 2**, con un costo de
0,04 posiciones de validación; los otros siete GP no cambian. La réplica anterior quedó respaldada en
`e2_all10_20261102/us_seed01_reemplazada/` y el cambio está anotado en el `SELECTED_SEED.txt` de ese GP.

La sonda amplia además muestra que la fragilidad fuera de la largada real es general, no un defecto de Miami:
Hungría descalifica en el 8,5 % con C3, Bélgica en el 5 % con C4 y Bahrein en el 1,5 % con C1. Esa es la razón de
fondo para quedarse con el criterio de la largada real, que es la única que el análisis B y la Fig. 11 usan.

## Control E3 del conjunto final (2-nov, noche)

`e3_control_all10_20261102.sh` corre las Figs. 6 y 7 (500 simulaciones por GP) con las ocho réplicas elegidas
definitivas, y el resumen sale de `e3_compare_20261102.py` apuntado a `e3_world2024_all10_20261102/`
(copia adaptada en el scratchpad de la sesión; el CSV queda en `e3_compare_all10_20261102.csv`).

| Escenario | PPO original (v10g) | PPO post-entrenado | Δ | QMIP | DSQ máximo |
|---|---|---|---|---|---|
| Fig. 6 (abandonos reales) | 7,725 | 7,693 | **-0,032** | 8,053 | 0,0 % (original 0,2 % en Miami) |
| Fig. 7 (carreras reales) | 8,657 | 8,701 | **+0,045** | 9,142 | 0,0 % (original 0,0 %) |

Los dos escenarios quedan dentro del umbral de 0,1 posiciones, el post-entrenamiento no descalifica en ninguna de
las 500 simulaciones de ningún GP, y el promedio de PPO sigue por delante de la QMIP. La QMIP es idéntica entre las
dos corridas, como debe ser, lo que valida que el único cambio es la política. Con esto el apéndice `app:realhist`
quedó reescrito en `\vten{}` en `supplementary.tex` (diez semillas, requisito de E1 con la largada real, cifras de
este cuadro).
