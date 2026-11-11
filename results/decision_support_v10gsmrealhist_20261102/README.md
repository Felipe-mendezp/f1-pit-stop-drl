# Barrido B (historia real 2024) con los agentes v10gsm — Fig. 11 del paper

2 y 3-nov-2026. Reemplaza a `../decision_support_v9r2smrealhist_20261026`, que era el que alimentaba la Fig. 11 y los
números de §7 del manuscrito.

- **Agentes**: `trained_models/retrained_v10g_startmix_20261101/selected` (post-entrenamiento con largada sorteada en
  v10g, 10 semillas por carrera, regla del paper más el control E1 con la largada real; ver
  `../startmix_v10g_20261101/README.md`).
- **Mundo**: v10g (ruido de la Ec. 2 con `F1_RACE_EFFECT_SD=0.28`, adelantamiento v10f, edad efectiva v10eGQ, VSC solo
  en la primera vuelta). Solo PPO: el manuscrito dejó RPPO fuera de §7 y del apéndice el 27-oct.
- **Corrida**: `run_20261102.sh` (`--version v10gsmhist`, M = 500, 16 corridas, 8 trabajadores). Terminó el 3-nov a
  las 00:38 con rc = 0 en las 16, 107 min. Log en `run_20261102.log`. La rama `PRUEBA=1` del script es la prueba chica
  obligatoria (un GP, M = 2).
- **Comparación vuelta a vuelta**: `compare_real_20261103.py` → `compare_laps_20261103.csv`,
  `compare_summary_20261103.csv`.
- **Fig. 11**: `fig11_20261103.sh` → `figs/realhist_stints_grid4x2.pdf`, copiada al paper como `realhist_stints.pdf`
  (la anterior quedó como `realhist_stints_old_v9r2_20261103.pdf`). Usa el argumento `--laps` que se le agregó a
  `scripts/plot_realhist_stints_20261026.py`.

## Resultados, escenario (ii) (banderas sorteadas, el que usa la figura) [V]

- La recomendación PPO coincide con la decisión real en **432 de 460 vueltas (93,9 %)**; el texto del paper pasa de
  95 % a 94 %.
- **Saudi v. 7**: blandos **+2,94 ± 0,42** puntos (antes +2,7). Sigue siendo el caso más grande con holgura.
- **Singapur v. 29**: no parar **+0,23 ± 0,19**, significativa.
- **EE.UU. v. 17**: no parar **+0,12 ± 0,05**, significativa pero pequeña. En v9r2 eran +2,0 puntos.
- **Bahrein v. 15**: **+0,05 ± 0,36**, no significativa. En v9r2 la parada real ganaba por 1,8 puntos.
- **Bahrein v. 38–40**: recomienda adelantar la segunda parada, hasta +0,8 significativo. **No se cita en §7** porque
  la figura agrupa las vueltas 35–40 y anota el valor de la v. 35, que es negativo.

Los dos últimos puntos son un cambio de fondo respecto del texto de v9r2 y están anotados como D42 en
`docs/PAPER_CODIGO_CORRESPONDENCIA.md`: falta el visto bueno del autor sobre qué casos ilustrar en §7.

## Fig. 11, revisión del 3-nov (mañana)

El autor observó que la figura sugería que la recomendación del DRL no mejoraba a la decisión del piloto y que había
paradas sin sentido (una en la v. 60 de 62 en Singapur). Las dos cosas eran defectos de la figura.

**Balance real del escenario (ii), PPO, sobre las 28 vueltas en que la recomendación difiere [V]:** +3,94 puntos a
favor del DRL; significativas, 7 a favor del DRL (+5,08) y 5 a favor del piloto (−1,16); neto positivo en 7 de los
8 GP (solo Bélgica −0,35). Comparable con v9r2 (+4,18) **filtrando `algo == 'PPO'`**: el CSV de aquel barrido trae
PPO y RecurrentPPO juntos. El 2 SE mediano bajó de 0,29 a 0,165, así que ahora pasan el corte diferencias de 0,2
puntos que antes no.

**Arreglo 1.** `group_rows` anotaba la primera vuelta del grupo; ahora anota la de mayor |diferencia| y dibuja su
continuación. Bahrein v. 35-40: de −0,07 (v. 35) a **+0,81 ± 0,18\*** (v. 39). Saudi v. 31-36: de −0,02 (v. 31) a
**−0,21 ± 0,15\*** (v. 34).

**Arreglo 2.** Las continuaciones se dibujan en el escenario base, sin banderas nuevas desde la vuelta de decisión
(`--no-future-yf`). Las paradas raras eran paradas bajo SC **sorteado** por `--sim-future`, invisible porque la
figura solo sombreaba las banderas reales (Singapur v. 29: SC sorteado en 58-61; Bahrein v. 39: 41-44; Bélgica
v. 27: 29-32 y 39-42). El sombreado es ahora por fila, hasta la vuelta de decisión más el bloque activo que la
contenga. **Los números no cambian**: salen del mismo barrido con `--sim-future`.

Regenerada con el mismo `fig11_20261103.sh` (log: `fig11_baseyf_20261103.log`, rc = 0). El caché
`figs/stints_cache_20261026.json` usa claves con sufijo `|baseyf`, de modo que las barras del modo anterior no se
reutilizan.

**Bahrein v. 39 no depende de las banderas [V]:** en los 500 rollouts la recomendación gana en 42,6 %, empata en
42,8 % y pierde en 14,6 %; separando por paradas, +0,85 en los 442 rollouts sin parada extra contra +0,52 en los 58
con parada extra.

## Fig. Q.1 del suplementario (3-nov)

`scripts/plot_realhist_dlap_20261103.py` dibuja la diferencia de puntos por vuelta (d = pts_rec - pts_real, +-2 SE)
alrededor de la segunda parada de Bahrein, leyendo `compare_laps_20261103.csv`: no corre rollouts nuevos. Salida en
`figs/realhist_dlap_bahrain_alo.{pdf,png}`, copiada al paper como `realhist_dlap_bahrain.pdf`. Hace visible que dentro
de un grupo de vueltas con la misma recomendacion el valor de seguirla cambia vuelta a vuelta (parar paga desde la
v. 38 y el maximo esta en la 39). Detalle en `docs/TRASPASO_SESION_20261015.md` 15.13.

## Fig. 11 nueva del cuerpo: Bahrein con la mejor de las cuatro acciones (11-nov)

Pedido del autor: la Fig. 11 de 8 GP pasa al suplementario (Fig. Q.1, `app:realhist`) y en el cuerpo va solo Bahrein,
una fila por vuelta de desacuerdo, con dos columnas: "DRL − actual" y "Best of the four − actual" ("Same as DRL" si la
mejor coincide con la recomendación). Script `scripts/plot_realhist_best_20261111.py` (importa el de la Fig. 11);
salida en `figs_best_20261111/`, copiada al paper como `realhist_best.pdf`.

**Verificación independiente** (`verif_best_20261111/`): las 28 vueltas de desacuerdo del escenario (ii), 4 acciones,
M = 500, semillas 100000+ (el barrido usa 1000+), mismo estado (`--prefix-seed 0`). `run_verif_20261111.sh PRUEBA` corre
con las semillas originales y reproduce el barrido bit a bit [V]. Corrida completa: 4 min con 8 procesos.
`analyze_verif_20261111.py` → `verif_best_20261111.csv`. Suma sobre las 28 vueltas, en puntos frente a la decisión real:

| | Barrido | Repetición |
|---|---|---|
| Seguir al DRL | +3,95 | +4,13 |
| Seguir la mejor de las cuatro (elegida en el barrido) | +5,82 | +4,92 |

La mejor pierde ≈ 0,9 puntos al medirla aparte (sesgo de selección); el DRL se replica. Las diferencias chicas por vuelta
no son estables: Bahrein v. 38 (+0,30\* → +0,06) y Saudi v. 33 y 36 no se replican; la mejor cambia de etiqueta en 10 de
28 vueltas, casi siempre entre acciones empatadas. Bahrein se sostiene: v. 35-36 mejor no parar (significativo en la
repetición), v. 39 +0,62\*, v. 40 medio +0,76\* (medio − duro +0,22 ± 0,10 en el barrido y +0,15 ± 0,09 en la repetición).

**Opción (a) del autor:** la mejor se elige con el barrido (`compare_laps_20261103.csv`) y las dos columnas de la figura
se miden con la repetición (`--vals verif_best_20261111`). Cifras del texto de §7: v. 39 +0,6; 14/10/4 (mejor = piloto /
DRL / otra, según el barrido); +4,9 frente a +4,1.
