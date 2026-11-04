# Juegos disponibles en los puntos de decision de la Fig. 11 (3-nov-2026)

Idea del autor: una tabla de apendice con los compuestos disponibles y su vida de uso en cada punto de decision
marcado en la Fig. 11 (`fig:realhist`). **Aplicada al manuscrito el 3-nov**: es la Tabla L.1 del suplementario
(apendice `app:tyre_sets`), con una frase que la presenta y un puntero desde la seccion 7 de `sn-article.tex`,
los tres en `\vten{}`.

- `run_20261103.sh` — mismo entorno que `../fig11_20261103.sh`; corre
  `scripts/tyre_sets_at_decisions_20261103.py` bajo `gp_configs_fix_20261028.py` + `ds_rollouts_realhist_20261024.py --run`
  (`F1_DS_FORCE=hist`, `--real-2024`, `prefix_seed` 0, el mismo prefijo real que dibuja la figura). 17 s en total.
- `<gp>.json` — por vuelta de decision: juego montado (compuesto y edad efectiva), pool restante por compuesto con las
  edades de cada juego, posicion y recomendacion del agente.
- `tab_tyre_sets_decisions.tex` — fragmento LaTeX armado por `scripts/table_tyre_sets_decisions_20261103.py`
  (una fila por fila de la Fig. 11: mismos grupos y vuelta representativa de mayor |diferencia|). Compila sin errores;
  lleva `\resizebox` porque a tamano normal se pasa 103 pt del ancho de `sn-jnl` (372 pt).

Con `F1_TYRE_REUSE=1` el juego que se saca vuelve al pool al parar, de modo que el compuesto montado siempre se puede
volver a calzar; por eso se informa aparte del pool. Detalle y discusion en `docs/TRASPASO_SESION_20261015.md` 15.13.
