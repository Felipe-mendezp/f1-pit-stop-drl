# Sensibilidad a la convención de TyreLife (v8, 2026-10-20)

Pregunta: el simulador cuenta la vida del neumático desde 0 (vuelta de salida = 0, `f1_env_all_drivers.py:122`
y `:1799`; no aumenta bajo SC/VSC), mientras que en los datos de FastF1 la vuelta de salida de un neumático nuevo
tiene TyreLife = 1 (y > 1 si el juego es usado: 15 % de las largadas y 23 % de las salidas de pits en 2024).
¿Cambian las Figs. 6–7 si se suma 1 vuelta de vida?

## Qué se corrió

`scripts/tl_plus1_20261020.py` envuelve el entorno (monkeypatch, sin tocar `src/`) y suma `F1_TL_OFFSET` (def. 1)
vueltas de vida a: (i) el tiempo de vuelta en pista despejada (`coef_` de TyreLife del compuesto, pilotos no DNF) y
(ii) el logit de parada de los rivales (`_get_rival_pit_probs_fast`, `_get_rival_pit_probs_batch`,
`_should_rival_pit`). La observación del agente no cambia (la política está entrenada con base 0).
Mismo protocolo que `results/world2024_v8_20261019` (Fig. 6 con `--real-dnf`, Fig. 7 sin), PPO y RecurrentPPO
(QMIP automático), 500 carreras por GP, carreras pareadas por orden de fila.

    zsh results/tyrelife_plus1_v8_20261020/scripts/run_20261020.sh        # ~40 min, 8 GP en paralelo
    python results/tyrelife_plus1_v8_20261020/scripts/cmp_tl_20261020.py results/tyrelife_plus1_v8_20261020
    cd results/tyrelife_plus1_v8_20261020 && python scripts/sig67_20261020.py   # pares de métodos

Control: offset 0 en Dutch PPO (n = 10) reproduce exactamente las posiciones publicadas.
`scripts/tl_offset_det_20261020.py`: óptimo determinista (betas QMIP de simulation_v7) con vida inicial 0 vs 1 y
error de tiempo de carrera de las estrategias reales por neumáticos usados.
La corrida original se hizo desde `$CLAUDE_JOB_DIR/tmp/`; `run_20261020.sh` tiene las rutas ya cambiadas a esta carpeta
(no re-ejecutado desde aquí).

## Resultados (verificados)

Posición final media en los 8 GP, base v8 → TyreLife+1:

| | Fig. 6 | Fig. 7 |
|---|---|---|
| PPO | 8.120 → 8.020 | 8.657 → 8.713 |
| RPPO | 8.061 → 8.078 | 8.968 → 9.001 |
| QMIP | 8.404 → 8.392 | 8.882 → 8.946 |
| PPO − QMIP | −0.285 → −0.371 | −0.224 → −0.233 |
| RPPO − QMIP | −0.343 → −0.313 | +0.086 → +0.055 |
| PPO − RPPO | +0.058 → −0.058 (n.s. en ambos) | −0.310 → −0.288 |

- Fig. 6: único cambio significativo por GP, PPO US (−0.318, IC [−0.56, −0.07]).
- Fig. 7: aumentos pequeños significativos: Bahrain PPO +0.15 y QMIP +0.15, Hungary QMIP +0.16, Belgium PPO +0.18
  (IC en el borde de 0).
- Primera parada de PPO se mueve < 0.2 vueltas (Hungary 22.97 → 22.88, US 18.68 → 18.49, Bahrain 17.81 → 17.68).
- Determinista: el óptimo no cambia en 7 de 8 GP (Bahrain M19 S19 S19 → H21 S18 S18, pérdida 0.07 s, casi empate).
  Error de tiempo de carrera de las estrategias reales por neumáticos usados: Bahrain 6.35 s medio (9 pilotos,
  máx. 8.3), Hungary 4.21, Dutch 3.14, Singapore 1.74, Miami 1.72, Saudi 0.52.

Conclusión: las conclusiones del paper no cambian; no se modifica el simulador. El manuscrito declara el supuesto de
neumático nuevo como limitación (20-oct).
