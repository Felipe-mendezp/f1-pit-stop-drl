# Figs. 6–7 y Tabla O.1 v9r2 (multi-piloto, juegos reutilizables) — 2026-10-23

Pipeline: `reproduce/pipeline_v9r2_20261023.sh` (copia de `pipeline_v8_20261019.sh` con `F1_TYRE_SETS`,
`F1_TYRE_REUSE=1`, `F1_NO_SET_PENALTY=-1000`, `F1_QMIP_SYMMETRY=full`; detalle en su cabecera).

## Hecho (23-oct)

1. `zsh reproduce/pipeline_v9r2_20261023.sh qmip` (08:00–08:21, 8 GP en paralelo) -> `qmip_only/fig6_2024_realdnf/`,
   `qmip_only/fig7_2024/` (500 carreras por GP, solo la QMIP: el directorio de agentes `qmip_only/no_agents` esta vacio y
   los DRL se saltan). Entorno con juegos y reutilizacion verificado (`tyre_sets=True`, `_reuse=True`, juegos del agente
   cargados).
2. Prueba de la cadena completa con agentes minimos (3000 pasos, Hungarian, 2 semillas, en el tmp del job, no en
   `trained_models/`): select (`--calib v9r2`), fig6, fig7, qmipcheck y figs corren. **qmipcheck: filas QMIP identicas**
   con y sin agentes DRL en la misma corrida (fig6 y fig7), asi que `qmip_only/` vale como resultado definitivo de la
   QMIP. La etapa figs dibuja las 2x4 con estos datos; la Tabla O.1 necesita las columnas DRL (falla solo con la QMIP).

## QMIP v9r2 frente a v8 (posicion final media, 500 carreras; leido de los CSV) [V]

| GP | Fig. 6 v9r2 | Fig. 6 v8 | Fig. 7 v9r2 | Fig. 7 v8 |
|---|---|---|---|---|
| Bahrain ALO | 4.39 | 5.40 | 6.55 | 7.13 |
| Belgian GAS | 11.00 | 11.23 | 11.48 | 11.48 |
| Dutch STR | 8.58 | 8.25 | 10.25 | 9.86 |
| Hungarian TSU | 7.91 | 8.73 | 9.16 | 8.31 |
| Miami OCO | 12.52 | 12.33 | 12.50 | 12.46 |
| Saudi Arabian NOR | 5.17 | 5.19 | 5.55 | 5.69 |
| Singapore HUL | 7.07 | 7.14 | 7.86 | 7.64 |
| United States MAG | 8.45 | 8.82 | 9.44 | 8.80 |
| **Media** | **8.14** | **8.39** | **9.10** | **8.92** |

v8 = `results/world2024_v8_qmipsym_20261021/merged/`. 0 descalificaciones en v9r2. Paradas medias de la QMIP casi
iguales salvo United States en la Fig. 6 (1.29 v9r2 vs 1.90 v8). Las diferencias mezclan dos cambios (juegos usados
para el agente y para los rivales), sin descomponer [H sobre la causa].

## Pendiente (cuando termine f1v9r2)

    rsync -avz -e 'ssh -p 4603' cthraves@leftraru.nlhpc.cl:F1DLR/trained_models/rl_agents_v9r2_20261022 trained_models/
    zsh reproduce/pipeline_v9r2_20261023.sh all    # select, fig6, fig7, qmipcheck, summary, figs

`all` vuelve a correr la QMIP dentro de fig6/fig7; qmipcheck la compara con `qmip_only/`. Duracion: en v8, fig6 y fig7
con los 5 DRL tomaron ~25 min cada una (8 GP en paralelo); la validacion de semillas (400 modelos x 200 carreras) no
esta medida en los logs v8 (se reuso el cache).
Salidas: `fig6_2024_realdnf/figure_6_2x4.pdf`, `fig7_2024/figure_7_2x4.pdf`, `ci_table_v9r2_20261023.tex`.
