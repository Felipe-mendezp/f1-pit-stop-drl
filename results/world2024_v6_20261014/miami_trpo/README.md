# TRPO en Miami (Fig. 7 v6): diagnóstico, 2026-10-15

Síntoma: TRPO queda 17.0 en la Fig. 7 (QMIP 12.4, A2C 10.5), 1.14 paradas, primera parada mediana en la vuelta 46 de 57.
Miami 2024 tuvo VSC en las vueltas 22-23 y SC en 28-32.

Verificado:
- No es la semilla elegida: 9 de 10 semillas de TRPO no pasan el filtro de selección; varias promedian 4-5 paradas o
  quedan descalificadas en hasta el 26 % de las carreras (`results/seed_selection_v5_20261010/v6/`). En v5h ya estaba mal
  (13.98, 2.62 paradas).
- No es el abandono de Sargeant (vuelta 29): sin abandono (`nodnf`) 17.8 y con el último gap visible (`dnfobs`) 17.0.
- Traza vuelta a vuelta (`trace_20261015.py`, 20 carreras): TRPO larga en blando (C4) y para en la vuelta 46-48 en el
  90 % de las carreras. Bajo el VSC su probabilidad de parar es 0.11 (para en 1 de 20) y en la primera vuelta del SC
  (única en que se permite parar) 0.01. Nunca elige el duro. A2C larga en medio y para siempre en la vuelta 28 bajo SC
  (9.2); la QMIP larga en blando y para en el VSC de la vuelta 22 (11.9).
- Control sin SC/VSC (`NOYF=1`): TRPO sigue parando en la 44-46 (13.3) frente a la QMIP en la 29 (11.2) y A2C en la
  34-35 (12.2).

Conclusión: dos efectos que se suman. (1) La política de TRPO en Miami alarga el primer stint en blando hasta ~46 vueltas
aun sin neutralizaciones (cuesta ~2 posiciones frente a la QMIP); (2) ignora las ventanas baratas de VSC/SC (otras ~4
posiciones con las neutralizaciones de 2024). Es un problema de aprendizaje de TRPO en este GP (degradación muy baja,
0.016-0.022 s/vuelta, así que el momento de la parada en verde casi no cambia la recompensa), no del entorno ni de la
evaluación. Hipótesis no verificada: el mismo patrón se repetirá en v7, donde los coeficientes de Miami casi no cambian.
