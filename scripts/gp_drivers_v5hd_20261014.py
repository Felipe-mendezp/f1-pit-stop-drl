"""Pilotos del agente multi-piloto v5hd (v5h con 4 pilotos de mitad de grilla) — 2026-10-14.

Decision (Charles, 2026-10-14): en los 4 GP donde el piloto del agente quedaba en promedio entre P2 y P4 en las
Figs. 6/7 v5h, se usa un piloto que partio en mitad de grilla, termino la carrera y no tuvo vueltas anomalas
(results/driver_choice_20261014/candidates_20261014.csv):
    Belgian    NOR -> GAS   (parte 12, llega 13 oficial; 14 en pista antes de la descalificacion de RUS)
    Hungarian  SAI -> TSU   (parte 10, llega 9)
    Dutch      RUS -> STR   (parte 8, llega 12)
    US         PIA -> MAG   (parte 8, llega 11)
El piloto que deja de ser agente pasa a ser rival con su estrategia real 2024 (data/session_2024_V2.csv, misma
codificacion que GP_CONFIGS: {0: compuesto de partida, vuelta de PitIn: compuesto siguiente}; la derivacion
reproduce 152 de 171 estrategias de rivales publicadas, las demas difieren por abandonos (-1) o por el config).
El nuevo agente se quita de los rivales. evaluate_all_gps.py no se modifica: GP_CONFIGS se cambia en memoria
(los demas scripts importan el mismo objeto) y luego se ejecuta el script pedido.

    python scripts/gp_drivers_v5hd_20261014.py --check                      # muestra los cambios y re-deriva desde los datos
    python scripts/gp_drivers_v5hd_20261014.py <script.py> <args del script>  # p. ej. scripts/train_ppo_fix_20260820.py ...
"""
import os, sys, runpy
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

# gp: (piloto nuevo, llegada oficial 2024 del nuevo, piloto anterior, estrategia real 2024 del anterior)
CHANGES = {
    'Belgian Grand Prix': ('Driver_GAS', 13, 'Driver_NOR', {0: 3, 15: 2, 29: 2}),
    'Hungarian Grand Prix': ('Driver_TSU', 9, 'Driver_SAI', {0: 4, 21: 3, 47: 4}),
    'Dutch Grand Prix': ('Driver_STR', 12, 'Driver_RUS', {0: 2, 25: 1, 54: 3}),
    'United States Grand Prix': ('Driver_MAG', 11, 'Driver_PIA', {0: 3, 32: 2}),
}


def install():
    import config  # noqa: F401
    import compat; compat.setup_all()
    import evaluate_all_gps as A
    for e in A.GP_CONFIGS:
        ch = CHANGES.get(e['gp_name'])
        if not ch or e['agent_driver'] == ch[0]:
            continue
        new, fin, old, strat = ch
        assert e['agent_driver'] == old, (e['gp_name'], e['agent_driver'])
        assert new in e['rival_strategies'] and old not in e['rival_strategies'], e['gp_name']
        short = e['gp_name'].replace(' Grand Prix', '')
        e['rival_strategies'] = {(old if k == new else k): (dict(strat) if k == new else v)
                                 for k, v in e['rival_strategies'].items()}
        e.update(agent_driver=new, model_dir=f"{short.replace(' ', '_')}_{new[7:]}",
                 display_name=f'{short} GP ({new[7:]})', real_finishing_pos=fin)
    return A


def check():
    import pandas as pd
    A = install()
    d = pd.read_csv(os.path.join(_ROOT, 'data', 'session_2024_V2.csv'))
    for e in A.GP_CONFIGS:
        if e['gp_name'] not in CHANGES:
            continue
        new, fin, old, strat = CHANGES[e['gp_name']]
        l = d[(d.GP == e['gp_name']) & (d.Driver == old[7:])].sort_values('LapNumber')
        s = {0: int(l.Compound_Detail.iloc[0][1:])}
        for r in l[l.PitIn == 1].itertuples():
            nxt = l[l.LapNumber == r.LapNumber + 1]
            if len(nxt):
                s[int(r.LapNumber)] = int(nxt.Compound_Detail.iloc[0][1:])
        print(f"{e['display_name']:22s} model_dir={e['model_dir']:16s} llegada={e['real_finishing_pos']:2d}  "
              f"rival {old}: {e['rival_strategies'][old]} (datos {s}, {'OK' if s == strat else 'DIFIERE'})  "
              f"rivales={len(e['rival_strategies'])}  {new} rival: {new in e['rival_strategies']}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    if sys.argv[1] == '--check':
        check(); sys.exit(0)
    install()
    target = sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(_ROOT, sys.argv[1])
    sys.argv = [target] + sys.argv[2:]
    runpy.run_path(target, run_name='__main__')
