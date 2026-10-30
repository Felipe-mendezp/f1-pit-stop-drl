"""
2026-10-29 — RecurrentPPO con ent_coef 0.05 (en vez de 0.02); el resto igual a hyperparams/config.py (v1).
Se carga con `train_rl_agents.py --hyperparams rppoent05_20261029`, que solo cambia RecurrentPPO.
Motivo: en Miami (v10eGQf) PPO/RPPO colapsan la entropía antes de 1M pasos y se quedan parando en la vuelta 6;
con PPO, ent_coef 0.05 a 5M pasos sacó a una de dos semillas de ahí (results/miami_lap6_diag_20261029/README.md §8).
"""
from hyperparams.config import RECURRENT_PPO_CONFIG as _BASE

RECURRENT_PPO_CONFIG = dict(_BASE, ent_coef=0.05)
