# Final simulator physics used for every result of the paper. Source this file:
#     source reproduce/env_paper.sh
# (bash or zsh). Model/data locations use the repository defaults of config.py:
#     trained_models/simulation       simulator (lap-time, yellow-flag, overtake and rival models)
#     trained_models/rl_agents        45 multi-driver agents (+ one_driver/ for the single-driver benchmark)
#     trained_models/rl_agents_reward reward-function comparison agents (Supplementary Section P)

export OMP_NUM_THREADS=1

# Multi-driver environment (src/environment/f1_env_all_drivers.py)
export F1_DSQ_PENALTY=-1000        # disqualification penalty of the 'mix' reward (no second compound)
export F1_LASTLAP_PITOUT=1         # a pit stop on the final lap also pays the pit-out time loss
export F1_SC_BUNCHING=1            # field bunching behind the safety car ...
export F1_SC_BUNCH_DELTA=1.5       # ... with a 1.5 s target gap
export F1_FOLLOWING_GAP_SCALE=1.0  # stochastic following gap (0.4 s + Exp(1)) after overtake blocking

# Single-driver benchmark (src/environment/f1_env_one_driver.py, Fig. 5 / Table N.1)
export F1_ONE_DRIVER_DSQ_PENALTY=-1000

# Python interpreter (override with: PYTHON=/path/to/python bash reproduce/...)
export PYTHON="${PYTHON:-python}"

# Physics flags for the training entry points (they configure the env explicitly).
PHYS_FLAGS="--following-gap-scale 1.0 --sc-bunching --sc-bunch-delta 1.5"
export PHYS_FLAGS
