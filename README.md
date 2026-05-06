# Optimizing Pit Stop Strategies in Formula 1: Deep Reinforcement Learning vs. Mathematical Programming

**Felipe Méndez**¹ and **Charles Thraves**¹,²,*

¹ Department of Industrial Engineering, University of Chile, Santiago, Chile
² Instituto Sistemas Complejos de Ingeniería (ISCI), Santiago, Chile
\* Corresponding author: `cthraves@dii.uchile.cl`

Contact: `felipe.mendez.p@ug.uchile.cl`

---

This repository contains the data and code accompanying the manuscript
*Optimizing Pit Stop Strategies in Formula 1: Deep Reinforcement Learning
vs. Mathematical Programming*, submitted to the **European Journal of
Operational Research (EJOR)**.

> Repository: `https://github.com/Felipe-mendezp/f1-pit-stop-drl` &nbsp;·&nbsp; Archived release: [`10.5281/zenodo.20060266`](https://doi.org/10.5281/zenodo.20060266)

## Abstract

Pit stop strategy optimization in Formula 1 is a challenging sequential
decision problem involving stochastic race events, tire degradation
dynamics, and strategic interactions among approximately 20 competing
drivers. At each lap, a driver must decide whether to pit and which tire
compound to use, with the optimal decision depending on uncertain future
disruptions such as Safety Car and Virtual Safety Car periods, as well as
the reactions of rivals. To address this problem, we develop a data-driven
simulation framework calibrated with historical race data from 2021 to
2024. The framework generates realistic lap times and captures key race
dynamics, including tire degradation, overtaking behavior, yellow flag
events, and rivals' pit-stop decisions. We train agents using five DRL
algorithms and compare their performance against a Quadratic Mixed-Integer
Programming (QMIP) formulation that provides optimal solutions in
deterministic single-driver settings. In simplified settings without
uncertainty or competition, DRL methods achieve near-optimal performance
relative to the QMIP benchmark. In more realistic scenarios with a full
grid of competitors and stochastic race events, Recurrent Proximal Policy
Optimization (Recurrent PPO) consistently achieves the best performance
among DRL methods, while the QMIP remains competitive despite its
simplified structure. When evaluated against actual 2024 race outcomes,
the Recurrent PPO-based policy outperforms the observed finishing
positions in most circuits. Finally, we analyze the impact of alternative
reward formulations on policy performance. Reward functions incorporating
championship points or final position outperform lap-time-based rewards,
while hybrid formulations combining lap times and points provide the most
robust performance across circuits and evaluation metrics.

**Keywords:** OR in Sports, Deep Reinforcement Learning, Pit-Stop Strategy,
Optimization, Quadratic Mixed-Integer Programming.

## Repository Structure

```
f1-pit-stop-drl/
├── config.py                    # Centralized path configuration
├── compat.py                    # Numpy/pickle compatibility helpers
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── CITATION.cff                 # Citation metadata (GitHub)
│
├── data/                        # Race data (2021-2024 seasons)
│   ├── session_{year}_V2.csv    # Lap-by-lap race data
│   ├── practice_{year}.csv      # Free practice session data
│   ├── coef_drivers_{year}.csv  # Driver regression coefficients
│   └── coef_pred.csv            # Predicted driver coefficients
│
├── src/                         # Source code (library modules)
│   ├── reg.py                   # FastLinearPredictor (32x faster than statsmodels)
│   ├── utils.py                 # Shared utilities (lap state, caching)
│   ├── model_loader.py          # Model loading and management
│   ├── gp_data_2024.py          # 2024 GP constants and initial positions
│   │
│   ├── environment/             # Gymnasium environments
│   │   ├── f1_env_all_drivers.py    # Multi-driver env (20 drivers)
│   │   └── f1_env_one_driver.py     # Single-driver env (QMIP benchmark)
│   │
│   ├── models/                  # Statistical model training
│   │   ├── regressions.py       # Clear-track lap time regression (RegLTC)
│   │   ├── lap_time_yf.py       # Yellow flag lap time model (RegLTYF)
│   │   ├── matrix_completion.py # SoftImpute for missing tire coefficients
│   │   ├── logit_overtakes.py   # Overtake probability logit (LogitOvertake)
│   │   ├── logit_yf.py          # Yellow flag logit model
│   │   ├── rival_models.py      # Rival strategy models (LogitStop + CondLogitCompound)
│   │   └── overtake_models.py   # Overtake model training pipeline
│   │
│   ├── optimization/            # Exact optimization
│   │   ├── qmip.py              # QMIP Gurobi solver
│   │   └── utils_qmip.py        # Simplified lap time calculator for QMIP
│   │
│   └── visualization/           # Plotting utilities
│       ├── race_visualizer.py   # Race progression visualization
│       ├── stint_visualizer.py  # Stint visualization by reward type
│       └── rival_stints.py      # Rival strategy stint grid
│
├── scripts/                     # Executable pipeline scripts
│   ├── build_dataframes.py            # Build data from FastF1
│   ├── train_regressions.py           # Train RegLTC models
│   ├── train_yf_models.py             # Train RegLTYF models
│   ├── run_matrix_completion.py       # Complete missing tire coefficients
│   ├── train_overtake_models.py       # Train LogitOvertake
│   ├── train_rival_models.py          # Train rival decision models
│   ├── train_rl_agents.py             # Train DRL agents (5 algorithms)
│   ├── train_reward_comparison.py     # Train reward comparison agents
│   ├── train_one_driver.py            # Single-driver training (QMIP benchmark)
│   ├── evaluate_all_gps.py            # Evaluate vs. real 2024 data
│   ├── evaluate_qmip_vs_drl.py        # QMIP vs. DRL comparison (Fig. 5, Tab. M.1)
│   ├── evaluate_reward_comparison.py            # Reward comparison (deterministic)
│   ├── evaluate_reward_comparison_stochastic.py # Reward comparison (stochastic)
│   ├── reward_comparison_stats.py     # Mann-Whitney U tests (Figs. O.1, O.2)
│   ├── plot_outlier_removal.py        # Tire performance pre/post outlier removal (Fig. 3)
│   ├── plot_driver_coef_mse.py        # Driver coefficient MSE (Fig. 4)
│   └── plot_final_position_distribution.py  # Final position boxplots (Figs. 6, 7)
│
├── hyperparams/                 # RL hyperparameter configurations
│   └── config.py                # DQN, A2C, TRPO, PPO, RPPO configs
│
└── trained_models/              # Pre-trained models (Git LFS)
    ├── simulation/              # Statistical models per GP (~933 MB)
    ├── rl_agents/               # Trained RL agents (~705 MB)
    └── rl_agents_reward/        # Reward comparison models (~83 MB)
```

## Requirements

- **Python** >= 3.10
- **GPU**: not required (trained on Apple M3 Pro); CPU is sufficient
- **Gurobi** (optional): required only for the QMIP benchmark (Section 4).
  Free [academic license](https://www.gurobi.com/academia/academic-program-and-licenses/)
  available.
- **Git LFS**: required to fetch the pre-trained models in `trained_models/`.

### Installation

```bash
git clone https://github.com/Felipe-mendezp/f1-pit-stop-drl
cd f1-pit-stop-drl
git lfs install && git lfs pull       # download trained models (~1.7 GB)
pip install -r requirements.txt

# Optional: for the QMIP benchmark
pip install gurobipy
```

## Quick Start: Reproduce the Main Results

The paper evaluates nine 2024 Grand Prix: Bahrain, Saudi Arabian, Miami,
Emilia Romagna, Hungarian, Belgian, Dutch, Singapore, and United States.
Five DRL algorithms are compared: Double DQN, A2C, TRPO, PPO, and Recurrent
PPO (RPPO).

```bash
# QMIP vs. DRL benchmark (Figure 5, Table M.1)
python scripts/evaluate_qmip_vs_drl.py

# Final-position distributions on the full grid (Figures 6 & 7)
python scripts/plot_final_position_distribution.py --mode simulated --output figs/figure_6.png
python scripts/plot_final_position_distribution.py --mode real      --output figs/figure_7.png

# Reward comparison (Figures 8 & 9)
python scripts/evaluate_reward_comparison.py
python scripts/evaluate_reward_comparison_stochastic.py

# Mann-Whitney U statistical tests (Figures O.1 & O.2)
python scripts/reward_comparison_stats.py
```

## Full Reproduction Pipeline

To reproduce results from scratch (data collection through evaluation):

### Step 1: Build DataFrames from FastF1

```bash
python scripts/build_dataframes.py
```
Downloads and processes lap data from the FastF1 API for the 2021–2024
seasons. Outputs `session_{year}_V2.csv` to `data/`.

### Step 2: Train Lap Time Regressions (RegLTC)

```bash
python scripts/train_regressions.py
```
Fits the clear-track lap time regression model (Eq. 1) with iterative
outlier removal (Algorithm 1) and driver coefficient updating (Eq. 2).

### Step 3: Train Yellow Flag Models (RegLTYF)

```bash
python scripts/train_yf_models.py
```
Fits lap time models under SC and VSC conditions (Eq. 3).

### Step 4: Matrix Completion

```bash
python scripts/run_matrix_completion.py
```
Applies the SoftImpute-based procedure (Algorithm B.1) to estimate missing
tire compound coefficients for 2024 circuits.

### Step 5: Train Overtake Models (LogitOvertake)

```bash
python scripts/train_overtake_models.py
```
Fits the logistic regression for overtake probability (Eq. 4).

### Step 6: Train Rival Models

```bash
python scripts/train_rival_models.py
```
Trains LogitStop (Eq. 5) and CondLogitCompound (Eq. 6) for rival pit-stop
decisions.

### Step 7: Train DRL Agents

```bash
# Train all 5 algorithms on a specific GP/driver
python scripts/train_rl_agents.py --gp "Bahrain Grand Prix" --driver Driver_ALO

# Train on every GP
python scripts/train_rl_agents.py --all
```
Trains DQN, A2C, TRPO, PPO, and Recurrent PPO using Stable-Baselines3
(5 M timesteps each). Hyperparameters are listed in `hyperparams/config.py`.

### Step 8: Evaluate

```bash
python scripts/evaluate_all_gps.py
python scripts/evaluate_qmip_vs_drl.py
python scripts/evaluate_reward_comparison.py
python scripts/evaluate_reward_comparison_stochastic.py
python scripts/reward_comparison_stats.py
```

## Reproducibility Map (Paper → Code)

### Methodology

| Paper Section | Equation / Algorithm | Code Location |
|---|---|---|
| 2.1 Lap time clear track (RegLTC) | Eq. 1 | `src/models/regressions.py` |
| 2.1.2 Outlier removal | Algorithm 1 | `src/models/regressions.py::filtrar_outliers_iqr()` |
| 2.1.3 Driver coefficient update | Eq. 2 | `src/models/regressions.py` |
| 2.2 Track with VSC/SC (RegLTYF) | Eq. 3 | `src/models/lap_time_yf.py` |
| 2.3 Overtaking (LogitOvertake) | Eq. 4 | `src/models/logit_overtakes.py` |
| 3.1 Action space | — | `src/environment/f1_env_all_drivers.py` |
| 3.2 State space | Table 2 | `src/environment/f1_env_all_drivers.py` |
| 3.3.1 Stop-or-continue (LogitStop) | Eq. 5 | `src/models/rival_models.py` |
| 3.3.2 Compound choice (CondLogitCompound) | Eq. 6 | `src/models/rival_models.py` |
| 3.4 Reward function | — | `src/environment/f1_env_all_drivers.py` |
| 3.5 State transition | — | `src/environment/f1_env_all_drivers.py` |
| 3.7 System evolution | Algorithm E.1 | `src/environment/f1_env_all_drivers.py::step()` |
| 4 QMIP formulation | Eq. 7 | `src/optimization/qmip.py` |
| 5 Numerical results | — | `scripts/evaluate_*.py`, `scripts/plot_*.py` |
| FastLinearPredictor (32× speedup) | — | `src/reg.py` |

### Appendices

| Appendix | Content | Code Location |
|---|---|---|
| A | Driver coefficient regressions (Eq. A.1) | `src/models/regressions.py` |
| B | Matrix completion for 2024 compounds (Algorithm B.1) | `src/models/matrix_completion.py`, `scripts/run_matrix_completion.py` |
| C | Selected compounds per GP | `src/gp_data_2024.py` |
| D | SC/VSC event probabilities | `src/environment/f1_env_all_drivers.py` (yellow-flag sampling) |
| E | Race evolution (Algorithm E.1) | `src/environment/f1_env_all_drivers.py::step()` |
| F | Hyperparameters (Tables F.1, F.2) | `hyperparams/config.py` |
| G | RegLTC coefficients (Belgian GP) | derivable from `trained_models/simulation/Belgian Grand Prix/laptime_clear.pkl` |
| H | Outlier laptimes (Belgian GP) | `scripts/plot_outlier_removal.py` (auxiliary) |
| I | RegLTYF coefficients (Belgian GP) | `trained_models/simulation/Belgian Grand Prix/laptime_yf.pkl` |
| J | Driver coefficient correction (Belgian GP) | `data/coef_drivers_2024.csv`, `data/coef_pred.csv` |
| K | LogitOvertake coefficients | `trained_models/simulation/<GP>/overtake_logit.pkl` |
| L | LogitStop & CondLogitCompound coefficients | `trained_models/simulation/<GP>/rival_pitstop_logit.pkl`, `rival_compound_logit.pkl` |
| M | DRL optimality gaps (single-driver) | `scripts/evaluate_qmip_vs_drl.py` |
| N | Drivers who did not finish | derivable from `data/session_2024_V2.csv` |
| O | Reward-function pairwise tests | `scripts/reward_comparison_stats.py` |

### Figures

| Figure | What it shows | Generating script |
|---|---|---|
| Fig. 3 | Tire performance with vs. without outlier removal (Belgian GP) | `scripts/plot_outlier_removal.py` |
| Fig. 4 | Driver-coefficient MSE with vs. without correction | `scripts/plot_driver_coef_mse.py` |
| Fig. 5 | DRL strategies vs. QMIP optimum (single-driver) | `scripts/evaluate_qmip_vs_drl.py` |
| Figs. 6 & 7 | Final position distributions (simulated / real-2024) | `scripts/plot_final_position_distribution.py` |
| Figs. 8 & 9 | Reward function comparison (simulated / real-2024) | `scripts/evaluate_reward_comparison*.py` |
| Figs. O.1 & O.2 | Mann–Whitney U pairwise heatmaps | `scripts/reward_comparison_stats.py` |

> **Note on hyperparameters.** The values in `hyperparams/config.py` are the
> source of truth for the trained agents shipped in `trained_models/`. If a
> minor discrepancy is observed against the typeset Tables F.1/F.2 in the
> manuscript, the values in this file are authoritative for reproduction.

## Data Sources

Race data is collected via the [FastF1](https://github.com/theOehrly/Fast-F1)
Python library, which provides telemetry, lap times, tire usage and event
data for the 2021–2024 F1 seasons.

## Funding

This work was supported by **CONICYT PIA/BASAL AFB220003** and
**ANID/CONICYT FONDECYT Iniciación 11241531**.

## Author Contributions (CRediT)

- **F.M.** — Data curation, Formal analysis, Methodology, Software, Writing – original draft.
- **C.T.** — Conceptualization, Funding acquisition, Methodology, Supervision, Writing – review & editing.

## Acknowledgments

The authors gratefully acknowledge financial support from CONICYT
PIA/BASAL AFB220003 and ANID/CONICYT FONDECYT Iniciación 11241531.

## Declaration of Generative AI Use

During the preparation of this work, the authors used Claude (Anthropic)
to assist with language editing and writing improvement. After using this
tool/service, the authors reviewed and edited the content as needed and
take full responsibility for the content of the published article.

## Data Availability

The data and code supporting the results of this study are available at
`https://github.com/Felipe-mendezp/f1-pit-stop-drl` and archived on Zenodo at [https://doi.org/10.5281/zenodo.20060266](https://doi.org/10.5281/zenodo.20060266).

## Citation

```bibtex
@article{mendez2025f1drl,
  title   = {Optimizing Pit Stop Strategies in Formula 1: Deep Reinforcement Learning vs. Mathematical Programming},
  author  = {M{\'e}ndez, Felipe and Thraves, Charles},
  journal = {European Journal of Operational Research},
  year    = {2025},
  note    = {Under review}
}

@misc{mendez2025f1drl_code,
  title     = {Data and Code for: Optimizing Pit Stop Strategies in Formula 1: Deep Reinforcement Learning vs. Mathematical Programming},
  author    = {M{\'e}ndez, Felipe and Thraves, Charles},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20060266},
  url       = {https://github.com/Felipe-mendezp/f1-pit-stop-drl}
}
```

## License

Released under the [MIT License](LICENSE).
