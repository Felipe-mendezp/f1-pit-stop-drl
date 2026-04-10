# Deep Reinforcement Learning for Pit Stop Strategy Optimization in Formula 1

**Felipe Mendez** and **Charles Thraves**  
Department of Industrial Engineering, University of Chile

## Abstract

In the pit-stop optimization problem in Formula 1, drivers must decide when to stop and which tire compound to use under stochastic race conditions and strategic interactions with rivals. We develop a data-driven approach based on Deep Reinforcement Learning (DRL). Using historical race data (2021--2024), we construct a simulation environment that generates realistic lap times while accounting for overtaking dynamics and rivals' decisions. We train agents using five DRL algorithms (DQN, A2C, TRPO, PPO, Recurrent PPO) and evaluate their performance under alternative reward formulations including lap-time-based, position-based, points-based, and hybrid objectives.

For simplified settings without uncertainty or competition, where optimal solutions can be computed via a quadratic mixed-integer programming (QMIP) model, we show that DRL methods attain near-optimal performance. In more realistic scenarios, Recurrent PPO (RPPO) consistently achieves the best performance among learning-based methods, while the QMIP remains competitive despite its simplified structure. We further show that hybrid reward formulations combining lap times and championship points provide a robust balance across race scenarios.

## Repository Structure

```
f1-pit-stop-drl/
├── config.py                    # Centralized path configuration
├── compat.py                    # Numpy/pickle compatibility helpers
├── requirements.txt             # Python dependencies
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
│   ├── build_dataframes.py      # Step 1: Build data from FastF1
│   ├── train_regressions.py     # Step 2: Train RegLTC models
│   ├── train_yf_models.py       # Step 3: Train RegLTYF models
│   ├── run_matrix_completion.py # Step 4: Complete missing tire coefficients
│   ├── train_overtake_models.py # Step 5: Train LogitOvertake
│   ├── train_rival_models.py    # Step 6: Train rival decision models
│   ├── train_rl_agents.py       # Step 7: Train DRL agents (5 algorithms)
│   ├── train_reward_comparison.py   # Step 8: Train reward comparison
│   ├── evaluate_all_gps.py          # Evaluate vs real 2024 data
│   ├── evaluate_qmip_vs_drl.py      # QMIP vs DRL comparison
│   ├── evaluate_reward_comparison.py # Reward comparison (deterministic)
│   ├── evaluate_reward_comparison_stochastic.py  # Reward comparison (stochastic)
│   ├── reward_comparison_stats.py    # Statistical tests (Mann-Whitney U)
│   └── train_one_driver.py          # Single-driver training (QMIP benchmark)
│
├── hyperparams/                 # RL hyperparameter configurations
│   └── config.py                # DQN, A2C, TRPO, PPO, RPPO configs
│
└── trained_models/              # Pre-trained models
    ├── simulation/              # Statistical models per GP (~933MB)
    ├── rl_agents/               # Trained RL agents (~705MB)
    └── rl_agents_reward/        # Reward comparison models (~83MB)
```

## Requirements

- **Python** >= 3.10
- **GPU**: Not required (trained on Apple M3 Pro); CPU is sufficient
- **Gurobi** (optional): Required only for QMIP benchmark (Section 4). Requires a [Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/) (free for academics).

### Installation

```bash
git clone https://github.com/<your-username>/f1-pit-stop-drl.git
cd f1-pit-stop-drl
pip install -r requirements.txt

# Optional: for QMIP benchmark
pip install gurobipy
```

If using Git LFS for trained models:
```bash
git lfs install
git lfs pull
```

## Quick Start: Evaluate Pre-trained Models

To reproduce the main results (Table 3 in the paper) using the pre-trained models:

```bash
# Evaluate all algorithms on all 2024 GPs
python scripts/evaluate_all_gps.py

# Compare QMIP vs DRL (requires Gurobi)
python scripts/evaluate_qmip_vs_drl.py

# Reward comparison
python scripts/evaluate_reward_comparison.py
python scripts/evaluate_reward_comparison_stochastic.py

# Statistical tests
python scripts/reward_comparison_stats.py
```

## Full Reproduction Pipeline

To reproduce the results from scratch (data collection through evaluation):

### Step 1: Build DataFrames from FastF1
```bash
python scripts/build_dataframes.py
```
Downloads and processes lap data from the FastF1 API for 2021-2024 seasons. Outputs `session_{year}_V2.csv` files to `data/`.

### Step 2: Train Lap Time Regressions (RegLTC)
```bash
python scripts/train_regressions.py
```
Fits the clear-track lap time regression model (Equation 1) with iterative outlier removal (Algorithm 1) and driver coefficient updating (Equation 2).

### Step 3: Train Yellow Flag Models (RegLTYF)
```bash
python scripts/train_yf_models.py
```
Fits lap time models under Safety Car and Virtual Safety Car conditions (Equation 3).

### Step 4: Matrix Completion
```bash
python scripts/run_matrix_completion.py
```
Applies SoftImpute to estimate missing tire compound coefficients for 2024 circuits (Appendix F).

### Step 5: Train Overtake Models (LogitOvertake)
```bash
python scripts/train_overtake_models.py
```
Fits logistic regression for overtake probability (Equation 4).

### Step 6: Train Rival Models
```bash
python scripts/train_rival_models.py
```
Trains LogitStop (Equation 5) and CondLogitCompound (Equation 6) for rival pit stop decisions.

### Step 7: Train DRL Agents
```bash
# Train all 5 algorithms on a specific GP and driver
python scripts/train_rl_agents.py --gp "Bahrain Grand Prix" --driver Driver_RUS

# Train on all GPs
python scripts/train_rl_agents.py --all
```
Trains DQN, A2C, TRPO, PPO, and Recurrent PPO agents using Stable-Baselines3 (5M timesteps each).

### Step 8: Evaluate
```bash
python scripts/evaluate_all_gps.py
python scripts/evaluate_qmip_vs_drl.py
python scripts/evaluate_reward_comparison.py
python scripts/evaluate_reward_comparison_stochastic.py
python scripts/reward_comparison_stats.py
```

## Paper-to-Code Mapping

| Paper Section | Equation/Algorithm | Code Location |
|---|---|---|
| 2.1 Lap Time Clear Track (RegLTC) | Eq. 1 | `src/models/regressions.py` |
| 2.1.2 Outlier Removal | Algorithm 1 | `src/models/regressions.py::filtrar_outliers_iqr()` |
| 2.1.3 Driver Coefficient Update | Eq. 2 | `src/models/regressions.py` |
| 2.2 Track with VSC/SC (RegLTYF) | Eq. 3 | `src/models/lap_time_yf.py` |
| 2.3 Overtaking (LogitOvertake) | Eq. 4 | `src/models/logit_overtakes.py` |
| 3.1 Action Space | -- | `src/environment/f1_env_all_drivers.py` |
| 3.2 State Space | Table 2 | `src/environment/f1_env_all_drivers.py` |
| 3.3.1 Stop or Continue (LogitStop) | Eq. 5 | `src/models/rival_models.py` |
| 3.3.2 Compound Selection (CondLogitCompound) | Eq. 6 | `src/models/rival_models.py` |
| 3.4 Reward | -- | `src/environment/f1_env_all_drivers.py` |
| 3.5 State Transition | -- | `src/environment/f1_env_all_drivers.py` |
| 3.7 System Evolution | Algorithm 2 | `src/environment/f1_env_all_drivers.py::step()` |
| 4 QMIP Formulation | -- | `src/optimization/qmip.py` |
| 5 Numerical Results | -- | `scripts/evaluate_*.py` |
| Appendix B: Hyperparameters | Tables B.1, B.2 | `hyperparams/config.py` |
| Appendix F: Matrix Completion | Algorithm F.1 | `src/models/matrix_completion.py` |
| FastLinearPredictor (32x speedup) | -- | `src/reg.py` |

## Data Sources

Race data is collected via the [FastF1](https://github.com/theOehrly/Fast-F1) Python library, which provides detailed telemetry, lap times, tire usage, and event data from the 2021--2024 F1 seasons.

## Citation

```bibtex
@article{mendez2025f1drl,
  title={Deep Reinforcement Learning for Pit Stop Strategy Optimization in Formula 1},
  author={M{\'e}ndez, Felipe and Thraves, Charles},
  year={2025},
  institution={Department of Industrial Engineering, University of Chile}
}
```

## License

MIT License
