# Optimizing Pit Stop Strategies in Formula 1: Deep Reinforcement Learning vs. Mathematical Programming

Code, data, trained agents and final figures for the article by **Felipe Méndez and Charles Thraves**
(Department of Industrial Engineering, University of Chile), submitted to *Annals of Operations Research*.

> Archived on Zenodo — concept DOI [`10.5281/zenodo.20060266`](https://doi.org/10.5281/zenodo.20060266)
> (always resolves to the latest version). **v2.0.0** is the version of the ANOR manuscript.

The repository contains a race simulator for 2024 Formula 1 Grands Prix calibrated on 2021–2023 race
data (lap-time, yellow-flag, overtaking and rival-strategy models), five deep reinforcement learning
agents (Double DQN, A2C, TRPO, PPO, Recurrent PPO) trained on it, a quadratic mixed-integer programming
benchmark (QMIP, solved by model-predictive control), and the decision-support analysis of the paper.

---

## Contents

```
├── src/                     simulator and models
│   ├── environment/         f1_env_all_drivers.py (multi-driver race), f1_env_one_driver.py (single driver)
│   ├── models/              lap-time regressions, matrix completion, overtake and rival models
│   ├── optimization/        QMIP (qmip.py) and its lap-time parameterisation (utils_qmip.py)
│   ├── model_loader.py      loads the per-GP simulator models
│   ├── double_dqn.py, reg.py, utils.py
├── scripts/                 training, evaluation and plotting entry points (see the map below)
├── hyperparams/config.py    DRL hyper-parameters (Table F.2)
├── reproduce/               one-command reproduction scripts + MODELS_MANIFEST.csv
├── data/                    race data (FastF1) and fitted coefficient tables
├── figures/                 the final figures and tables of the paper (main/, supplementary/)
├── results/                 result tables behind the figures (CSV)
├── trained_models/          empty in git: download the Zenodo archive (trained_models/README.md)
├── config.py, compat.py     paths and unpickling compatibility shims
└── requirements.txt
```

Script file names keep the dates of the research log in which they were introduced
(e.g. `train_dqn_fix_20260831.py`); they were left unchanged so that the provenance recorded in
`reproduce/MODELS_MANIFEST.csv` points to the exact code that produced each model.

## Installation

Python **3.12** (tested with 3.12.12 on macOS, Apple M3 Pro, CPU only).

```bash
git clone https://github.com/Felipe-mendezp/f1-pit-stop-drl && cd f1-pit-stop-drl
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

* **Gurobi.** The QMIP benchmark uses `gurobipy`; the licence bundled with the pip package is
  size-limited but sufficient for all models of the paper. Scripts that import the QMIP module
  (Figs. 5–7 and the single-driver tools) require it.
* **Exact versions matter.** The simulator pickles were written with pandas 3.0 / numpy 2.4; keep the
  pinned versions. Every entry point calls `compat.setup_all()` before unpickling.

### Trained models and simulator (Zenodo)

Download `f1-pit-stop-drl-v2.0.0-models.tar.gz` (116 MB) from the Zenodo record of this version and
extract it at the repository root:

```bash
tar -xzf f1-pit-stop-drl-v2.0.0-models.tar.gz      # -> trained_models/{simulation,rl_agents,rl_agents_reward}
```

See [`trained_models/README.md`](trained_models/README.md) for its contents and SHA-256.

## Quick start: regenerate every figure of the paper

```bash
bash reproduce/figures.sh          # a few minutes; writes figures_regenerated/{main,supplementary}
```

It re-plots Figs. 6–7 and Section P from the cached 500-race results, re-solves the QMIP and re-runs
the agents for Fig. 5 / Table N.1, re-runs the agent rollouts of Fig. 9, rebuilds Figs. 8 and 10 from
the shipped decision-support results and Figs. 3, 4 and H.1 from the data. At the end it checks that the regenerated tables and CSVs are
byte-identical to the published ones.

## Reproducibility map

| Paper item | Published file (`figures/`) | Generator | Inputs |
|---|---|---|---|
| Fig. 3 | `main/fig03_facet_compounds_belgian.pdf` | `scripts/plot_fig3_facet_compounds.py --gp "Belgian Grand Prix"` | `data/session_2021–2023_V2.csv`, `data/completed_coefs_V2_filled.xlsx` |
| Fig. 4 | `main/fig04_driver_coef_mse.png` | `scripts/plot_driver_coef_mse.py` | `data/coef_drivers_2024.csv`, `data/practice_2024.csv` |
| Fig. 5, Table N.1 | `main/fig05_stints_single_driver.*`, `supplementary/tableN1_optimality_gap.tex` | `scripts/one_driver_filter_20260925.py fig5` / `m1` | `trained_models/rl_agents/one_driver/`, `data/compound_filter_one_driver_5pct_20260925.json` |
| Fig. 6 + table | `main/fig06_qmip_vs_drl_stochastic.*`, `main/table_fig06_*.tex` | `scripts/evaluate_qmip_vs_drl.py --n-sims 500` (`--plot-only` from cache) | `trained_models/rl_agents/`, `results/fig06_stochastic/` |
| Fig. 7 + table | `main/fig07_real_2024_replay.*`, `main/table_fig07_*.tex` | `scripts/evaluate_all_gps_with_qmip.py` (`--plot-only` from cache) | `trained_models/rl_agents/`, `results/fig07_real_2024/` |
| Fig. 8 | `main/fig08_decision_support_singapore_lap18.*` | `scripts/plot_decision_support.py --tag singapore_hul_real_lap18` | `results/decision_support/` |
| Fig. 9 | `main/fig09_decision_stints_singapore_lap18.*` | `scripts/plot_decision_stints.py --lap 18 --algo PPO …` | `trained_models/rl_agents/Singapore_HUL/PPO`, `results/decision_support/` |
| Fig. 10 | `main/fig10_decision_evolution_singapore.*` | `scripts/plot_decision_evolution.py --name singapore_hul_real` | `results/decision_support/singapore_hul_real_lap{1..61}.*` |
| Fig. H.1 | `supplementary/laptimes_outliers_2022.*` | `scripts/plot_fig_h1_outlier_laptimes.py` | `data/session_*_V2.csv` |
| Figs. P.1–P.4 | `supplementary/sectionP_*` | `scripts/evaluate_reward_comparison{,_stochastic}.py`, `scripts/reward_comparison_stats.py [--stochastic]` | `trained_models/rl_agents_reward/`, `results/sectionP_reward_comparison/` |

The exact command lines (paths, environment variables, flags) are in `reproduce/figures.sh`.
Figures 8–10 use the PPO agent (`F1_DS_ALGO=PPO`) and the paper style (`F1_DS_PAPER_STYLE=1`).

Other supplementary values come directly from shipped artefacts: circuit data and 2024 safety-car laps
(Tables C.1, D.1) from `trained_models/simulation/<GP>/config.json`; yellow-flag coefficients (J.1) from
`data/yf_regression_2024.xlsx` and `trained_models/simulation/<GP>/laptime_yf_fast_v2.pkl`; overtake
logit (L.1) from `trained_models/simulation/<GP>/overtake_logit.pkl`; DRL hyper-parameters (F.2) from
`hyperparams/config.py` and `reproduce/MODELS_MANIFEST.csv`.

## Simulator configuration of the paper

All results use the following settings (set by `reproduce/env_paper.sh`; the defaults in the code
reproduce earlier, superseded configurations and are **not** those of the paper):

| Variable | Value | Meaning |
|---|---|---|
| `F1_DSQ_PENALTY` | `-1000` | penalty of the *mix* reward when no second compound is used (disqualification) |
| `F1_LASTLAP_PITOUT` | `1` | a pit stop on the final lap also pays the pit-out time loss |
| `F1_SC_BUNCHING`, `F1_SC_BUNCH_DELTA` | `1`, `1.5` | the field bunches up behind the safety car (1.5 s target gap) |
| `F1_FOLLOWING_GAP_SCALE` | `1.0` | stochastic following gap after a blocked overtake |
| `F1_ONE_DRIVER_DSQ_PENALTY` | `-1000` | single-driver benchmark (Fig. 5) |

Training entry points receive the same physics through `--following-gap-scale 1.0 --sc-bunching
--sc-bunch-delta 1.5`. The reward-comparison agents (Section P) keep the disqualification penalty of
their own reward (−500 time, −200 position, −50 points), i.e. `F1_DSQ_PENALTY` is unset for them.

**Single-driver benchmark (Fig. 5).** Compounds used in less than 5 % of a circuit's 2021–2023 race laps
are excluded for both the agents and the QMIP, because their coefficients rest on very few stints
(Dutch C1, Emilia-Romagna C5, Hungarian C5, Miami C4, Saudi Arabian C4, United States C4;
`data/compound_filter_one_driver_5pct_20260925.json`). The filter is applied in memory by
`scripts/one_driver_filter_20260925.py`, which wraps the original single-driver tools. The
multi-driver environment uses all three compounds.

## Full pipeline (from scratch)

| Step | Command | Time (laptop CPU) |
|---|---|---|
| 1. Simulator (auxiliary models) | `bash reproduce/build_simulator.sh` (optionally `WITH_FASTF1=1` to re-download the race data) | ~hours |
| 2. Multi-driver agents (45) | `bash reproduce/train_multi_driver.sh` | 20–60 min per agent |
| 3. Single-driver agents (45) | `bash reproduce/train_single_driver.sh` | ~45 min per GP |
| 4. Reward-comparison agents (9) | `bash reproduce/train_reward_comparison.sh` | ~4 h |
| 5. Re-evaluation | `bash reproduce/evaluate.sh {fig7,fig6,ds,reward,all}` | 45 min – several hours each |
| 6. Figures | `bash reproduce/figures.sh` | minutes |

Training writes to new trees (`trained_models/retrained_*`), and evaluation runs on copies, so the
shipped models and caches are never overwritten. Use `--dry-run` to print the exact training commands.

### Provenance of the agents

`reproduce/MODELS_MANIFEST.csv` records, for every shipped agent, the script, arguments, effective seed
and simulator switches with which it was trained. In short:

* 5 M environment steps per agent; DQN with final exploration 0.02 and PPO with entropy coefficient
  0.02 (Table F.2).
* Seed 42. `train_rl_agents.py` offsets the seed by the algorithm's position in `--algorithms`, so the
  effective seeds are recorded per model. Recurrent PPO in Belgian (seed 7) and Saudi Arabian (seed 2024)
  are rescue seeds: the seed-42 agent was degenerate in the pit-timing screening (1–4 stops, first
  stop after lap 4; `scripts/diag_prefix_pits.py`).
* Closing the final-lap pit loophole (`F1_LASTLAP_PITOUT=1`) required retraining the 7 agents that
  exploited it (Saudi Arabian RPPO, PPO, TRPO, A2C; Miami TRPO; Dutch TRPO; Emilia-Romagna A2C). The
  other 38 were trained before the fix. They never pit on the final lap, and 37 of them evaluate
  bit-identically with and without it; Singapore TRPO differs only through rivals' final-lap stops.
* Single-driver: the configuration of Table F.1 (without competition). Bahrain, Belgian and Singapore
  (no compound filtered) come from the unfiltered run.

Training is seeded, but bit-identical retraining across machines and library versions is not guaranteed.
The figures are reproduced from the shipped agents.

## Citation

```bibtex
@article{mendez_thraves_pitstops,
  author  = {M{\'e}ndez, Felipe and Thraves, Charles},
  title   = {Optimizing Pit Stop Strategies in {Formula} 1: Deep Reinforcement Learning vs. Mathematical Programming},
  journal = {Annals of Operations Research},
  note    = {Under review},
  year    = {2026}
}
@software{mendez_thraves_code,
  author    = {M{\'e}ndez, Felipe and Thraves, Charles},
  title     = {Data and Code for: Optimizing Pit Stop Strategies in {Formula} 1},
  version   = {2.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20060266},
  url       = {https://doi.org/10.5281/zenodo.20060266}
}
```

## Version history

* **v2.0.0 (2026)** — version of the *Annals of Operations Research* manuscript. The simulator was audited
  and corrected, with changes to:
  * the yellow-flag coefficients (re-estimated on 2024 data);
  * the pit-aware overtake logit;
  * safety-car bunching and the stochastic following gap;
  * the disqualification penalty (−1000);
  * the final-lap pit-out loss;
  * the 5 % compound filter of the single-driver benchmark.

  All agents were retrained, and only the final code, figures, data and agents are included.
* v1.1.0 / v1.0.0 (2026) — earlier submission snapshots (DOI `10.5281/zenodo.20114287` for v1.1.0).

## License

MIT — see [`LICENSE`](LICENSE). Race data were obtained with [FastF1](https://github.com/theOehrly/Fast-F1)
from the public Formula 1 timing feeds.
