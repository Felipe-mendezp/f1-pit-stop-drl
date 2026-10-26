# Optimizing Pit Stop Strategies in Formula 1: Deep Reinforcement Learning vs. Mathematical Programming

Code, data, trained agents and final figures for the article by **Felipe Méndez and Charles Thraves**
(Department of Industrial Engineering, University of Chile), submitted to *Annals of Operations Research*.

> Archived on Zenodo — concept DOI [`10.5281/zenodo.20060265`](https://doi.org/10.5281/zenodo.20060265)
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

## Revised version (v8, manuscript under revision)

> **Status (2026-10-20).** The figures and tables of the revised manuscript come from the **v8** agents and the
> **v7** simulator described here. The sections above still describe the shipped v2.0.0 archive; the Zenodo record
> will be updated with the v8 archive (`reproduce/pack_models_v8_20261020.py`). Nothing in the v2.0.0 trees
> (`trained_models/simulation`, `trained_models/rl_agents*`) was modified: every revised artefact lives in a new,
> dated tree.

### What changed

| Component | v2.0.0 | Revised (v8) | Code |
|---|---|---|---|
| Tire coefficients | matrix completion | joint hierarchical model (offsets and monotone slopes, GP random effect), 2021–2023, with 2021 tire life rescaled by 0.716 | `results/tyre_coef_study_20260930/`, `results/tyrelife_2021_20261015/`, `results/sim_v7_20261015/` |
| Driver coefficients (2024) | Eq. 2 (v2.0.0 version) | Eq. 2 in percent; races where the reference driver has < 15 clean laps are excluded | `results/driver_coef_study_20261010/` |
| Lap times under SC/VSC | OLS on 2024 | single 2021–2023 regression on 100 % SC/VSC laps (YF v4) | `scripts/fit_yf_model_v4_20261007.py` |
| QMIP pit loss under SC/VSC | half the green-flag loss | same loss as the simulator (`F1_QMIP_YF_LOSS=model`) | `src/optimization/qmip.py` |
| SC/VSC timing | revealed after the decision | revealed before the agent decides (`F1_SC_REVEAL=1`) | `src/environment/f1_env_all_drivers.py` |
| Miami SC/VSC probability | 0.10 / 0.15 | 0.50 / 0.50 (2022–2023 frequency) | `scripts/patch_miami_yf_v6_20261014.py` |
| Rival pit-stop logit | as published | re-estimated with the rescaled 2021 tire life | `results/sim_v7_20261015/rivals_v7_20261015.py` |
| Rival compound choice | binary logit per alternative, normalised | conditional logit (base C3, no GP effects) | `src/models/cond_logit_compound.py` |
| Rival starting compounds | uniform | per race p ~ Dirichlet(α, α, α), α = 0.4368 (ML on lap 1 of 19 races, 2021–2023) | `scripts/fit_rival_start_alpha_20261018.py`, `F1_RIVAL_START_ALPHA` |
| Rival retirements | never in training or Fig. 6 | random in training and seed selection (p = 0.132 per rival, lap from 2021–2023); real 2024 retirements in Figs. 6 and 7 | `scripts/rival_dnf_calibration_20261014.py` |
| Overtaking logit | 9 GP | re-estimated with the rescaled 2021 tire life, 8 GP | `results/sim_v7_20261015/overtake_v7_20261015.py` |
| Action 0 at lap 0 | random compound, −600 (single-driver −150) | medium compound, −1000 (both environments) | `F1_LAP0_ACTION0_MEDIUM`, `F1_LAP0_PENALTY`, `scripts/one_driver_v8_20261019.py` |
| Single-driver penalties | consecutive −20, laps 1–5 −15, last lap −30 | −200, −100, −150 (as multi-driver) | `scripts/one_driver_v8_20261019.py` |
| Agent's driver | — | Belgian GAS, Hungarian TSU, Dutch STR, US MAG (mid-grid starters); others unchanged | `scripts/gp_drivers_v5hd_20261014.py` |
| Circuits | 9 | 8 (Emilia Romagna dropped) | |
| Seeds | 42 (+ rescue seeds) | multi-driver: 10 per agent, selected on 200 validation races (1–4 stops, rules 5/7/8, ≤ 1 % disqualified); single-driver: 5 per agent, smallest gap to the QMIP | `scripts/select_seeds_v5_20261010.py --calib v8`, `results/onedriver_v8_20261019/select_seeds_20261019.py` |
| Evaluation world (Figs. 6–7) | same as training | races simulated with 2021–2024 tire coefficients; agents and QMIP keep the 2021–2023 ones (`F1_WORLD2024_TYRES=v7`) | `scripts/evaluate_world2024_20261010.py` |

### Trees

| Tree | Content | Built by |
|---|---|---|
| `trained_models/simulation_v7_20261015` | multi-driver simulator | `results/sim_v7_20261015/README.md` |
| `trained_models/simulation_onedriver_v7_20261015` | single-driver simulator (links into the above) | same |
| `trained_models/rl_agents_v8_20261018/seedXX` | 400 agents (8 GP × 5 methods × 10 seeds) | cluster, `reproduce/cluster/tasks_v8.tsv` (wrapper `scripts/rival_start_v8_20261018.py`) |
| `trained_models/rl_agents_v8_20261018/selected` | the 40 selected agents (Figs. 6–10) | `reproduce/pipeline_v8_20261019.sh select` |
| `trained_models/rl_agents_onedriver_v8_20261019/seedXX` | 200 single-driver agents (8 GP × 5 methods × 5 seeds); selection in `results/onedriver_v8_20261019/selection_20261019.tsv` (Fig. 5, Table N.1) | cluster, `reproduce/cluster/tasks_onedriver_v8.tsv` |

Provenance of the 80 agents used in the paper: `reproduce/MODELS_MANIFEST_v8_20261020.csv`
(`scripts/make_manifest_v8_20261020.py`, same columns as `MODELS_MANIFEST.csv`).

### Environment

Source `reproduce/env_paper_v3_20261007.sh` (it sources `env_paper.sh` and adds `F1_SC_REVEAL=1`,
`F1_QMIP_YF_LOSS=model` and `F1_YF_LAPS_2024`). The v8 pipelines then set `F1_SIMULATION_DIR`,
`F1_WORLD2024_TYRES=v7`, `F1_RIVAL_START_ALPHA`, `F1_LAP0_ACTION0_MEDIUM=1` and `F1_LAP0_PENALTY=-1000`
themselves; the training wrappers (`scripts/rival_start_v8_20261018.py`, `scripts/one_driver_v8_20261019.py`) also
set `F1_RIVAL_DNF` and the single-driver penalties.

### Pipeline and figures

```bash
# training (cluster): reproduce/cluster/tasks_v8.tsv (400) and tasks_onedriver_v8.tsv (200)
zsh reproduce/pipeline_v8_20261019.sh all                         # seed selection, Figs. 6 and 7 (500 races per GP), summary
zsh results/decision_support_v8w24belief_20261019/run_20261019.sh  # look-ahead rollouts, 8 GP x 2 scenarios, PPO and RPPO (~4.6 h)
# single-driver: steps 1-4 of results/onedriver_v8_20261019/README.md
zsh reproduce/figures_v8_20261020.sh                              # all paper figures and tables from the cached results (minutes)
```

`figures_v8_20261020.sh` collects the figures, with the file names used by the manuscript, in
`figures_regenerated_v8/`. Figs. 8–10 are the Saudi Arabian GP 2024 (NOR), decision lap 7, PPO, scenario with the
actual race replayed up to the decision lap.

### In progress: real 2024 tyre sets (v9r2)

> **Status (2026-10-23).** Single-driver agents trained, selected, and used in the manuscript (Fig. 5, Table N.1).
> Multi-driver agents (`rl_agents_v9r2_20261022`) training on the cluster; the QMIP part of Figs. 6–7 is already
> computed. Without the variables below everything is bit-identical to v8.

With `F1_TYRE_SETS` every driver starts each 2024 race with the dry tyre sets available to him at the start of the race,
inferred from the sets used in each session and the returns of the 2024 Sporting Regulations (art. 30.2; 6.73 sets per
driver on average, half of them new). With `F1_TYRE_REUSE=1` (v9r2) a stop fits the least-worn set of the chosen
compound and the tyre life of the new stint starts at its age; the set taken off goes back to the driver's sets with its
age (the in-lap included unless under SC/VSC) and can be fitted again, so a stop never lacks a set. Rivals fit their sets
in the same way; rivals with their real 2024 strategy (Fig. 7) take the real age of each stint. The agent observes
`tyre_sets` = age / 20 of the set it would fit by stopping now, per compound. The QMIP plans with the same sets
(`qmip._solve_sets`: future stints indexed by set, sets of a compound fitted from the least worn); it does not refit a
set it takes off, and an exact dynamic program gives the same optimum with and without refitting in the 8 GPs.

The first version, v9r (`F1_TYRE_SETS` alone: sets are consumed, the number of sets left is observed and asking for a
compound with no set left costs `F1_NO_SET_PENALTY` = −1000), is superseded: the set counts in the observation and the
penalty broke the learning of the single-driver agents (gaps of 20–50 s; diagnosis in
`results/onedriver_v9r2_20261021/README.md`). v9r agents do not work in v9r2 and vice versa (different observation).

| File | Content |
|---|---|
| `data/f1_2024_tire_usage.xlsx` | source: per-set (Lapchive) and per-stint (OpenF1) tyre usage, all 2024 sessions |
| `data/tyre_sets_2024_reg_20261021.csv` | sets available at the start of each 2024 race (art. 30.2 returns) |
| `data/tyre_race_stint_ages_2024_20261021.csv` | age of the set fitted at each real 2024 race stint |
| `results/tyre_usage_xlsx_check_20261021/`, `results/tyre_inventory_mixed_2024_20261021/` | checks against FastF1 and construction/validation of the inventory |
| `src/tyre_sets.py` | flag logic (`F1_TYRE_SETS`, `F1_TYRE_REUSE`) |
| `scripts/tyre_reuse_v9r2_20261022.py` + `scripts/tyre_sets_v9_20261021.py` | wrappers (v8 wrappers + the two variables); every v9r2 training, selection and evaluation runs through them |
| `scripts/check_tyre_reuse_20261022.py`, `scripts/check_tyre_sets_equivalence_20261021.py` | checks (reuse logic; bit-identical with the flags off) |
| `reproduce/cluster/tasks_v9r2.tsv`, `tasks_onedriver_v9r2.tsv` | 400 multi-driver and 200 single-driver training tasks (`trained_models/rl_agents_v9r2_20261022`, `rl_agents_onedriver_v9r2_20261022`) |
| `results/onedriver_v9r2_20261022/` | single-driver screening, seed selection, Fig. 5 and Table N.1 (steps in its README) |
| `reproduce/pipeline_v9r2_20261023.sh` | multi-driver: seed selection (`--calib v9r2`), Figs. 6–7, QMIP-only stage and control, Table O.1 |
| `results/world2024_v9r2_20261023/` | Figs. 6–7 and Table O.1 v9r2 (`qmip_only/` = QMIP runs without agents) |

```bash
# multi-driver, after copying trained_models/rl_agents_v9r2_20261022 from the cluster
zsh reproduce/pipeline_v9r2_20261023.sh all      # select, fig6, fig7, qmipcheck, summary, figs
```

### In progress: start-mix fine-tuning of PPO and RPPO (v9r2sm)

> **Status (2026-10-25).** Implemented; a 2-task test is running on the cluster. Not used in the manuscript.

The selected v9r2 PPO/RPPO agents always start on the same compound, and 14 of the 16 are disqualified in 87–100 % of
the races when forced to start on the compound they always pit to (`scripts/start_compound_probe_20261025.py`,
`results/startmix_proposal_20261025/`). With `F1_START_MIX=p`, `reset()` plays lap 0 itself with probability `p`
(uniform compound, valid action) and the episode starts at lap 1; without the variable everything is bit-identical.
`scripts/finetune_startmix_20261025.py` continues training a selected agent from its weights and VecNormalize stats
(p = 0.5, constant lr 1e-4 PPO / 1.5e-4 RPPO, 1.5 M steps, evaluation every 50 000 steps with the same mix,
`--vec dummy` = the 10 envs in one process, same results). 48 tasks (8 GP × 2 algorithms × 3 seeds) in
`reproduce/cluster/tasks_v9r2sm.tsv` → `trained_models/retrained_v9r2_startmix_20261025/seedXX/`.

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
  doi       = {10.5281/zenodo.20060265},
  url       = {https://doi.org/10.5281/zenodo.20060265}
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
* v1.1.0 / v1.0.0 (2026) — earlier submission snapshots (version DOIs `10.5281/zenodo.20060266` for v1.0.0 and `10.5281/zenodo.20114287` for v1.1.0).

## License

MIT — see [`LICENSE`](LICENSE). Race data were obtained with [FastF1](https://github.com/theOehrly/Fast-F1)
from the public Formula 1 timing feeds.
