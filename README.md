# AI-Assisted Decision Support for Formula 1 Pit-Stop Strategy: Deep Reinforcement Learning and Mathematical Programming

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
├── reproduce/               figures_v10g_20261105.sh, pipelines, cluster tasks and archive scripts
├── data/                    race data (FastF1) and fitted coefficient tables
├── figures/                 the final figures and tables of the paper (main/, supplementary/)
├── results/                 scripts and result tables behind the figures, by dated study
├── trained_models/          empty in git: download the Zenodo archives (trained_models/README.md)
├── config.py, compat.py     paths and unpickling compatibility shims
└── requirements.txt
```

Script file names keep the dates of the research log in which they were introduced; they were left unchanged so that
the recorded provenance points to the exact code that produced each model. Comments in many of the dated scripts and
READMEs are in Spanish.

## Installation

Python **3.12** (tested with 3.12.12 on macOS, Apple M3 Pro, CPU only).

```bash
git clone https://github.com/Felipe-mendezp/f1-pit-stop-drl && cd f1-pit-stop-drl
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

* **Gurobi.** The QMIP benchmark uses `gurobipy`; the licence bundled with the pip package is
  size-limited but sufficient for all models of the paper. Scripts that import the QMIP module
  (Figs. 5-7 and the single-driver tools) require it.
* **Exact versions matter.** The simulator pickles were written with pandas 3.0 / numpy 2.4; keep the
  pinned versions. Every entry point calls `compat.setup_all()` before unpickling.

### Trained models and simulator (Zenodo)

The Zenodo record of this version holds two archives. Extract both at the repository root:

```bash
tar -xzf f1-pit-stop-drl-v2.0.0-models.tar.gz    # 106 MB: simulator and the 88 selected agents
tar -xzf f1-pit-stop-drl-v2.0.0-rollouts.tar.gz  # 51 MB: look-ahead rollouts behind Figs. 8-11
```

See [`trained_models/README.md`](trained_models/README.md) for their contents.

The scripts under `results/` call the interpreter at `~/.venvs/f1-drl-v2/bin/python`, so create the environment there:

```bash
python3.12 -m venv ~/.venvs/f1-drl-v2 && ~/.venvs/f1-drl-v2/bin/pip install -r requirements.txt
```

## Quick start: regenerate the figures of the paper

```bash
zsh reproduce/figures_v10g_20261105.sh    # about 1 min; collects the files in figures_regenerated_v2.0.0/
```

It re-solves the QMIP and re-runs the agents for Fig. 5 and Table N.1, and re-simulates one continuation per action for
Fig. 9. The other figures are redrawn from the cached results. Starting from a clean copy of the repository and the two
archives, the 10 regenerated PDFs matched the manuscript's pixel by pixel at 110 dpi, and Tables N.1 and O.1 matched the
versioned files byte by byte. Figs. 1, 2 and 4 are drawn in LaTeX. `reproduce/figures.sh` and `reproduce/evaluate.sh`
belong to earlier versions and need models that are not shipped.

## Reproducibility map

| Paper item | Manuscript file | Generator | Inputs |
|---|---|---|---|
| Fig. 3 | `facet_compounds.pdf` | `results/sim_v10eGQ_20261028/fig3_v10_20261027.py --laps 28` | `data/`, `results/sim_v10eGQ_20261028/` |
| Fig. 5, Table N.1 | `stints_alone.pdf` | `results/onedriver_v10eGQ_20261028/run_fig5_tableN1_20261028.sh` | `results/onedriver_v10eGQ_20261028/selected`, simulator |
| Figs. 6 and 7, Table P.1 | `qmip_vs_drl_stochastic_boxplot.pdf`, `all_gps_2024_qmip_boxplot.pdf` | `reproduce/pipeline_v10g_20261031.sh figs`, `results/world2024_v9r2_20261023/fig7_actual_20261027.py`; Table P.1 with significance marks: `results/world2024_v10g_20261031/ci_table_v10g_20261111.py` | `results/world2024_v10g_20261031/`, `results/overtake_noise_frac_20261030/l000_r28_k/` |
| Figs. 8, 9 and 10 | `decision_support_real_lap7.pdf`, `decision_stints_real_lap7.pdf`, `saudi_arabian_nor_real_evolution_points.pdf` | `results/decision_support_v10gw24belief_20261103/figs_paper_20261103.sh` | rollouts archive, `retrained_v10g_20261030/selected` |
| Fig. 11 | `realhist_best.pdf` | `scripts/plot_realhist_best_20261111.py --gp bahrain_alo --vals results/decision_support_v10gsmrealhist_20261102/verif_best_20261111 --no-title` (environment of `fig11_20261103.sh`) | `compare_laps_20261103.csv` (choice of the best action), `verif_best_20261111/PPO/` (independent rollouts, seeds 100000+, `run_verif_20261111.sh`) |
| Fig. Q.1 | `realhist_stints.pdf` | `results/decision_support_v10gsmrealhist_20261102/fig11_20261103.sh` | rollouts archive, `compare_laps_20261103.csv`, `figs/stints_cache_20261026.json` |
| Fig. H.1 | `laptimes_outliers_2022.pdf` | `results/sim_v10eGQ_20261028/outliers_us_v10_20261027.py` | `data/` |
| Table L.1 | body of the table | `tyre_sets_20261103/run_20261103.sh`, `scripts/table_tyre_sets_decisions_20261103.py` | rollouts archive, simulator |

## Simulator configuration of the paper

The defaults in the code reproduce earlier configurations and are **not** those of the paper. The current scripts first
source `reproduce/env_paper_v3_20261007.sh`, then set the following:

| Variable | Value | Meaning |
|---|---|---|
| `F1_SIMULATION_DIR` | `trained_models/simulation_v10eGQ_20261028` | simulator (effective tire age; single-driver: `simulation_onedriver_v10eGQ_20261028`) |
| `F1_WORLD2024_TYRES` | `v10eGQ` | evaluation races use the 2021-2024 tire coefficients; agents and QMIP keep the 2021-2023 ones |
| `F1_VSC_FIRST_LAP_ONLY` | `1` | under VSC, the agent and the QMIP can stop only on the first lap of the period, like the rivals |
| `F1_TYRE_SETS`, `F1_TYRE_STINT_AGES` | `data/tyre_sets_2024_reg_v10eGQ_20261028.csv`, `data/tyre_race_stint_ages_2024_v10eGQ_20261028.csv` | 2024 tire sets available in each race and the age of each real stint |
| `F1_TYRE_REUSE`, `F1_NO_SET_PENALTY` | `1`, `-1000` | a set taken off can be fitted again; penalty for asking for a compound with no set |
| `F1_RIVAL_START_ALPHA` | from `data/rival_start_alpha_2021_2023_20261018.json` | rivals' starting compounds |
| `F1_LAP0_ACTION0_MEDIUM`, `F1_LAP0_PENALTY` | `1`, `-1000` | action 0 at lap 0: medium compound with a penalty |
| `F1_OVERTAKE_MAX_GAP`, `F1_OVERTAKE_MEAN_GAP`, `F1_FOLLOWING_GAP_SCALE` | `inf`, `1`, `0.26` | overtaking (`scripts/overtake_v10f_20261029.py`) |
| `F1_RACE_EFFECT_SD`, `F1_RACE_EFFECT_KEEP_STD` | `0.28`, `1` | per-race noise on the Eq. 2 driver coefficient (`scripts/eq2_noise_20261030.py`) |

Every training, selection and evaluation runs through a chain of wrapper scripts that applies these options; see `WRAP`
in `reproduce/pipeline_v10g_20261031.sh`.

## Full pipeline (from scratch)

| Step | Command or tasks | Notes |
|---|---|---|
| 1. Simulator | `results/sim_v10eGQ_20261028/run_chain_v10e_20261028.sh`, `scripts/build_simulation_v10e_20261028.py GQ` | steps in `results/sim_v10eGQ_20261028/` |
| 2. Multi-driver agents | `reproduce/cluster/tasks_v10g_ppo8_20261030.tsv`, `tasks_v10g_others_rppo05_20261031.tsv` | cluster; 8 GP, 5 methods, 10 seeds each |
| 3. Single-driver agents | `reproduce/cluster/tasks_onedriver_v10eGQ_20261028.tsv` | cluster; 5 seeds each; selection and figure in `results/onedriver_v10eGQ_20261028/README.md` |
| 4. Seed selection, Figs. 6-7 | `zsh reproduce/pipeline_v10g_20261031.sh all` | about 2 h per figure with 8 jobs |
| 5. Further training (Fig. 11) | `reproduce/cluster/tasks_v10gsm_20261101.tsv`, `tasks_v10gsm_miami_20261102.tsv`, `tasks_v10gsm_rest_20261102.tsv` | selection in `results/startmix_v10g_20261101/README.md` |
| 6. Look-ahead rollouts | `results/decision_support_v10gw24belief_20261103/run_20261103.sh`, `results/decision_support_v10gsmrealhist_20261102/run_20261102.sh` | about 2 h each with 8 workers |
| 7. Figures | `zsh reproduce/figures_v10g_20261105.sh` | about 1 min |

Training writes to new dated trees under `trained_models/`.

### Provenance of the agents

`MODELS_MANIFEST_v10g_20261105.csv`, inside the models archive, lists the seed and SHA-256 of each shipped agent. The
training command of each one is in the task files above. `reproduce/MODELS_MANIFEST*.csv` describe earlier versions.

* Multi-driver: 5 M timesteps with early stopping and an entropy coefficient of 0.05 in PPO and Recurrent PPO. Seeds are
  selected on 200 validation races (`scripts/select_seeds_v5_20261010.py`).
* Further training: PPO only, from the selected agents, 1.5 M timesteps with a random start on half of the races
  (`F1_START_MIX=0.5`). The selection also requires the check with the real start.
* Single-driver: the seed with the smallest gap to the QMIP (`results/onedriver_v10eGQ_20261028/selection_20261028.tsv`).

Training is seeded, but bit-identical retraining across machines and library versions is not guaranteed. The figures
are reproduced from the shipped agents.

## Development history

> The notes below record how the simulator and the agents changed on the way to v2.0.0, in date order. Their status
> lines are as of their dates, and the models of the earlier trees they mention are not shipped. For the versions before
> this work, see the tags `v1.0.0` and `v1.1.0`.

### What changed

| Component | v1.1.0 | v8 | Code |
|---|---|---|---|
| Tire coefficients | matrix completion | joint hierarchical model (offsets and monotone slopes, GP random effect), 2021–2023, with 2021 tire life rescaled by 0.716 | `results/tyre_coef_study_20260930/`, `results/tyrelife_2021_20261015/`, `results/sim_v7_20261015/` |
| Driver coefficients (2024) | Eq. 2 (v1.1.0 version) | Eq. 2 in percent; races where the reference driver has < 15 clean laps are excluded | `results/driver_coef_study_20261010/` |
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

### Start-mix fine-tuning of PPO and RPPO (v9r2sm)

> **Status (2026-10-27).** All 48 tasks done and checked (E1–E4, `results/startmix_proposal_20261025/README.md`). In the
> E4 sweep, mask corrections drop from 9.9 % to 2.2 %; US RPPO still needs the mask. The fine-tuned agents
> (`trained_models/retrained_v9r2_startmix_20261025/selected`) are used **only** for the review of the actual 2024
> decisions in §7, which is Fig. 11 (`scripts/plot_realhist_stints_20261026.py`), with data in
> `results/decision_support_v9r2smrealhist_20261026/`. Only PPO is used there since 2026-10-27; Table Q.1
> (`scripts/plot_realhist_audit_20261026.py`) was removed from the paper. Figs. 5–10 keep the original v9r2 agents.

### Checks added on 2026-10-27

- **Simulated actual 2024 strategy.** `scripts/validate_real_strategy_20261027.py` replays each driver's actual start
  and pit stops in the Fig. 6/7 environments, with the same seeds, and records the final position of all 20 drivers.
  - Runs: `results/real_strategy_check_20261027/run_one_20261027.sh <fig6|fig7> <GP_DRV>`, about 3 min in total.
  - Comparison with the actual classification: `compare_20261027.py`.
  - The paper's Fig. 7 now includes this replay as a grey "Actual" box
    (`results/world2024_v9r2_20261023/fig7_actual_20261027.py`).
- **Look-ahead runtime:** `results/lookahead_timing_20261027/`. One recommendation with M = 500 takes about 1 min on one core.
- **Decision ambiguity per lap:** `results/decision_ambiguity_20261027/`, from the E4 rollouts.
  - Open issue: a forced pit stop has no effect on some SC laps (Saudi lap 8, US lap 4); see that README.

The selected v9r2 PPO/RPPO agents always start on the same compound, and 14 of the 16 are disqualified in 87–100 % of
the races when forced to start on the compound they always pit to (`scripts/start_compound_probe_20261025.py`,
`results/startmix_proposal_20261025/`). With `F1_START_MIX=p`, `reset()` plays lap 0 itself with probability `p`
(uniform compound, valid action) and the episode starts at lap 1; without the variable everything is bit-identical.
`scripts/finetune_startmix_20261025.py` continues training a selected agent from its weights and VecNormalize stats
(p = 0.5, constant lr 1e-4 PPO / 1.5e-4 RPPO, 1.5 M steps, evaluation every 50 000 steps with the same mix,
`--vec dummy` = the 10 envs in one process, same results). 48 tasks (8 GP × 2 algorithms × 3 seeds) in
`reproduce/cluster/tasks_v9r2sm.tsv` → `trained_models/retrained_v9r2_startmix_20261025/seedXX/`.

### Tire sets at the decision points of Fig. 11 (2026-11-03)

Table L.1 of the supplementary material lists, for every row of Fig. 11, the sets the driver still has at that lap,
with the age of each one, and the set on the car. It answers why the hard tire sometimes runs fewer laps than a
softer one: in the United States and Singapore the driver has a single hard set, so its stint length is the remainder
of the plan.

- `scripts/tyre_sets_at_decisions_20261103.py` rebuilds the driver's actual prefix with the same chain as the figure
  (`gp_configs_fix_20261028.py` + `ds_rollouts_realhist_20261024.py --run`, `F1_DS_FORCE=hist`, `--real-2024`,
  prefix seed 0) and dumps, at each decision lap, the mounted set with its effective age and the remaining pool per
  compound. With `F1_TYRE_REUSE=1` the set taken off returns to the pool only at the stop, so it is reported apart.
- `scripts/table_tyre_sets_decisions_20261103.py` builds the LaTeX table, one row per row of Fig. 11 (same groups,
  representative lap = largest |difference|).
- Runner, per-GP JSON and the LaTeX fragment: `results/decision_support_v10gsmrealhist_20261102/tyre_sets_20261103/`
  (17 s in total).

### Lap-by-lap value of a recommendation (2026-11-03)

Figure Q.1 of the supplementary material plots the mean paired difference in points, with 2 SE, for each lap around
the second pit stop of the Bahrain Grand Prix. It shows that within a group of laps carrying the same recommendation
the value of following it changes from lap to lap: stopping pays only from lap 38 and peaks at lap 39.

- `scripts/plot_realhist_dlap_20261103.py --res results/decision_support_v10gsmrealhist_20261102 --gp bahrain_alo
  --from-lap 33 --to-lap 42` (with `F1_PAPER_FONTS=1`). It reads the sweep CSV, so it runs no new rollouts.
- Output: `results/decision_support_v10gsmrealhist_20261102/figs/realhist_dlap_<gp>.{pdf,png}`, copied to the paper
  folder as `realhist_dlap_bahrain.pdf`.

## Citation

```bibtex
@article{mendez_thraves_pitstops,
  author  = {M{\'e}ndez, Felipe and Thraves, Charles},
  title   = {{AI}-Assisted Decision Support for {Formula} 1 Pit-Stop Strategy: Deep Reinforcement Learning and Mathematical Programming},
  journal = {Annals of Operations Research},
  note    = {Under review},
  year    = {2026}
}
@software{mendez_thraves_code,
  author    = {M{\'e}ndez, Felipe and Thraves, Charles},
  title     = {Data and Code for: {AI}-Assisted Decision Support for {Formula} 1 Pit-Stop Strategy},
  version   = {2.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20060265},
  url       = {https://doi.org/10.5281/zenodo.20060265}
}
```

## Version history

* **v2.0.0 (2026)**: version of the *Annals of Operations Research* manuscript, with the simulator v10eGQ and the agents
  v10g and v10gsm. See "Development history" for what changed.
* v1.1.0 / v1.0.0 (2026): earlier submission snapshots (version DOIs `10.5281/zenodo.20060266` for v1.0.0 and `10.5281/zenodo.20114287` for v1.1.0).

## License

MIT — see [`LICENSE`](LICENSE). Race data were obtained with [FastF1](https://github.com/theOehrly/Fast-F1)
from the public Formula 1 timing feeds.
