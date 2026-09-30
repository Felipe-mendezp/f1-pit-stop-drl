# trained_models/

This folder is empty in the git repository. The simulator and all trained agents are distributed
as a separate archive in the Zenodo record of this release:

    f1-pit-stop-drl-v2.0.0-models.tar.gz   (116 MB compressed, ~190 MB extracted)
    SHA-256  fae7c6db55eab59daaf5eeadff467a6cd1c7701071cb682e241d69ba00f8f8fe

Download it from https://doi.org/10.5281/zenodo.20060265 (latest version) and extract it at the
repository root:

    tar -xzf f1-pit-stop-drl-v2.0.0-models.tar.gz        # creates trained_models/{simulation,rl_agents,rl_agents_reward}

Contents

| Path | What | Used by |
|---|---|---|
| `simulation/<GP>/` (9 GPs) | lap-time (clear track and yellow flag), overtake and rival models, `config.json` | every environment |
| `simulation/<GP>_fast_predictor.pkl` | clear-track lap-time predictor | single-driver environment |
| `rl_agents/<GP>_<DRIVER>/<ALGO>/` | 45 multi-driver agents (`best_model.zip`, `vecnormalize_stats.pkl`) | Figs. 6-10 |
| `rl_agents/*.csv` | cached 500-race results of Figs. 6 and 7 (`--plot-only`) | Figs. 6-7 |
| `rl_agents/one_driver/<GP>/<ALGO>/` | 45 single-driver agents | Fig. 5, Table N.1 |
| `rl_agents_reward/<GP>/RecurrentPPO*/` (mix, time, position, points) | reward-comparison agents + cached results | Supplementary Section P |

`rl_agents/PROVENANCE_dsq1000fix.md`, `rl_agents/one_driver/PROVENANCE.txt` and
`rl_agents_reward/README.md` are the original provenance notes (partly in Spanish); the
authoritative, per-model record is `reproduce/MODELS_MANIFEST.csv`.
