# trained_models/

This folder is empty in the git repository. The simulator and the agents used in the manuscript are distributed in the
Zenodo record of this version, https://doi.org/10.5281/zenodo.20060265 (latest version), which lists the checksum of
each file. Extract the archives at the repository root:

    tar -xzf f1-pit-stop-drl-v2.0.0-models.tar.gz      # 106 MB compressed, 181 MB extracted
    tar -xzf f1-pit-stop-drl-v2.0.0-rollouts.tar.gz    # 51 MB compressed, 183 MB extracted

## f1-pit-stop-drl-v2.0.0-models.tar.gz

| Path | What | Used by |
|---|---|---|
| `simulation_v10eGQ_20261028/<GP>/` (8 GPs) | lap-time, overtake and rival models and `config.json` | multi-driver environment, every figure from 6 on |
| `simulation_onedriver_v10eGQ_20261028/` | links into the above, for the single-driver environment | Fig. 5, Table N.1 |
| `simulation/`, `simulation_v6_20261014/` | only the files that the two trees above link to | same |
| `retrained_v10g_20261030/selected/<GP>_<DRIVER>/<ALGO>/` | 40 multi-driver agents, with `SELECTED_SEED.txt` | Figs. 6-10 |
| `retrained_v10g_startmix_20261101/selected/<GP>_<DRIVER>/PPO/` | 8 PPO agents after further training | Fig. 11, Fig. Q.1, Table L.1 |
| `retrained_v10eGQ_onedriver_20261028/seedNN/one_driver/<GP>/<ALGO>/` | 40 single-driver agents, reached through the links in `results/onedriver_v10eGQ_20261028/selected` | Fig. 5, Table N.1 |
| `MODELS_MANIFEST_v10g_20261105.csv` (at the repository root) | seed, path and SHA-256 of each of the 88 agents | |

Only the selected seeds are shipped. Built with `reproduce/pack_models_v10g_20261105.py`.

## f1-pit-stop-drl-v2.0.0-rollouts.tar.gz

Look-ahead rollouts, not versioned in git because of their size. Built with `reproduce/pack_results_v10g_20261105.py`.

| Path | Used by |
|---|---|
| `results/decision_support_v10gw24belief_20261103/PPO/` | Figs. 8-10 |
| `results/decision_support_v10gsmrealhist_20261102/PPO/` | Fig. 11, Table L.1 |
