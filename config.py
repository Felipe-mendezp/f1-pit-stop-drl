"""
Centralized configuration for the F1 Pit Stop DRL project.
All paths are relative to the repository root.
"""
import os
import sys
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"

# ── Trained models ────────────────────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "trained_models"
# F1_SIMULATION_DIR lets evaluation/training point at an alternative simulation
# model tree (e.g. trained_models/simulation_pitlogit_20260807, which swaps the
# overtake logit for the pit-aware refit) without touching the original tree.
SIMULATION_DIR = Path(os.environ.get("F1_SIMULATION_DIR", str(MODELS_DIR / "simulation")))
# F1_RL_AGENTS_DIR lets training/evaluation point at an alternative agent set
# (e.g. trained_models/rl_agents_v2) without touching the paper (v1) models.
RL_AGENTS_DIR = Path(os.environ.get("F1_RL_AGENTS_DIR", str(MODELS_DIR / "rl_agents")))
# F1_RL_REWARD_DIR: same mechanism for the reward-comparison agent set
# (training output AND evaluation input), so re-runs never touch the
# original rl_agents_reward models.
RL_REWARD_DIR = Path(os.environ.get("F1_RL_REWARD_DIR", str(MODELS_DIR / "rl_agents_reward")))

# ── Source code ───────────────────────────────────────────────────────────────
SRC_DIR = ROOT_DIR / "src"

# ── Ensure src/ and root are importable ───────────────────────────────────────
for _p in (str(SRC_DIR), str(ROOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
