"""
Centralized configuration for the F1 Pit Stop DRL project.
All paths are relative to the repository root.
"""
from pathlib import Path
import sys

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"

# ── Trained models ────────────────────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "trained_models"
SIMULATION_DIR = MODELS_DIR / "simulation"
RL_AGENTS_DIR = MODELS_DIR / "rl_agents"
RL_REWARD_DIR = MODELS_DIR / "rl_agents_reward"

# ── Source code ───────────────────────────────────────────────────────────────
SRC_DIR = ROOT_DIR / "src"

# ── Ensure src/ and root are importable ───────────────────────────────────────
for _p in (str(SRC_DIR), str(ROOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
