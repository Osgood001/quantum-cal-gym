"""quantum_cal_gym — Gymnasium environments for quantum qubit calibration."""
from __future__ import annotations

import gymnasium as gym

from .cal_env import QubitCalibrationEnv  # noqa: F401

gym.register(
    id="QubitCalibration-v0",
    entry_point="quantum_cal_gym.cal_env:QubitCalibrationEnv",
    max_episode_steps=50,
)
