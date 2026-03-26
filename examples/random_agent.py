"""
random_agent.py — smoke test / demo for QubitCalibration-v0.

Runs one episode with a random agent and saves per-step plots +
a calibration progress chart to runs/episode_NNNN/.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
import quantum_cal_gym                          # registers QubitCalibration-v0
from quantum_cal_gym.logger import EpisodeLogger

# ── Setup ─────────────────────────────────────────────────────────────────────

SEED     = 42
LOG_ROOT = os.path.join(os.path.dirname(__file__), "..", "runs")
run_id   = int(time.time())
log_dir  = os.path.join(LOG_ROOT, f"episode_{run_id}")

env    = gym.make("QubitCalibration-v0", config={"max_steps": 50})
logger = EpisodeLogger(log_dir, verbose=True)

# ── Episode loop ──────────────────────────────────────────────────────────────

obs, _ = env.reset(seed=SEED)

print("QubitCalibration-v0  —  random agent")
print(f"log dir : {log_dir}")
print(f"action  : {env.action_space}")
print()

total_reward = 0.0
for step in range(1, 51):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    logger.record(step, obs, reward, info)

    if terminated:
        break

# ── Save logs ─────────────────────────────────────────────────────────────────

print()
logger.save()
print()
print(f"total reward : {total_reward:.2f}")
print(f"calibrated   : {sorted(info['calibrated'])}")
print(f"true params  : { {k: f'{v:.4g}' for k, v in info['true_params'].items()} }")
env.close()
