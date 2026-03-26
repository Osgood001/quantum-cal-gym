"""
random_agent.py — smoke test / demo for QubitCalibration-v0.

Mirrors the style of fib-gym's random_agent.py.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
import quantum_cal_gym  # noqa: F401 — registers QubitCalibration-v0

env = gym.make("QubitCalibration-v0", config={"max_steps": 50})
obs, _ = env.reset(seed=42)

print("QubitCalibration-v0  —  random agent")
print(f"action_space : {env.action_space}")
print(f"obs keys     : {list(obs.keys())}")
print()

total_reward = 0.0
for step in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    bar = "█" * info["n_calibrated"] + "░" * (6 - info["n_calibrated"])
    print(
        f"step {step+1:2d} | {info['exp_type']:12s} | "
        f"[{bar}] {info['n_calibrated']}/6 | reward={reward:+.2f}"
    )
    if terminated:
        break

print()
print(f"total reward : {total_reward:.2f}")
print(f"calibrated   : {sorted(info['calibrated'])}")
env.close()
