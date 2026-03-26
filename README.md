<p align="center">
  <img src="logo.svg" width="480" alt="QC Gym logo">
</p>

<h3 align="center">Gymnasium environment for qubit calibration</h3>

<p align="center">
  <a href="https://gymnasium.farama.org/"><img alt="Gymnasium" src="https://img.shields.io/badge/Gymnasium-v1-blue?logo=openai"></a>
  <a href="https://numpy.org/"><img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.24+-orange"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

---

**QC Gym** wraps a physics-based superconducting transmon qubit simulator into an [OpenAI Gymnasium](https://gymnasium.farama.org/) reinforcement-learning environment.

An agent controls which experiment to run (S21, Spectrum, PowerRabi, TimeRabi, T1, Ramsey) and the sweep parameters. The simulator responds with measured IQ signals from a hidden qubit whose parameters must be discovered through adaptive experimentation.

> **Status:** v0 lightweight proof-of-concept. Single-qubit dispersive readout model. Multi-qubit and real-hardware backends planned.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Osgood001/quantum-cal-gym.git
cd quantum-cal-gym

# 2. Install dependencies
pip install gymnasium numpy

# 3. Run the example
python examples/random_agent.py
```

## Environment: `QubitCalibration-v0`

```python
import gymnasium as gym
import quantum_cal_gym

env = gym.make("QubitCalibration-v0")
obs, info = env.reset()

for _ in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break

env.close()
```

### Action Space

| Index | Name | Range | Maps to |
|-------|------|-------|---------|
| 0 | `exp_type` | [0, 1] | Experiment: 0=S21, 1=Spectrum, 2=PowerRabi, 3=TimeRabi, 4=T1, 5=Ramsey |
| 1 | `sweep_center` | [0, 1] | Frequency centre or Ramsey detuning (normalised) |
| 2 | `sweep_span` | [0, 1] | Frequency span or maximum time (normalised) |
| 3 | `amplitude` | [0, 1] | Drive amplitude |
| 4 | `delay_frac` | [0, 1] | Maximum delay for time-domain experiments |

### Observation Space

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `signal_re` | `(64,)` | `float32` | Real part of measured IQ signal |
| `signal_im` | `(64,)` | `float32` | Imaginary part |
| `x_axis` | `(64,)` | `float32` | Sweep axis values, normalised to [0, 1] |
| `state` | `(9,)` | `float32` | Parameter estimates + calibration progress |

### Reward

+1 per newly calibrated parameter (within 5 % of true value), −0.02 per step, +10 on full calibration.

### Configuration

```python
env = gym.make("QubitCalibration-v0", config={"max_steps": 100})
```

## Coding Agents as Calibration Agents

Beyond RL, this gym supports a newer paradigm: **coding agents** (LLM-based agents
that can reason and call tools) as scientific experimenters.  The idea is simple —
give the agent an HTTP API to a hidden qubit and a budget of 30 calls, then let it
design and execute its own calibration protocol.

```
┌─────────────────────────┐        HTTP        ┌──────────────────────────┐
│   Coding Agent          │◄──────────────────►│  quantum_cal_gym/server  │
│  (Claude Code, GPT-4o…) │  POST /session/run  │  FastAPI + TransmonSim   │
│                         │  POST /session/sub  │  EpisodeLogger           │
└─────────────────────────┘                     └──────────────────────────┘
         reasons, plans,                           true params hidden;
         fits curves,                              plots + log.json saved
         adapts strategy                           to runs/server_sessions/
```

The agent never sees the simulator source or the true parameters.
It only observes the JSON signal data returned by each experiment call —
exactly the information a real experimentalist would have.

### Why this is interesting

- **Coding agents generalise differently from RL agents.** They bring physics
  knowledge, curve-fitting intuition, and adaptive planning from pretraining —
  without any task-specific training.
- **The gym becomes an evaluation benchmark.** How many parameters can an agent
  calibrate in 30 shots?  Does it zoom in on resonances?  Does it set Ramsey
  detuning correctly?
- **The interface is minimal by design.** The server exposes only signal data.
  The agent must reason about physics from first principles.

### Quick setup

**Terminal 1 — start the hidden-qubit server:**

```bash
cd quantum-cal-gym
uvicorn quantum_cal_gym.server:app --port 8765
```

**Terminal 2 — run Claude Code as the agent** (requires a separate workspace so
the agent has no access to the gym source):

```bash
# one-time setup
git clone https://github.com/Osgood001/qcal-agent ~/qcal-agent
# or: mkdir ~/qcal-agent && cp docs/CLAUDE_AGENT.md ~/qcal-agent/CLAUDE.md

cd ~/qcal-agent
claude          # Claude Code reads CLAUDE.md and starts calibrating
```

Claude Code will autonomously create a session, run experiments, analyse the
returned JSON, and submit its final estimates.  Watch its reasoning live in the
TUI.

### What gets saved

After each experiment, `runs/server_sessions/{sid}/` contains:

| File | Contents |
|------|----------|
| `step_NNN_<exp>.png` | Signal curve with true-value marker + IQ plane panel |
| `log.json` | Structured record of every call (exp type, params, signal summary) |
| `progress.png` | Experiment timeline regenerated after each step |

These are produced by the same `EpisodeLogger` used by the Gymnasium env —
no separate logging infrastructure needed.

### Headless / scripted mode

For automated benchmarking, `examples/cal_agent_harness.py` spawns Claude Code
non-interactively and scores the result:

```bash
python examples/cal_agent_harness.py --seed 42
```

## Project Structure

```
quantum-cal-gym/
  quantum_cal_gym/
    __init__.py          # Gymnasium registration
    cal_env.py           # QubitCalibrationEnv class
    qubit_sim.py         # Transmon qubit physics simulator
    mock_quark.py        # Drop-in stub for the lab quark SDK
    server.py            # FastAPI server for coding-agent sessions
    logger.py            # EpisodeLogger: per-step plots + log.json
  examples/
    random_agent.py      # RL smoke test / demo
    run_exp.py           # CLI experiment runner (used by headless harness)
    cal_agent_harness.py # Spawn + score Claude Code as a calibration agent
  docs/
    report.md            # Technical report (Chinese, with physics derivations)
  pyproject.toml
  logo.svg
```

## Physics Model

The simulator models a single transmon qubit with dispersive readout:

- **S21** — complex Lorentzian transmission centred at resonator frequency `f_r`
- **Spectrum** — qubit absorption dip at `f_q`, linewidth `1/(π T2*)`
- **PowerRabi** — sinusoidal population oscillation vs drive amplitude
- **TimeRabi** — sinusoidal oscillation with T1 decay envelope
- **T1** — exponential population decay `exp(−t/T1)`
- **Ramsey** — damped cosine `0.5(1 + cos(2π·Δf·t)·exp(−t/T2*))`

Hidden parameters drawn uniformly at each `reset()`:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `f_q` | 4 – 6 GHz | Qubit frequency |
| `f_r` | `f_q` + 0.5 – 2 GHz | Readout resonator |
| `T1` | 10 – 200 µs | Energy relaxation |
| `T2*` | 5 – 100 µs | Dephasing time |
| `amp_pi` | 0.3 – 0.9 | π-pulse amplitude |
| `t_pi` | 20 – 200 ns | π-pulse duration |

## Mock quark SDK

`mock_quark.py` provides a drop-in stub for the proprietary `quark` lab SDK so notebook-style experiment code can run offline:

```python
from quantum_cal_gym.mock_quark import install
install()   # injects quark.app into sys.modules

from quark.app import Recipe, s
s.login()

rcp = Recipe("T1")
rcp["delay"] = np.linspace(0, 200e-6, 64)
result = rcp.run()
```

## Motivation

Quantum processor calibration is bottlenecked by expensive hardware time. A physics-based simulation environment enables fast, low-cost development of:

- **RL agents** learning optimal calibration strategies
- **AI scientists** exploring superconducting qubit parameter spaces
- **Benchmarking** autonomous calibration systems

Inspired by [fib-gym](https://github.com/Osgood001/fib-gym), [ChemGymRL](https://docs.chemgymrl.com/), and [qualibrate](https://github.com/qua-platform/qualibrate).

## Roadmap

- [x] `QubitCalibration-v0` — single-qubit, 6 experiment types
- [x] Physics backend — scqubits (EJ/EC→f_q) + QuTiP Lindblad solver
- [x] FastAPI server + `EpisodeLogger` for coding-agent sessions
- [x] Claude Code calibration agent (`examples/cal_agent_harness.py`)
- [ ] Flux-tunable qubit (f_q vs bias curve)
- [ ] Two-qubit experiments (CZ, iSWAP)
- [ ] Real hardware backend via `quark` SDK
- [ ] Shaped reward functions for calibration graphs
- [ ] `pip install quantum-cal-gym` (PyPI package)

## License

MIT — see [LICENSE](LICENSE).
