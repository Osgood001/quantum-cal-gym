"""
QubitCalibration-v0 — Gymnasium environment for autonomous qubit calibration.

The agent chooses which experiment to run (S21, Spectrum, Rabi, T1, Ramsey)
and with what sweep parameters. The environment runs the experiment on a hidden
superconducting qubit simulator and returns the measured IQ signal.

A lightweight built-in analyser converts each result into updated parameter
estimates, which accumulate in the ``state`` observation. Reward is earned for
newly-calibrated parameters (within 5 % of true value).
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .qubit_sim import (
    SuperconductingQubit,
    FREQ_LO, FREQ_HI, FREQ_SPAN_MIN, FREQ_SPAN_MAX,
    TIME_MIN, TIME_MAX, AMP_PI_RANGE,
)

# ── Constants ─────────────────────────────────────────────────────────────────

N_POINTS = 64          # sweep points per experiment
CAL_TOL  = 0.05        # 5 % relative error → "calibrated"

EXP_TYPES = ['s21', 'spectrum', 'power_rabi', 'time_rabi', 't1', 'ramsey']
N_EXP     = len(EXP_TYPES)

PARAM_NAMES = ['f_q', 'f_r', 'T1', 'T2star', 'amp_pi', 't_pi']
N_PARAMS    = len(PARAM_NAMES)

_DEFAULT_CONFIG = dict(max_steps=50)


class QubitCalibrationEnv(gym.Env):
    """Qubit calibration Gymnasium environment.

    Action  (Box[5], each in [0, 1])
    ────────────────────────────────
    0  exp_type_frac   → experiment index = floor(a[0] * 6) in {0..5}
                         0=s21, 1=spectrum, 2=power_rabi,
                         3=time_rabi, 4=t1, 5=ramsey
    1  sweep_center    → frequency centre or Ramsey detuning (normalised)
    2  sweep_span      → frequency span or maximum time (normalised)
    3  amplitude       → drive amplitude (normalised, min 0.01)
    4  delay_frac      → max delay for time-domain expts (normalised)

    Observation  (Dict)
    ───────────────────
    signal_re   (N,)  float32  real part of measured IQ
    signal_im   (N,)  float32  imaginary part
    x_axis      (N,)  float32  x values normalised to [0, 1]
    state       (9,)  float32  parameter estimates + progress indicators

    state layout
    ────────────
    [0]  f_q_est   / 1e10     (≈ 0.4–0.6 for 4–6 GHz)
    [1]  f_r_est   / 1e10
    [2]  T1_est    / 200e-6   (0–1 for 0–200 µs)
    [3]  T2star_est/ 100e-6
    [4]  amp_pi_est            (0–1)
    [5]  t_pi_est  / 200e-9   (0–1 for 0–200 ns)
    [6]  n_calibrated / N_PARAMS
    [7]  step / max_steps
    [8]  last_exp / (N_EXP-1)

    Reward
    ──────
    +1.0 per newly calibrated parameter
    -0.02 per step (efficiency incentive)
    +10.0 bonus on full calibration
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = {**_DEFAULT_CONFIG, **(config or {})}
        self._max_steps = int(cfg['max_steps'])

        self.action_space = spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'signal_re': spaces.Box(-2.0, 2.0,        shape=(N_POINTS,), dtype=np.float32),
            'signal_im': spaces.Box(-2.0, 2.0,        shape=(N_POINTS,), dtype=np.float32),
            'x_axis':    spaces.Box( 0.0, 1.0,        shape=(N_POINTS,), dtype=np.float32),
            'state':     spaces.Box(-np.inf, np.inf,  shape=(9,),        dtype=np.float32),
        })

        self._qubit: SuperconductingQubit | None = None
        self._step_count  = 0
        self._estimates   = {k: 0.0 for k in PARAM_NAMES}
        self._calibrated  : set[str] = set()
        self._last_exp_idx = 0

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._qubit       = SuperconductingQubit(seed=seed)
        self._step_count  = 0
        self._estimates   = {k: 0.0 for k in PARAM_NAMES}
        self._calibrated  = set()
        self._last_exp_idx = 0
        return self._null_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(0.0, 1.0)

        exp_idx = min(int(action[0] * N_EXP), N_EXP - 1)
        exp     = EXP_TYPES[exp_idx]
        sweep_c = float(action[1])
        sweep_s = float(max(action[2], 0.02))
        amp     = float(max(action[3], 0.01))
        delay_f = float(action[4])

        x_norm, signal = self._run_experiment(exp, sweep_c, sweep_s, amp, delay_f)

        prev_cal = len(self._calibrated)
        self._analyse(exp, x_norm, signal, sweep_c, delay_f)
        new_cal  = len(self._calibrated)

        self._step_count  += 1
        self._last_exp_idx = exp_idx

        terminated = (new_cal == N_PARAMS) or (self._step_count >= self._max_steps)
        reward     = float(new_cal - prev_cal) - 0.02
        if new_cal == N_PARAMS:
            reward += 10.0

        obs  = self._make_obs(x_norm, signal)
        info = {
            'exp_type':    exp,
            'calibrated':  list(self._calibrated),
            'n_calibrated': new_cal,
            'estimates':   dict(self._estimates),
        }
        return obs, reward, terminated, False, info

    def close(self):
        pass

    # ── Experiment runner ──────────────────────────────────────────────────────

    def _run_experiment(self, exp: str, sweep_c, sweep_s, amp, delay_f):
        q = self._qubit

        if exp in ('s21', 'spectrum'):
            f_c   = sweep_c * (FREQ_HI - FREQ_LO) + FREQ_LO
            f_s   = sweep_s * (FREQ_SPAN_MAX - FREQ_SPAN_MIN) + FREQ_SPAN_MIN
            freqs = np.linspace(f_c - f_s / 2, f_c + f_s / 2, N_POINTS)
            signal = q.s21(freqs) if exp == 's21' else q.spectrum(freqs, amp=amp * 0.8 + 0.2)
            x_norm = (freqs - FREQ_LO) / (FREQ_HI - FREQ_LO)

        elif exp == 'power_rabi':
            amp_max = amp * AMP_PI_RANGE[1] * 2.5   # generous range
            amps    = np.linspace(0, amp_max, N_POINTS)
            signal  = q.power_rabi(amps)
            x_norm  = amps / (AMP_PI_RANGE[1] * 2.5)

        elif exp == 'time_rabi':
            t_max  = max(delay_f, 0.01) * 500e-9    # up to 500 ns
            times  = np.linspace(0, t_max, N_POINTS)
            signal = q.time_rabi(times)
            x_norm = times / 500e-9

        elif exp == 't1':
            t_max  = max(delay_f, 0.02) * TIME_MAX
            times  = np.linspace(0, t_max, N_POINTS)
            signal = q.t1_decay(times)
            x_norm = times / TIME_MAX

        elif exp == 'ramsey':
            t_max    = max(delay_f, 0.02) * TIME_MAX
            times    = np.linspace(0, t_max, N_POINTS)
            detuning = sweep_c * 20e6              # 0–20 MHz
            signal   = q.ramsey(times, detuning=detuning)
            x_norm   = times / TIME_MAX

        else:
            x_norm = np.linspace(0.0, 1.0, N_POINTS)
            signal = np.zeros(N_POINTS, dtype=complex)

        return x_norm.astype(np.float32), signal

    # ── Parameter analysis ────────────────────────────────────────────────────

    def _analyse(self, exp: str, x_norm, signal, sweep_c, delay_f):
        """Update parameter estimates from the latest experiment result."""
        mag = np.abs(signal)
        q   = self._qubit

        if exp == 's21':
            idx   = int(np.argmin(mag))
            f_est = x_norm[idx] * (FREQ_HI - FREQ_LO) + FREQ_LO
            self._set('f_r', f_est, q.f_r)

        elif exp == 'spectrum':
            idx   = int(np.argmin(mag))
            f_est = x_norm[idx] * (FREQ_HI - FREQ_LO) + FREQ_LO
            self._set('f_q', f_est, q.f_q)

        elif exp == 'power_rabi':
            # First local maximum = pi-pulse amplitude
            idx = _first_peak(mag)
            if idx is not None:
                amp_est = x_norm[idx] * AMP_PI_RANGE[1] * 2.5
                self._set('amp_pi', amp_est, q.amp_pi)

        elif exp == 'time_rabi':
            idx = _first_peak(mag)
            if idx is not None:
                t_est = x_norm[idx] * 500e-9
                self._set('t_pi', t_est, q.t_pi)

        elif exp == 't1':
            # 1/e crossing
            t_max  = max(delay_f, 0.02) * TIME_MAX
            thresh = mag[0] / np.e
            below  = np.where(mag < thresh)[0]
            if len(below):
                T1_est = x_norm[below[0]] * TIME_MAX
                if T1_est > 0:
                    self._set('T1', T1_est, q.T1)

        elif exp == 'ramsey':
            # Estimate T2* from decay of oscillation envelope
            t_max   = max(delay_f, 0.02) * TIME_MAX
            env     = _smooth_envelope(mag)
            thresh  = env[0] / np.e
            below   = np.where(env < thresh)[0]
            if len(below):
                T2_est = x_norm[below[0]] * TIME_MAX
                if T2_est > 0:
                    self._set('T2star', T2_est, q.T2star)

    def _set(self, param: str, estimate: float, true_val: float):
        self._estimates[param] = estimate
        rel_err = abs(estimate - true_val) / max(abs(true_val), 1e-30)
        if rel_err < CAL_TOL:
            self._calibrated.add(param)
        else:
            self._calibrated.discard(param)

    # ── Observation builders ──────────────────────────────────────────────────

    def _make_obs(self, x_norm, signal) -> dict:
        e = self._estimates
        state = np.array([
            e['f_q']    / 1e10,
            e['f_r']    / 1e10,
            e['T1']     / 200e-6,
            e['T2star'] / 100e-6,
            e['amp_pi'],
            e['t_pi']   / 200e-9,
            len(self._calibrated) / N_PARAMS,
            self._step_count / self._max_steps,
            self._last_exp_idx / (N_EXP - 1),
        ], dtype=np.float32)
        return {
            'signal_re': signal.real.astype(np.float32),
            'signal_im': signal.imag.astype(np.float32),
            'x_axis':    x_norm.astype(np.float32),
            'state':     state,
        }

    def _null_obs(self) -> dict:
        return {
            'signal_re': np.zeros(N_POINTS, dtype=np.float32),
            'signal_im': np.zeros(N_POINTS, dtype=np.float32),
            'x_axis':    np.linspace(0, 1, N_POINTS, dtype=np.float32),
            'state':     np.zeros(9, dtype=np.float32),
        }


# ── Utility functions ─────────────────────────────────────────────────────────

def _first_peak(arr: np.ndarray, min_order: int = 3) -> int | None:
    """Return index of first local maximum (simple finite-difference method)."""
    diff = np.diff(arr)
    # Sign changes from + to -
    signs = np.sign(diff)
    for i in range(len(signs) - 1):
        if signs[i] > 0 and signs[i + 1] <= 0 and i + 1 >= min_order:
            return i + 1
    return None


def _smooth_envelope(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving-average envelope approximation."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')
