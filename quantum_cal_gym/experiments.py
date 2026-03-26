"""
experiments.py — Ctoolbox-aligned characterize / analyze wrappers.

Mirrors the Ctoolbox pattern:
    task = characterize(qubit, ...)
    params = analyze(task)

Each function uses ``mock_quark.s.submit()`` internally, so the same
code runs against the simulator or (if mock_quark is not installed) the
real quark SDK.

Usage
-----
    from quantum_cal_gym.mock_quark import install, reset
    install()   # optional: patch sys.modules
    reset(seed=42)

    from quantum_cal_gym import experiments as exp
    task = exp.characterize_s21("Q0")
    result = exp.analyze_s21(task)
    print(result)   # {'f_r': 5.42e9, 'kappa': 3.1e6, ...}
"""
from __future__ import annotations

import numpy as np

# Import s either from the real SDK (if installed) or mock
try:
    from quark.app import s, Recipe
except ImportError:
    from .mock_quark import s, Recipe


# ── S21 ───────────────────────────────────────────────────────────────────────

def characterize_s21(qubit: str = "Q0", *,
                     f_center: float | None = None,
                     span: float = 200e6,
                     n_pts: int = 64) -> object:
    """
    Sweep resonator transmission around ``f_center``.

    Parameters
    ----------
    qubit    : qubit label, e.g. "Q0"
    f_center : probe centre frequency (Hz); defaults to current config value
    span     : frequency span (Hz)
    n_pts    : number of sweep points
    """
    if f_center is None:
        f_center = s.query(f"{qubit}.Measure.frequency") or 6.0e9
    freqs = np.linspace(f_center - span / 2, f_center + span / 2, n_pts)
    rcp = Recipe("s21", signal="IQ")
    rcp["qubit"] = qubit
    rcp["freq"]  = freqs
    return s.submit(rcp.export())


def analyze_s21(task) -> dict:
    """
    Extract resonator frequency and linewidth from S21 data.

    Returns
    -------
    dict with keys: f_r (Hz), kappa (Hz)
    """
    result = task.result()
    sig    = result["meta"]["other"]["signal"]
    qubit  = result["meta"]["other"]["qubits"][0]
    iq     = result["data"][sig][:, 0]              # (n_pts,) complex
    freqs  = list(result["meta"]["axis"].values())[0]["def"]

    mag    = np.abs(iq)
    idx    = int(np.argmin(mag))
    f_r    = float(freqs[idx])

    # Estimate half-power linewidth from points around minimum
    half   = (mag.max() + mag.min()) / 2.0
    above  = np.where(mag > half)[0]
    if len(above) >= 2:
        kappa = float(abs(freqs[above[-1]] - freqs[above[0]]))
    else:
        kappa = float(freqs[1] - freqs[0]) * 4

    return {"f_r": f_r, "kappa": kappa}


# ── Spectrum ──────────────────────────────────────────────────────────────────

def characterize_spectrum(qubit: str = "Q0", *,
                           f_center: float | None = None,
                           span: float = 100e6,
                           n_pts: int = 64) -> object:
    """Qubit spectroscopy sweep around ``f_center``."""
    if f_center is None:
        f_center = s.query(f"{qubit}.R.frequency") or 5.0e9
    freqs = np.linspace(f_center - span / 2, f_center + span / 2, n_pts)
    rcp = Recipe("spectrum", signal="IQ")
    rcp["qubit"] = qubit
    rcp["freq"]  = freqs
    return s.submit(rcp.export())


def analyze_spectrum(task) -> dict:
    """Extract qubit frequency from spectroscopy dip. Returns {'f_q': Hz}."""
    result = task.result()
    sig    = result["meta"]["other"]["signal"]
    iq     = result["data"][sig][:, 0]
    freqs  = list(result["meta"]["axis"].values())[0]["def"]
    mag    = np.abs(iq)
    f_q    = float(freqs[int(np.argmin(mag))])
    return {"f_q": f_q}


# ── PowerRabi ─────────────────────────────────────────────────────────────────

def characterize_power_rabi(qubit: str = "Q0", *,
                             amp_max: float = 1.0,
                             n_pts: int = 64) -> object:
    """Sweep drive amplitude from 0 to ``amp_max``."""
    amps = np.linspace(0.0, amp_max, n_pts)
    rcp  = Recipe("power_rabi", signal="IQ")
    rcp["qubit"] = qubit
    rcp["amp"]   = amps
    return s.submit(rcp.export())


def analyze_power_rabi(task) -> dict:
    """Find first π-pulse amplitude from population peak. Returns {'amp_pi': arb}."""
    result = task.result()
    sig    = result["meta"]["other"]["signal"]
    iq     = result["data"][sig][:, 0]
    amps   = list(result["meta"]["axis"].values())[0]["def"]
    mag    = np.abs(iq)
    idx    = _first_peak(mag)
    amp_pi = float(amps[idx]) if idx is not None else float(amps[len(amps) // 2])
    return {"amp_pi": amp_pi}


# ── TimeRabi ──────────────────────────────────────────────────────────────────

def characterize_time_rabi(qubit: str = "Q0", *,
                            width_max: float = 300e-9,
                            n_pts: int = 64) -> object:
    """Sweep drive width from 0 to ``width_max``."""
    widths = np.linspace(0.0, width_max, n_pts)
    rcp    = Recipe("time_rabi", signal="IQ")
    rcp["qubit"] = qubit
    rcp["width"] = widths
    return s.submit(rcp.export())


def analyze_time_rabi(task) -> dict:
    """Find first π-pulse duration from population peak. Returns {'t_pi': s}."""
    result = task.result()
    sig    = result["meta"]["other"]["signal"]
    iq     = result["data"][sig][:, 0]
    widths = list(result["meta"]["axis"].values())[0]["def"]
    mag    = np.abs(iq)
    idx    = _first_peak(mag)
    t_pi   = float(widths[idx]) if idx is not None else float(widths[len(widths) // 2])
    return {"t_pi": t_pi}


# ── T1 ────────────────────────────────────────────────────────────────────────

def characterize_t1(qubit: str = "Q0", *,
                    delay_max: float = 200e-6,
                    n_pts: int = 64,
                    repeat: int = 1) -> object:
    """Measure T1 decay from |1⟩ with delays up to ``delay_max``."""
    delays = np.linspace(0.0, delay_max, n_pts)
    rcp    = Recipe("t1", signal="IQ")
    rcp["qubit"]  = qubit
    rcp["delay"]  = delays
    rcp["repeat"] = repeat
    return s.submit(rcp.export())


def analyze_t1(task) -> dict:
    """
    Fit 1/e crossing of T1 decay. Returns {'T1': s}.

    Ctoolbox T1 data shape: (repeat, n_delay, n_qubits) — averages over repeat.
    """
    result  = task.result()
    sig     = result["meta"]["other"]["signal"]
    data    = result["data"][sig]          # (repeat, n_delay, 1)
    delays  = result["meta"]["axis"]["delay"]["def"]
    mag     = np.abs(data[:, :, 0].mean(axis=0))     # average over repeats

    thresh  = mag[0] / np.e
    below   = np.where(mag < thresh)[0]
    T1      = float(delays[below[0]]) if len(below) else float(delays[-1] / 3)
    return {"T1": T1}


# ── Ramsey ────────────────────────────────────────────────────────────────────

def characterize_ramsey(qubit: str = "Q0", *,
                         delay_max: float = 50e-6,
                         detuning: float = 0.0,
                         n_pts: int = 64) -> object:
    """
    Ramsey fringe measurement.

    Parameters
    ----------
    detuning : intentional off-resonance frequency (Hz). 0 → auto (≈3 fringes).
    """
    delays = np.linspace(0.0, delay_max, n_pts)
    rcp    = Recipe("ramsey", signal="IQ")
    rcp["qubit"]    = qubit
    rcp["delay"]    = delays
    rcp["detuning"] = detuning
    return s.submit(rcp.export())


def analyze_ramsey(task) -> dict:
    """
    Estimate T2* from Ramsey Gaussian decay envelope.

    Uses Ctoolbox formula: signal ∝ exp(−t/(2T₁) − t²/T₂ᵣ²)·cos(…).
    Approximates T2r from the oscillation envelope decay.

    Returns {'T2star': s, 'T2r': s}.
    """
    result = task.result()
    sig    = result["meta"]["other"]["signal"]
    iq     = result["data"][sig][:, 0]
    delays = result["meta"]["axis"]["delay"]["def"]
    mag    = np.abs(iq)

    env    = _smooth_envelope(mag)
    thresh = env[0] / np.e
    below  = np.where(env < thresh)[0]
    T2r    = float(delays[below[0]]) if len(below) else float(delays[-1] / 3)
    return {"T2star": T2r, "T2r": T2r}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _first_peak(arr: np.ndarray, min_idx: int = 3) -> int | None:
    diff  = np.diff(arr)
    signs = np.sign(diff)
    for i in range(len(signs) - 1):
        if signs[i] > 0 and signs[i + 1] <= 0 and i + 1 >= min_idx:
            return i + 1
    return None


def _smooth_envelope(arr: np.ndarray, window: int = 5) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")
