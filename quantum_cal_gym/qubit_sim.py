"""
Superconducting transmon qubit physics simulator.

Models a single qubit + dispersive readout resonator system.
All parameters are hidden from the agent and must be discovered
through experiments.
"""
from __future__ import annotations

import numpy as np

# ── Physical parameter ranges ─────────────────────────────────────────────────
# Typical single-junction transmon values

F_Q_RANGE      = (4.0e9,  6.0e9)   # qubit frequency, Hz
F_R_OFFSET     = (0.5e9,  2.0e9)   # resonator offset above f_q, Hz
CHI_RANGE      = (-10e6,  -1e6)    # dispersive shift, Hz
T1_RANGE       = (10e-6, 200e-6)   # energy relaxation time, s
T2STAR_RANGE   = (5e-6,  100e-6)   # dephasing time, s
KAPPA_RANGE    = (1e6,    10e6)    # resonator linewidth, Hz
AMP_PI_RANGE   = (0.3,    0.9)     # pi-pulse amplitude (normalised)
T_PI_RANGE     = (20e-9, 200e-9)   # pi-pulse duration, s
NOISE_FLOOR    = 0.025             # measurement noise std (IQ units)

# Sweep extent constants used by cal_env
FREQ_LO        = 4.0e9             # Hz
FREQ_HI        = 8.0e9             # Hz
FREQ_SPAN_MIN  = 10e6              # Hz
FREQ_SPAN_MAX  = 2000e6            # Hz
TIME_MIN       = 1e-9              # s
TIME_MAX       = 300e-6            # s


class SuperconductingQubit:
    """
    Single transmon qubit with dispersive readout.

    Parameters are drawn randomly at construction and kept fixed for
    the lifetime of the object (one calibration episode).

    All ``simulate_*`` methods return complex numpy arrays (I + jQ)
    of length ``n_pts``.
    """

    def __init__(self, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.f_q     = float(rng.uniform(*F_Q_RANGE))
        self.f_r     = float(self.f_q + rng.uniform(*F_R_OFFSET))
        self.chi     = float(rng.uniform(*CHI_RANGE))
        self.T1      = float(rng.uniform(*T1_RANGE))
        self.T2star  = float(rng.uniform(*T2STAR_RANGE))
        self.kappa   = float(rng.uniform(*KAPPA_RANGE))
        self.amp_pi  = float(rng.uniform(*AMP_PI_RANGE))
        self.t_pi    = float(rng.uniform(*T_PI_RANGE))
        self._rng    = rng

    # ── Experiments ───────────────────────────────────────────────────────

    def s21(self, freqs: np.ndarray) -> np.ndarray:
        """
        Resonator transmission (S21) vs probe frequency.

        Yields a complex Lorentzian dip centred at f_r.
        """
        delta  = freqs - self.f_r
        # Transmission: S21 = 1 - (kappa/2) / (i*delta + kappa/2)
        signal = 1.0 - (self.kappa / 2.0) / (1j * delta + self.kappa / 2.0)
        return signal + self._iq_noise(freqs.shape)

    def spectrum(self, freqs: np.ndarray, amp: float = 0.6) -> np.ndarray:
        """
        Qubit spectroscopy: transmission dip when drive hits f_q.

        ``amp`` controls the depth of the qubit absorption feature.
        """
        gamma  = 1.0 / (np.pi * self.T2star)          # FWHM linewidth
        dip    = amp * (gamma / 2) ** 2 / ((freqs - self.f_q) ** 2 + (gamma / 2) ** 2)
        signal = (1.0 - dip) + 0j
        return signal + self._iq_noise(freqs.shape)

    def power_rabi(self, amps: np.ndarray) -> np.ndarray:
        """
        Qubit population vs drive amplitude (PowerRabi).

        Returns |excited-state probability| as a real-valued complex array.
        """
        P = 0.5 * (1.0 - np.cos(np.pi * amps / self.amp_pi))
        P = np.clip(P + self._rng.normal(0, NOISE_FLOOR, amps.shape), 0, 1)
        return P.astype(complex)

    def time_rabi(self, times: np.ndarray) -> np.ndarray:
        """
        Qubit population vs drive duration (TimeRabi).

        Includes a T1 decay envelope.
        """
        P  = 0.5 * (1.0 - np.cos(np.pi * times / self.t_pi))
        P *= np.exp(-times / (2.0 * self.T1))   # amplitude envelope
        P  = np.clip(P + self._rng.normal(0, NOISE_FLOOR, times.shape), 0, 1)
        return P.astype(complex)

    def t1_decay(self, times: np.ndarray) -> np.ndarray:
        """
        Excited-state population vs free-evolution delay (T1 measurement).
        """
        P = np.exp(-times / self.T1)
        P = np.clip(P + self._rng.normal(0, NOISE_FLOOR, times.shape), 0, 1)
        return P.astype(complex)

    def ramsey(self, times: np.ndarray, detuning: float | None = None) -> np.ndarray:
        """
        Ramsey fringes vs free-evolution time.

        ``detuning`` is the intentional off-resonance frequency (Hz).
        If None, a detuning that yields ~3 fringes over ``times`` is chosen.
        """
        if detuning is None or detuning == 0.0:
            t_span   = float(times[-1] - times[0]) if len(times) > 1 else 1e-6
            detuning = 3.0 / t_span
        P = 0.5 * (1.0 + np.cos(2 * np.pi * detuning * times) * np.exp(-times / self.T2star))
        P = np.clip(P + self._rng.normal(0, NOISE_FLOOR, times.shape), 0, 1)
        return P.astype(complex)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def true_params(self) -> dict:
        """Ground-truth parameters (not visible to the agent)."""
        return {
            'f_q':    self.f_q,
            'f_r':    self.f_r,
            'T1':     self.T1,
            'T2star': self.T2star,
            'amp_pi': self.amp_pi,
            't_pi':   self.t_pi,
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    def _iq_noise(self, shape) -> np.ndarray:
        return (self._rng.normal(0, NOISE_FLOOR, shape) +
                1j * self._rng.normal(0, NOISE_FLOOR, shape))
