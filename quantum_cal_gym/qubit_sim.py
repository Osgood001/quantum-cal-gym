"""
Superconducting transmon qubit physics simulator.

Backend
-------
* scqubits  — Josephson circuit parametrization (EJ, EC → f_q)
* QuTiP     — Lindblad master equation for T1 / Rabi dynamics
* Ctoolbox  — aligned analytical models for Ramsey (Gaussian decay)
              and S21 (linear_resonator)

References
----------
* Ctoolbox Ramsey_func:
    A * exp(-t/2/T1 - t**2/T2r**2) * cos(2*pi*Delta*t + phi) + B
* Ctoolbox linear_resonator_abs:
    S21 = amp * (1 - Q/Q_e / (1 + 2i*Q*(f-f0)/f0)) + offset
"""
from __future__ import annotations

import warnings
import numpy as np

import qutip as qt
import scqubits as scq

# Silence verbose scqubits progress bars / qutip deprecation noise
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Physical parameter ranges ─────────────────────────────────────────────────

EJ_RANGE      = (10.0,  22.0)      # GHz   — Josephson energy  → f_q ≈ 4–7 GHz
EC_RANGE      = (0.20,   0.35)     # GHz   — charging energy
F_R_OFFSET    = (0.4e9,  1.2e9)    # Hz    — resonator offset above f_q
Q_INT_RANGE   = (1e4,    1e5)      # internal quality factor
Q_EXT_RANGE   = (1e3,    1e4)      # external / coupling Q
T1_RANGE      = (10e-6, 200e-6)    # s     — energy relaxation
T2R_RANGE     = (50e-6, 300e-6)    # s     — Gaussian dephasing time (Ramsey)
AMP_PI_RANGE  = (0.3,    0.9)      # arb.  — pi-pulse amplitude
T_PI_RANGE    = (20e-9, 200e-9)    # s     — pi-pulse duration
NOISE_FLOOR   = 0.025              # IQ measurement noise (σ)

# Sweep-extent constants re-exported for use in cal_env
FREQ_LO       = 3.0e9             # Hz
FREQ_HI       = 10.0e9            # Hz
FREQ_SPAN_MIN = 10e6              # Hz
FREQ_SPAN_MAX = 2000e6            # Hz
TIME_MIN      = 1e-9              # s
TIME_MAX      = 300e-6            # s


class TransmonSim:
    """
    Single transmon qubit + dispersive readout resonator.

    Physical parameter workflow
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1.  Draw EJ, EC from realistic ranges.
    2.  Use ``scqubits.Transmon`` to compute the qubit transition
        frequency f_q = E_{01}/h — physically consistent with
        anharmonicity and charge dispersion.
    3.  T1, T2r drawn independently; T2* ≤ 2·T1 enforced.

    Experiment backends
    ~~~~~~~~~~~~~~~~~~~
    * ``t1_decay``  / ``time_rabi`` — QuTiP ``mesolve`` on a 2-level system.
    * ``s21``      — Ctoolbox ``linear_resonator`` complex S21.
    * ``spectrum`` — Lorentzian dip with T2*-derived linewidth.
    * ``power_rabi`` — sin² formula (exact for ideal resonant drive).
    * ``ramsey``   — Ctoolbox Gaussian-decay formula (1/f flux noise model).
    """

    def __init__(self, seed: int | None = None):
        rng = np.random.default_rng(seed)

        # ── scqubits: EJ/EC → f_q ────────────────────────────────────────
        EJ = float(rng.uniform(*EJ_RANGE))
        EC = float(rng.uniform(*EC_RANGE))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transmon = scq.Transmon(EJ=EJ, EC=EC, ng=0.0, ncut=31, truncated_dim=3)
            evals    = transmon.eigenvals(evals_count=3)   # GHz
        # f_q = E_01 / h  (scqubits already returns in GHz units)
        self.f_q = float((evals[1] - evals[0]) * 1e9)     # → Hz
        self._EJ, self._EC = EJ, EC

        # ── Resonator: Ctoolbox linear_resonator parametrisation ─────────
        self.f_r      = float(self.f_q + rng.uniform(*F_R_OFFSET))
        Q_int         = float(rng.uniform(*Q_INT_RANGE))
        Q_ext         = float(rng.uniform(*Q_EXT_RANGE))
        self.Q_int    = Q_int
        self.Q_ext    = Q_ext
        self.Q_e_real = Q_ext          # purely real coupling Q (no asymmetry)
        self.Q_e_imag = 0.0
        self.Q        = 1.0 / (1.0/Q_int + 1.0/Q_ext)    # loaded Q
        self.kappa    = self.f_r / self.Q                  # linewidth, Hz

        # ── Coherence ─────────────────────────────────────────────────────
        self.T1    = float(rng.uniform(*T1_RANGE))
        # Gaussian decay time for Ramsey (Ctoolbox T2r)
        self.T2r   = float(rng.uniform(*T2R_RANGE))
        # Effective T2* for spectral linewidth — enforced ≤ 2·T1
        T2star_max  = min(2.0 * self.T1, self.T2r)
        self.T2star = float(rng.uniform(5e-6, max(5.1e-6, T2star_max)))

        # ── Rabi ──────────────────────────────────────────────────────────
        self.amp_pi = float(rng.uniform(*AMP_PI_RANGE))
        self.t_pi   = float(rng.uniform(*T_PI_RANGE))

        self._rng = rng

    # ── Experiment methods ────────────────────────────────────────────────────

    def s21(self, freqs: np.ndarray) -> np.ndarray:
        """
        Resonator transmission — Ctoolbox ``linear_resonator`` model.

        S21 = (1 − Q/Q_e / (1 + 2i·Q·(f−f₀)/f₀))
        Returns complex array (I + jQ).
        """
        f0  = self.f_r
        Q   = self.Q
        Q_e = complex(self.Q_e_real, self.Q_e_imag)
        S21 = 1.0 - Q / Q_e / (1.0 + 2j * Q * (freqs - f0) / f0)
        return S21 + self._iq_noise(freqs.shape)

    def spectrum(self, freqs: np.ndarray, amp: float = 0.6) -> np.ndarray:
        """
        Qubit spectroscopy — Lorentzian dip with T2*-derived FWHM linewidth.
        """
        gamma  = 1.0 / (np.pi * self.T2star)    # FWHM, Hz
        dip    = amp * (gamma / 2)**2 / ((freqs - self.f_q)**2 + (gamma / 2)**2)
        signal = (1.0 - dip) + 0j
        return signal + self._iq_noise(freqs.shape)

    def power_rabi(self, amps: np.ndarray) -> np.ndarray:
        """
        Excited-state population vs drive amplitude.

        P(amp) = sin²(π·amp / (2·amp_π))  =  (1 − cos(π·amp/amp_π)) / 2
        """
        P = 0.5 * (1.0 - np.cos(np.pi * amps / self.amp_pi))
        P = np.clip(P + self._rng.normal(0, NOISE_FLOOR, amps.shape), 0.0, 1.0)
        return P.astype(complex)

    def time_rabi(self, times: np.ndarray) -> np.ndarray:
        """
        Qubit population vs drive duration — QuTiP Lindblad dynamics.

        Resonant drive Ω = π/t_π, with T1 (σ⁻ collapse) and pure
        dephasing (σz collapse) derived from T2*.
        """
        return self._qutip_rabi(times)

    def t1_decay(self, times: np.ndarray) -> np.ndarray:
        """
        Excited-state population vs free-evolution time — QuTiP mesolve.

        |ψ₀⟩ = |1⟩, collapse operator: √(1/T₁) · σ⁻
        """
        return self._qutip_t1(times)

    def ramsey(self, times: np.ndarray, detuning: float | None = None) -> np.ndarray:
        """
        Ramsey fringes — Ctoolbox Gaussian decay model.

        Signal = A · exp(−t/(2T₁) − t²/T₂ᵣ²) · cos(2π·Δf·t + φ) + B

        The Gaussian envelope (T₂ᵣ) models 1/f flux noise, matching the
        Ctoolbox ``Ramsey_func`` fit function.
        """
        if detuning is None or detuning == 0.0:
            t_span   = float(times[-1] - times[0]) if len(times) > 1 else 1e-6
            detuning = 3.0 / t_span
        A   = 0.5
        B   = 0.5
        phi = 0.0
        env = np.exp(-times / (2.0 * self.T1) - times**2 / self.T2r**2)
        P   = A * env * np.cos(2.0 * np.pi * detuning * times + phi) + B
        P   = np.clip(P + self._rng.normal(0, NOISE_FLOOR, times.shape), 0.0, 1.0)
        return P.astype(complex)

    # ── QuTiP backends ────────────────────────────────────────────────────────

    def _qutip_t1(self, times: np.ndarray) -> np.ndarray:
        """QuTiP Lindblad solution for T1 decay from |1⟩."""
        sm    = qt.destroy(2)
        H     = 0.0 * qt.sigmax()
        c_ops = [np.sqrt(1.0 / self.T1) * sm]
        psi0  = qt.basis(2, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = qt.mesolve(H, psi0, times, c_ops, e_ops=[sm.dag() * sm])
        P = np.array(res.expect[0], dtype=float)
        P = np.clip(P + self._rng.normal(0, NOISE_FLOOR, times.shape), 0.0, 1.0)
        return P.astype(complex)

    def _qutip_rabi(self, times: np.ndarray) -> np.ndarray:
        """QuTiP Lindblad: resonant Rabi from |0⟩ with T1 + pure dephasing."""
        sm        = qt.destroy(2)
        sx        = qt.sigmax()
        sz        = qt.sigmaz()
        Omega     = np.pi / self.t_pi             # calibrated to flip at t_pi
        H         = Omega / 2.0 * sx
        gamma_1   = 1.0 / self.T1
        gamma_phi = max(0.0, 1.0 / self.T2star - 0.5 / self.T1)
        c_ops     = [np.sqrt(gamma_1) * sm]
        if gamma_phi > 0.0:
            c_ops.append(np.sqrt(gamma_phi / 2.0) * sz)
        psi0 = qt.basis(2, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = qt.mesolve(H, psi0, times, c_ops, e_ops=[sm.dag() * sm])
        P = np.array(res.expect[0], dtype=float)
        P = np.clip(P + self._rng.normal(0, NOISE_FLOOR, times.shape), 0.0, 1.0)
        return P.astype(complex)

    # ── Properties ────────────────────────────────────────────────────────────

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

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _iq_noise(self, shape) -> np.ndarray:
        return (self._rng.normal(0, NOISE_FLOOR, shape) +
                1j * self._rng.normal(0, NOISE_FLOOR, shape))


# Backwards-compat alias so existing imports still work
SuperconductingQubit = TransmonSim
