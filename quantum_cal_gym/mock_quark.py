"""
mock_quark — Drop-in stub for the proprietary ``quark`` SDK.

Mirrors the API patterns used in the lab notebooks and Ctoolbox:
    from quark.app import Recipe, s, get_data_by_rid

Result format is Ctoolbox-compatible:
    result['data'][signal]           — IQ data array
    result['meta']['axis'][ax_name]  — sweep axis dict
    result['meta']['other']          — qubit list, signal name, …

Usage
-----
    from quantum_cal_gym.mock_quark import install
    install()   # patches sys.modules so lab notebook code works offline

Or standalone:
    from quantum_cal_gym.mock_quark import s, Recipe
"""
from __future__ import annotations

import sys
import time
import types
import numpy as np
from typing import Any

from .qubit_sim import TransmonSim

# ── Module-level state ────────────────────────────────────────────────────────

_qubit: TransmonSim | None = None
_config_store: dict[str, Any] = {}
_data_store:   dict[int, Any] = {}
_rid_counter = 0


def set_qubit(qubit: TransmonSim):
    """Attach a simulator instance to the mock backend."""
    global _qubit
    _qubit = qubit


def reset(seed=None):
    """Create a fresh random qubit and reset all stores."""
    global _qubit, _config_store, _data_store, _rid_counter
    _qubit        = TransmonSim(seed=seed)
    _config_store = {}
    _data_store   = {}
    _rid_counter  = 0
    return _qubit


# ── TaskResult — mirrors quark's async task handle ───────────────────────────

class TaskResult:
    """
    Returned by ``s.submit()``.  Mirrors the quark SDK's task object:

        thistask = s.submit(rcp.export())
        data = thistask.result()
        rid  = thistask.rid
    """

    def __init__(self, rid: int, result_dict: dict):
        self.rid      = rid
        self._result  = result_dict

    def result(self) -> dict:
        """Return the Ctoolbox-format result dict."""
        return self._result

    def __repr__(self):
        return f"<TaskResult rid={self.rid}>"


# ── s object: mirrors quark.app.s ────────────────────────────────────────────

class _SessionProxy:
    """
    Mimics ``s`` from ``quark.app``.

    Supports::
        s.login()
        s.query('Q0.R.frequency')
        s.update('Q0.R.frequency', 5.1e9)
        s.submit(rcp.export())   → TaskResult
    """

    def login(self, *args, **kwargs):
        if _qubit is None:
            reset()
        print("[mock_quark] Logged in (simulator mode).")

    def query(self, key: str) -> Any:
        return _config_store.get(key, None)

    def update(self, key: str, value: Any):
        _config_store[key] = value

    def submit(self, exported: dict) -> TaskResult:
        """
        Execute an exported recipe dict and return a TaskResult.

        ``exported`` should be the dict returned by ``Recipe.export()``.
        """
        if _qubit is None:
            reset()
        return _run_exported(exported)

    def __repr__(self):
        return f"<mock quark session, {len(_config_store)} config keys>"


s = _SessionProxy()


# ── Recipe ────────────────────────────────────────────────────────────────────

class Recipe:
    """
    Simplified Recipe that maps named experiments to simulator calls.

    Supported experiment names (matched by substring):
        s21, spectrum, power_rabi, time_rabi, t1, ramsey
    """

    def __init__(self, name: str, signal: str = "IQ"):
        self._name   = name
        self._signal = signal
        self._params: dict[str, Any] = {}
        self.circuit = None

    def __setitem__(self, key: str, value: Any):
        self._params[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._params.get(key)

    def export(self) -> dict:
        """Return a serialisable dict passed to ``s.submit()``."""
        return {"name": self._name, "signal": self._signal,
                "params": dict(self._params)}

    def run(self, n_shots: int = 1) -> TaskResult:
        """Convenience: run immediately without s.submit."""
        if _qubit is None:
            reset()
        return _run_exported(self.export())

    def step(self, idx: int = 0) -> dict:
        return {"main": {}, "step": idx}

    def __repr__(self):
        return f"<Recipe '{self._name}' params={list(self._params.keys())}>"


# ── Experiment dispatcher ─────────────────────────────────────────────────────

def _run_exported(exported: dict) -> TaskResult:
    """
    Dispatch an exported recipe dict to the simulator and return a
    Ctoolbox-compatible TaskResult.

    Result format
    -------------
    result['data'][signal_name]   — np.ndarray, shape (n_pts, 1) complex
    result['meta']['axis']        — {ax_key: {'def': array}}
    result['meta']['other']       — {'signal': name, 'qubits': [q_name], ...}
    """
    global _rid_counter

    q      = _qubit
    name   = exported["name"].lower()
    sig    = exported.get("signal", "IQ")
    p      = exported.get("params", {})
    qubit  = p.get("qubit", "Q0")

    # ── S21 ──────────────────────────────────────────────────────────────
    if name == "s21":
        freqs  = np.asarray(p.get("freq", np.linspace(q.f_r - 200e6, q.f_r + 200e6, 64)))
        iq     = q.s21(freqs)
        data   = {sig: iq[:, np.newaxis]}                # (n_pts, n_qubits)
        axis   = {f"${qubit}.Measure.frequency": {"def": freqs}}
        other  = {"signal": sig, "qubits": [qubit]}

    # ── Spectrum ──────────────────────────────────────────────────────────
    elif "spectrum" in name:
        freqs  = np.asarray(p.get("freq", np.linspace(q.f_q - 100e6, q.f_q + 100e6, 64)))
        amp    = float(p.get("amp", 0.5))
        iq     = q.spectrum(freqs, amp=amp)
        data   = {sig: iq[:, np.newaxis]}
        axis   = {f"${qubit}.R.frequency": {"def": freqs}}
        other  = {"signal": sig, "qubits": [qubit]}

    # ── PowerRabi ─────────────────────────────────────────────────────────
    elif "powerrabi" in name or name == "power_rabi":
        amps   = np.asarray(p.get("amp", np.linspace(0.0, 1.2 * q.amp_pi, 64)))
        iq     = q.power_rabi(amps)
        data   = {sig: iq[:, np.newaxis]}
        axis   = {f"${qubit}.R.amp": {"def": amps}}
        other  = {"signal": sig, "qubits": [qubit]}

    # ── TimeRabi ──────────────────────────────────────────────────────────
    elif "timerabi" in name or name == "time_rabi":
        widths = np.asarray(p.get("width", np.linspace(0.0, 3.0 * q.t_pi, 64)))
        iq     = q.time_rabi(widths)
        data   = {sig: iq[:, np.newaxis]}
        axis   = {f"${qubit}.R.width": {"def": widths}}
        other  = {"signal": sig, "qubits": [qubit]}

    # ── T1 ────────────────────────────────────────────────────────────────
    elif name == "t1":
        delays = np.asarray(p.get("delay", np.linspace(0.0, 3.0 * q.T1, 64)))
        repeat = int(p.get("repeat", 1))
        iq     = q.t1_decay(delays)
        # T1 shape: (repeat, n_delay, n_qubits)
        data   = {sig: np.tile(iq[:, np.newaxis], (1, 1))[np.newaxis, :, :].repeat(repeat, axis=0)}
        axis   = {"delay": {"def": delays}}
        other  = {"signal": sig, "qubits": [qubit], "echonum": 0}

    # ── Ramsey ────────────────────────────────────────────────────────────
    elif "ramsey" in name:
        delays   = np.asarray(p.get("delay", np.linspace(0.0, 5.0 * q.T2r, 64)))
        detuning = float(p.get("detuning", 0.0))
        iq       = q.ramsey(delays, detuning=detuning)
        data     = {sig: iq[:, np.newaxis]}
        axis     = {"delay": {"def": delays}}
        other    = {"signal": sig, "qubits": [qubit], "echonum": 0}

    else:
        n = 64
        x  = np.linspace(0, 1, n)
        iq = np.zeros(n, dtype=complex)
        data  = {sig: iq[:, np.newaxis]}
        axis  = {"x": {"def": x}}
        other = {"signal": sig, "qubits": [qubit]}

    _rid_counter += 1
    result = {
        "data": data,
        "meta": {
            "axis":  axis,
            "other": other,
        },
    }
    _data_store[_rid_counter] = result
    return TaskResult(rid=_rid_counter, result_dict=result)


# ── Data retrieval ────────────────────────────────────────────────────────────

def get_data_by_rid(rid: int) -> dict | None:
    return _data_store.get(rid, None)


def get_config_by_rid(rid: int) -> dict:
    return {}


def preview(cmds, **kwargs) -> dict:
    """Stub — waveform preview not available in simulator mode."""
    return {}


def rollback(rid: int):
    print(f"[mock_quark] rollback to rid={rid} (no-op in simulator mode)")


def lookup(key: str):
    return s.query(key)


# ── sys.modules injection ─────────────────────────────────────────────────────

def install():
    """
    Inject mock modules into ``sys.modules`` so that::

        from quark.app import Recipe, s, get_data_by_rid

    works in existing notebook code without modification.
    """
    quark_mod = types.ModuleType("quark")
    quark_app = types.ModuleType("quark.app")

    quark_app.s                 = s
    quark_app.Recipe            = Recipe
    quark_app.TaskResult        = TaskResult
    quark_app.get_data_by_rid   = get_data_by_rid
    quark_app.get_config_by_rid = get_config_by_rid
    quark_app.preview           = preview
    quark_app.rollback          = rollback
    quark_app.lookup            = lookup

    quark_mod.app = quark_app
    sys.modules["quark"]     = quark_mod
    sys.modules["quark.app"] = quark_app
    print("[mock_quark] quark.app injected into sys.modules.")
