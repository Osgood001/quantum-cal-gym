"""
mock_quark — Drop-in stub for the proprietary ``quark`` SDK.

Mirrors the API patterns used in the lab notebooks:
    from quark.app import Recipe, s, get_data_by_rid

This module connects ``quark``-style experiment code to the
``SuperconductingQubit`` simulator so notebook experiments can be
tested offline without lab hardware.

Usage
-----
    from quantum_cal_gym.mock_quark import install
    install()   # monkey-patches sys.modules so ``from quark.app import ...`` works

Or use it standalone:
    from quantum_cal_gym.mock_quark import s, Recipe
"""
from __future__ import annotations

import sys
import types
import numpy as np
from typing import Any

from .qubit_sim import SuperconductingQubit

# ── Global qubit instance (shared across all Recipe calls) ────────────────────
_qubit: SuperconductingQubit | None = None
_config_store: dict[str, Any] = {}
_data_store: dict[int, Any] = {}
_rid_counter = 0


def set_qubit(qubit: SuperconductingQubit):
    """Attach a simulator instance to the mock backend."""
    global _qubit
    _qubit = qubit


def reset(seed=None):
    """Create a fresh random qubit and reset all stores."""
    global _qubit, _config_store, _data_store, _rid_counter
    _qubit         = SuperconductingQubit(seed=seed)
    _config_store  = {}
    _data_store    = {}
    _rid_counter   = 0
    return _qubit


# ── s object: mirrors quark.app.s ────────────────────────────────────────────

class _SessionProxy:
    """
    Mimics ``s`` from ``quark.app``.

    Supports::
        s.login()
        s.query('Q0.R.frequency')
        s.update('Q0.R.frequency', 5.1e9)
    """

    def login(self, *args, **kwargs):
        if _qubit is None:
            reset()
        print("[mock_quark] Logged in (simulator mode).")

    def query(self, key: str) -> Any:
        return _config_store.get(key, None)

    def update(self, key: str, value: Any):
        _config_store[key] = value

    def __repr__(self):
        return f"<mock quark session, {len(_config_store)} config keys>"


s = _SessionProxy()


# ── Recipe ────────────────────────────────────────────────────────────────────

class Recipe:
    """
    Simplified Recipe that maps circuit functions to simulator calls.

    Supported experiment names (matched by prefix):
      S21, spectrum, Spectrum, PowerRabi, TimeRabi, T1, T2, Ramsey
    """

    def __init__(self, name: str, signal: str = "S21"):
        self._name    = name
        self._signal  = signal
        self._params: dict[str, Any] = {}
        self.circuit  = None

    def __setitem__(self, key: str, value: Any):
        self._params[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._params.get(key)

    def run(self, n_shots: int = 1) -> "RecipeResult":
        """Execute the experiment on the simulator and return results."""
        if _qubit is None:
            raise RuntimeError("No qubit attached. Call mock_quark.reset() first.")
        return _run_recipe(self)

    def step(self, idx: int = 0):
        """Return a dummy command dict (for preview/waveform inspection)."""
        return {"main": {}, "step": idx}

    def __repr__(self):
        return f"<Recipe '{self._name}' params={list(self._params.keys())}>"


class RecipeResult:
    def __init__(self, name: str, x: np.ndarray, y: np.ndarray, rid: int):
        self.name  = name
        self.x     = x      # sweep axis
        self.y     = y      # complex IQ result
        self.rid   = rid    # run id for data retrieval

    def __repr__(self):
        return f"<RecipeResult '{self.name}' rid={self.rid} x={self.x.shape}>"


def _run_recipe(rcp: Recipe) -> RecipeResult:
    global _rid_counter
    q    = _qubit
    name = rcp._name.lower()
    p    = rcp._params

    # ── S21 ──────────────────────────────────────────────────────────────
    if name == 's21':
        freqs  = np.asarray(p.get('freq', np.linspace(q.f_r - 100e6, q.f_r + 100e6, 64)))
        result = q.s21(freqs)
        x      = freqs

    # ── Qubit spectrum ────────────────────────────────────────────────────
    elif 'spectrum' in name:
        freqs  = np.asarray(p.get('freq', np.linspace(q.f_q - 50e6, q.f_q + 50e6, 64)))
        amp    = float(p.get('amp', 0.5))
        result = q.spectrum(freqs, amp=amp)
        x      = freqs

    # ── PowerRabi ─────────────────────────────────────────────────────────
    elif 'powerrabi' in name or name == 'power_rabi':
        amps   = np.asarray(p.get('amp', np.linspace(0, 1.0, 64)))
        result = q.power_rabi(amps)
        x      = amps

    # ── TimeRabi ──────────────────────────────────────────────────────────
    elif 'timerabi' in name or name == 'time_rabi':
        times  = np.asarray(p.get('width', np.linspace(0, 300e-9, 64)))
        result = q.time_rabi(times)
        x      = times

    # ── T1 ────────────────────────────────────────────────────────────────
    elif name == 't1':
        times  = np.asarray(p.get('delay', np.linspace(0, 200e-6, 64)))
        result = q.t1_decay(times)
        x      = times

    # ── Ramsey ────────────────────────────────────────────────────────────
    elif 'ramsey' in name:
        times    = np.asarray(p.get('delay', np.linspace(0, 50e-6, 64)))
        detuning = float(p.get('detuning', 0.0))
        result   = q.ramsey(times, detuning=detuning)
        x        = times

    else:
        x      = np.linspace(0, 1, 64)
        result = np.zeros(64, dtype=complex)

    _rid_counter += 1
    rid = _rid_counter
    _data_store[rid] = {'x': x, 'y': result, 'config': dict(rcp._params)}
    return RecipeResult(name=rcp._name, x=x, y=result, rid=rid)


# ── Data retrieval stubs ──────────────────────────────────────────────────────

def get_data_by_rid(rid: int) -> dict | None:
    return _data_store.get(rid, None)


def get_config_by_rid(rid: int) -> dict:
    entry = _data_store.get(rid, {})
    return entry.get('config', {})


def preview(cmds, **kwargs):
    """Stub for waveform preview — returns empty dict."""
    return {}


def rollback(rid: int):
    """Stub for config rollback."""
    print(f"[mock_quark] rollback to rid={rid} (no-op in simulator mode)")


def lookup(key: str):
    return s.query(key)


# ── sys.modules injection ─────────────────────────────────────────────────────

def install():
    """
    Inject mock modules into sys.modules so that::

        from quark.app import Recipe, s, get_data_by_rid

    works in existing notebook code without modification.
    """
    quark_mod = types.ModuleType("quark")
    quark_app = types.ModuleType("quark.app")

    quark_app.s                 = s
    quark_app.Recipe            = Recipe
    quark_app.get_data_by_rid   = get_data_by_rid
    quark_app.get_config_by_rid = get_config_by_rid
    quark_app.preview           = preview
    quark_app.rollback          = rollback
    quark_app.lookup            = lookup

    quark_mod.app = quark_app
    sys.modules["quark"]     = quark_mod
    sys.modules["quark.app"] = quark_app
    print("[mock_quark] quark.app injected into sys.modules.")
