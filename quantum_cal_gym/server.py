"""
server.py — FastAPI wrapper around QubitCalibration-v0.

Keeps the qubit simulator in server memory.
True parameters are NEVER included in any HTTP response.

Start:
    uvicorn quantum_cal_gym.server:app --port 8765 --reload

Or:
    python -m quantum_cal_gym.server
"""
from __future__ import annotations

import random
import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from quantum_cal_gym.logger import EpisodeLogger

warnings.filterwarnings("ignore")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="QubitCal Server",
    description="Superconducting qubit calibration experiment API. "
                "True parameters are hidden — discover them through experiments.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

BUDGET_MAX = 30
PLOT_DIR   = Path(__file__).parent.parent / "runs" / "server_sessions"

# ── Qubit loaded once at startup — represents the sample in the fridge ────────

from quantum_cal_gym.qubit_sim import TransmonSim as _TransmonSim
_seed  = random.randint(0, 2**31 - 1)
_qubit = _TransmonSim(seed=_seed)
print(f"[server] Qubit loaded  (seed={_seed}). Parameters hidden.")

# session_id → {budget, plots, logger}
_sessions: dict[str, dict[str, Any]] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(sid: str) -> dict:
    if sid not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{sid}' not found.")
    return _sessions[sid]


def _record_step(sid: str, step: int, exp: str,
                 sig, x_phys, x_label: str, y_label: str) -> str:
    """Record one experiment via EpisodeLogger and return the plot filename."""
    obs = {
        "signal_re": np.asarray(sig).real,
        "signal_im": np.asarray(sig).imag,
        "x_axis":    np.linspace(0.0, 1.0, len(sig)),
    }
    info = {
        "exp_type":     exp,
        "x_phys":       x_phys,
        "x_label":      x_label,
        "y_label":      y_label,
        "n_calibrated": 0,
        "calibrated":   [],
        "estimates":    {},
        "true_params":  _qubit.true_params,  # server-side only
    }
    logger = _sessions[sid]["logger"]
    logger.record(step, obs, reward=0.0, info=info)
    logger.save()
    return f"step_{step:03d}_{exp}.png"


def _tof(arr) -> list[float]:
    return [round(float(v), 8) for v in np.asarray(arr)]


# ── Session endpoints ─────────────────────────────────────────────────────────

@app.post("/session", summary="Create a new calibration session")
def new_session():
    """
    Open a new calibration session on the loaded qubit.
    Returns a `session_id` you must pass to all subsequent calls.
    The true parameters are hidden inside the server.
    Restart the server to load a different qubit.
    """
    sid = uuid.uuid4().hex[:8]
    log = EpisodeLogger(str(PLOT_DIR / sid), verbose=True)
    _sessions[sid] = {"budget": 0, "plots": [], "logger": log}
    return {
        "session_id":   sid,
        "budget_total": BUDGET_MAX,
        "message":      "Session created. Parameters hidden. Good luck.",
        "experiments":  ["s21", "spectrum", "power_rabi", "time_rabi", "t1", "ramsey"],
    }


@app.get("/session/{sid}", summary="Session status")
def session_status(sid: str):
    s = _get(sid)
    return {
        "session_id":     sid,
        "budget_used":    s["budget"],
        "budget_remaining": BUDGET_MAX - s["budget"],
        "plots":          s["plots"],
    }


# ── Experiment endpoint ───────────────────────────────────────────────────────

class RunRequest(BaseModel):
    experiment: str
    center_hz:  float | None = None   # s21, spectrum
    span_hz:    float | None = None   # s21, spectrum
    amp_max:    float | None = None   # power_rabi
    width_max_s: float | None = None  # time_rabi
    delay_max_s: float | None = None  # t1, ramsey
    detuning_hz: float | None = None  # ramsey


@app.post("/session/{sid}/run", summary="Run one experiment")
def run_experiment(sid: str, req: RunRequest):
    """
    Run a single experiment on the hidden qubit.

    Returns signal data (x_values, magnitude or population) and a hint.
    Also saves a PNG plot — fetch it via GET /session/{sid}/plots/{filename}.

    **Never returns the true qubit parameters.**
    """
    s = _get(sid)
    if s["budget"] >= BUDGET_MAX:
        raise HTTPException(status_code=429,
                            detail=f"Budget exhausted ({BUDGET_MAX} experiments used). "
                                   "Submit your estimates now.")

    q    = _qubit
    exp  = req.experiment.lower()
    step = s["budget"] + 1
    n    = 64

    if exp == "s21":
        center = req.center_hz or 6.5e9
        span   = req.span_hz   or 500e6
        freqs  = np.linspace(center - span / 2, center + span / 2, n)
        sig    = q.s21(freqs)
        mag    = np.abs(sig)
        mag_db = 20.0 * np.log10(np.maximum(mag, 1e-10))
        fname  = _record_step(sid, step, "s21",
                               sig, freqs / 1e9, "Probe Frequency (GHz)", "|S₂₁| (dB)")
        result = {
            "experiment":   "s21",
            "x_label":      "Probe Frequency (GHz)",
            "x_values":     _tof(freqs / 1e9),
            "magnitude_dB": _tof(mag_db),
            "hint": "Resonator dip = minimum in magnitude_dB. "
                    "Narrow the span to improve frequency resolution.",
        }

    elif exp == "spectrum":
        center = req.center_hz or 5.5e9
        span   = req.span_hz   or 200e6
        freqs  = np.linspace(center - span / 2, center + span / 2, n)
        sig    = q.spectrum(freqs, amp=0.7)
        mag    = np.abs(sig)
        fname  = _record_step(sid, step, "spectrum",
                               sig, freqs / 1e9, "Drive Frequency (GHz)", "Magnitude")
        result = {
            "experiment": "spectrum",
            "x_label":    "Drive Frequency (GHz)",
            "x_values":   _tof(freqs / 1e9),
            "magnitude":  _tof(mag),
            "hint": "Qubit absorption dip. Typically 0.5–1.5 GHz below f_r. "
                    "Zoom in — linewidth can be very narrow.",
        }

    elif exp == "power_rabi":
        amp_max = req.amp_max or 1.2
        amps    = np.linspace(0.0, amp_max, n)
        sig     = q.power_rabi(amps)
        mag     = np.abs(sig)
        fname   = _record_step(sid, step, "power_rabi",
                                sig, amps, "Drive Amplitude (arb.)", "Population |1⟩")
        result  = {
            "experiment": "power_rabi",
            "x_label":    "Drive Amplitude (arb. u.)",
            "x_values":   _tof(amps),
            "population": _tof(mag),
            "hint": "First peak = amp_pi. Pattern: 0→1→0→1. "
                    "Typical amp_pi: 0.3–0.9. Increase amp_max if no peak visible.",
        }

    elif exp == "time_rabi":
        wmax   = req.width_max_s or 400e-9
        widths = np.linspace(0.0, wmax, n)
        sig    = q.time_rabi(widths)
        mag    = np.abs(sig)
        fname  = _record_step(sid, step, "time_rabi",
                               sig, widths * 1e9, "Pulse Duration (ns)", "Population |1⟩")
        result = {
            "experiment": "time_rabi",
            "x_label":    "Pulse Duration (ns)",
            "x_values":   _tof(widths * 1e9),
            "population": _tof(mag),
            "hint": "First peak = t_pi (ns). Typical: 20–200 ns. "
                    "Envelope decays slowly with T1.",
        }

    elif exp == "t1":
        dmax   = req.delay_max_s or 400e-6
        delays = np.linspace(0.0, dmax, n)
        sig    = q.t1_decay(delays)
        mag    = np.abs(sig)
        fname  = _record_step(sid, step, "t1",
                               sig, delays * 1e6, "Delay (µs)", "Population |1⟩")
        result = {
            "experiment": "t1",
            "x_label":    "Delay (µs)",
            "x_values":   _tof(delays * 1e6),
            "population": _tof(mag),
            "hint": "Exponential decay P(t)=exp(-t/T1). "
                    "T1 = delay where population ≈ 0.368. "
                    "Set delay_max ~ 3×T1. Typical T1: 10–200 µs.",
        }

    elif exp == "ramsey":
        dmax     = req.delay_max_s or 60e-6
        detuning = req.detuning_hz or 2e6
        delays   = np.linspace(0.0, dmax, n)
        sig      = q.ramsey(delays, detuning=detuning)
        mag      = np.abs(sig)
        fname    = _record_step(sid, step, "ramsey",
                                 sig, delays * 1e6, "Free Evolution (µs)", "Population |1⟩")
        result   = {
            "experiment":  "ramsey",
            "x_label":     "Free Evolution Time (µs)",
            "x_values":    _tof(delays * 1e6),
            "population":  _tof(mag),
            "detuning_hz": float(detuning),
            "hint": "Signal = A·exp(-t/2T1 - t²/T2r²)·cos(2π·Δf·t) + B. "
                    "Gaussian envelope → T2*. "
                    "Aim for ~5 fringes: set detuning_hz = 5 / delay_max_s.",
        }

    else:
        raise HTTPException(status_code=400,
                            detail=f"Unknown experiment '{exp}'. "
                                   "Choose: s21 | spectrum | power_rabi | time_rabi | t1 | ramsey")

    s["budget"] += 1
    if fname:
        s["plots"].append(fname)
        result["plot_url"] = f"/session/{sid}/plots/{fname}"

    result["budget_remaining"] = BUDGET_MAX - s["budget"]
    return result


# ── Submit & score ────────────────────────────────────────────────────────────

class Estimates(BaseModel):
    f_q:    float
    f_r:    float
    T1:     float
    T2star: float
    amp_pi: float
    t_pi:   float


@app.post("/session/{sid}/submit", summary="Submit parameter estimates for scoring")
def submit(sid: str, est: Estimates):
    """
    Submit your final parameter estimates.
    Returns pass/fail for each parameter (5% relative error threshold).
    Reveals the true values only after submission.
    """
    s  = _get(sid)
    tp = _qubit.true_params

    estimates = est.model_dump()
    report    = {}
    for key, true_val in tp.items():
        ev      = estimates.get(key, None)
        if ev is None:
            report[key] = {"status": "MISSING", "true_value": true_val}
            continue
        rel_err = abs(ev - true_val) / max(abs(true_val), 1e-30)
        report[key] = {
            "status":      "PASS" if rel_err < 0.05 else "FAIL",
            "your_estimate": ev,
            "true_value":  true_val,
            "relative_error_pct": round(rel_err * 100, 2),
        }

    n_pass = sum(1 for v in report.values() if v.get("status") == "PASS")
    return {
        "calibrated":    n_pass,
        "total":         len(tp),
        "budget_used":   s["budget"],
        "results":       report,
    }


# ── Plot serving ──────────────────────────────────────────────────────────────

@app.get("/session/{sid}/plots/{filename}", summary="Fetch an experiment plot (PNG)")
def get_plot(sid: str, filename: str):
    _get(sid)   # 404 if session missing
    path = PLOT_DIR / sid / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Plot not found.")
    return Response(content=path.read_bytes(), media_type="image/png")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("quantum_cal_gym.server:app", host="0.0.0.0", port=8765, reload=False)
