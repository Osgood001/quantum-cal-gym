"""
logger.py — per-step signal plots and calibration progress tracking.

All output is headless (matplotlib Agg backend) and saved to files,
suitable for remote/CI use and visual debugging.

Saved outputs
─────────────
  <log_dir>/step_NNN_<exp>.png  — measured signal for each step
  <log_dir>/progress.png        — parameter estimates + reward over time
  <log_dir>/log.json            — structured step log
"""
from __future__ import annotations

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")                          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator

# ── Plot theme (matches the project's dark aesthetic) ─────────────────────────

_RC = {
    "figure.facecolor":  "#0f172a",
    "axes.facecolor":    "#1e293b",
    "axes.edgecolor":    "#334155",
    "axes.labelcolor":   "#94a3b8",
    "axes.titlecolor":   "#e2e8f0",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "axes.labelsize":     9,
    "axes.titlesize":    11,
    "grid.color":        "#334155",
    "grid.alpha":         0.5,
    "text.color":        "#e2e8f0",
    "legend.facecolor":  "#1e293b",
    "legend.edgecolor":  "#475569",
    "legend.fontsize":    8,
    "lines.linewidth":    1.6,
}

_C_SIGNAL = "#22d3ee"   # cyan  — measured signal
_C_TRUE   = "#f59e0b"   # amber — ground-truth value (debug)
_C_EST    = "#a855f7"   # purple — current estimate (not yet calibrated)
_C_CAL    = "#4ade80"   # green  — calibrated estimate
_C_POS    = "#4ade80"   # positive reward bar
_C_NEG    = "#f87171"   # negative reward bar

# Param display config: (label for plots, divisor to display unit, unit string)
_PARAM_DISPLAY = {
    "f_q":    ("f_q",   1e9,   "GHz"),
    "f_r":    ("f_r",   1e9,   "GHz"),
    "T1":     ("T₁",    1e-6,  "µs"),
    "T2star": ("T₂*",   1e-6,  "µs"),
    "amp_pi": ("amp_π", 1.0,   ""),
    "t_pi":   ("t_π",   1e-9,  "ns"),
}

# Which param does each experiment primarily reveal?
_EXP_PARAM = {
    "s21":        "f_r",
    "spectrum":   "f_q",
    "power_rabi": "amp_pi",
    "time_rabi":  "t_pi",
    "t1":         "T1",
    "ramsey":     "T2star",
}


class EpisodeLogger:
    """
    Records one calibration episode.  Call ``record()`` after every
    ``env.step()``, then ``save()`` at episode end.

    Parameters
    ----------
    log_dir : str
        Directory where plots and JSON are written (created if absent).
    verbose : bool
        Print a one-liner per step to stdout.
    """

    def __init__(self, log_dir: str, verbose: bool = True):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.verbose = verbose
        self._steps: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, step_idx: int, obs: dict, reward: float, info: dict):
        """
        Record one step and save its signal plot.

        Parameters
        ----------
        step_idx : int   1-based step number
        obs      : dict  observation returned by env.step()
        reward   : float
        info     : dict  info returned by env.step()
        """
        entry = {
            "step":         step_idx,
            "exp_type":     info["exp_type"],
            "reward":       float(reward),
            "n_calibrated": int(info["n_calibrated"]),
            "calibrated":   list(info["calibrated"]),
            "estimates":    {k: float(v) for k, v in info["estimates"].items()},
            "true_params":  {k: float(v) for k, v in info.get("true_params", {}).items()},
        }
        self._steps.append(entry)

        if self.verbose:
            bar = "█" * entry["n_calibrated"] + "░" * (6 - entry["n_calibrated"])
            cal = sorted(entry["calibrated"]) or "-"
            print(f"  step {step_idx:2d} | {info['exp_type']:12s} | "
                  f"[{bar}] {entry['n_calibrated']}/6 | r={reward:+.2f} | {cal}")

        self._save_step_plot(step_idx, obs, info)

    def save(self):
        """Write ``log.json`` and ``progress.png``."""
        log_path = os.path.join(self.log_dir, "log.json")
        with open(log_path, "w") as f:
            json.dump(self._steps, f, indent=2, default=float)
        print(f"[logger] log        → {log_path}")

        self._save_progress_plot()

    # ── Per-step signal plot ──────────────────────────────────────────────────

    def _save_step_plot(self, step_idx: int, obs: dict, info: dict):
        exp      = info["exp_type"]
        x_phys   = np.asarray(info.get("x_phys", obs["x_axis"]))
        x_label  = info.get("x_label", "x")
        y_label  = info.get("y_label", "|signal|")
        sig_re   = obs["signal_re"]
        sig_im   = obs["signal_im"]

        # Magnitude; for S21 optionally convert to dB
        mag = np.sqrt(sig_re ** 2 + sig_im ** 2)
        if exp == "s21":
            y       = 20.0 * np.log10(np.maximum(mag, 1e-10))
            y_label = "|S₂₁| (dB)"
        else:
            y = mag

        # Which parameter does this experiment reveal?
        param_key = _EXP_PARAM.get(exp)
        est_val   = info["estimates"].get(param_key, 0.0) if param_key else 0.0
        true_val  = info.get("true_params", {}).get(param_key, None) if param_key else None
        is_cal    = (param_key in info["calibrated"]) if param_key else False

        if param_key and param_key in _PARAM_DISPLAY:
            plabel, pscale, punit = _PARAM_DISPLAY[param_key]
        else:
            plabel, pscale, punit = "", 1.0, ""

        with plt.rc_context(_RC):
            fig, (ax_sig, ax_iq) = plt.subplots(
                1, 2, figsize=(10, 3.5), dpi=120,
                gridspec_kw={"width_ratios": [3, 1]})
            fig.patch.set_facecolor("#0f172a")

            # ── Left: signal vs sweep axis ────────────────────────────────
            ax_sig.plot(x_phys, y, color=_C_SIGNAL, lw=1.8, alpha=0.9,
                        label="measured")
            ax_sig.fill_between(x_phys, y, y.min(),
                                color=_C_SIGNAL, alpha=0.07)

            # True value (debug reference)
            if true_val is not None and true_val > 0:
                tv = true_val / pscale
                ax_sig.axvline(tv, color=_C_TRUE, lw=1.3, ls="--",
                               label=f"true {plabel} = {tv:.4g} {punit}")

            # Current estimate
            if est_val > 0:
                ev    = est_val / pscale
                ecol  = _C_CAL if is_cal else _C_EST
                elabel = f"est {plabel} = {ev:.4g} {punit}" + (" ✓" if is_cal else "")
                ax_sig.axvline(ev, color=ecol, lw=1.8, ls="-", label=elabel)

            cal_str = f"{info['n_calibrated']}/6 calibrated"
            ax_sig.set_title(
                f"Step {step_idx:02d} — {exp.replace('_', ' ').title()}  [{cal_str}]")
            ax_sig.set_xlabel(x_label)
            ax_sig.set_ylabel(y_label)
            ax_sig.xaxis.set_minor_locator(AutoMinorLocator())
            ax_sig.legend(loc="best")
            ax_sig.grid(True, which="major", alpha=0.3)
            ax_sig.grid(True, which="minor", alpha=0.12)

            # ── Right: IQ scatter ─────────────────────────────────────────
            sc = ax_iq.scatter(sig_re, sig_im,
                               c=np.linspace(0, 1, len(sig_re)),
                               cmap="cool", s=12, alpha=0.75)
            ax_iq.axhline(0, color="#475569", lw=0.7)
            ax_iq.axvline(0, color="#475569", lw=0.7)
            ax_iq.set_aspect("equal", "box")
            ax_iq.set_title("IQ plane")
            ax_iq.set_xlabel("I")
            ax_iq.set_ylabel("Q")
            ax_iq.grid(True, alpha=0.3)
            fig.colorbar(sc, ax=ax_iq, label="sweep →", pad=0.02,
                         fraction=0.06).ax.tick_params(labelsize=7)

            fig.tight_layout(pad=0.8)
            path = os.path.join(self.log_dir, f"step_{step_idx:03d}_{exp}.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)

    # ── Episode progress plot ─────────────────────────────────────────────────

    def _save_progress_plot(self):
        if not self._steps:
            return

        steps   = [s["step"]         for s in self._steps]
        rewards = [s["reward"]        for s in self._steps]
        n_cal   = [s["n_calibrated"]  for s in self._steps]
        true_p  = self._steps[-1].get("true_params", {})

        param_keys   = list(_PARAM_DISPLAY.keys())
        param_labels = [_PARAM_DISPLAY[k][0]  for k in param_keys]
        param_scales = [_PARAM_DISPLAY[k][1]  for k in param_keys]
        param_units  = [_PARAM_DISPLAY[k][2]  for k in param_keys]

        with plt.rc_context(_RC):
            fig = plt.figure(figsize=(15, 9), dpi=120)
            fig.patch.set_facecolor("#0f172a")
            gs  = gridspec.GridSpec(3, 4, figure=fig,
                                    hspace=0.55, wspace=0.4)

            # ── Row 0: reward bars ────────────────────────────────────────
            ax_rew = fig.add_subplot(gs[0, :2])
            colors = [_C_POS if r > 0 else _C_NEG for r in rewards]
            ax_rew.bar(steps, rewards, color=colors, alpha=0.85, width=0.7)
            ax_rew.axhline(0, color="#475569", lw=0.8)
            # Cumulative reward overlay
            cum = np.cumsum(rewards)
            ax_rew2 = ax_rew.twinx()
            ax_rew2.plot(steps, cum, color="#f59e0b", lw=1.5, ls="--",
                         alpha=0.7, label="cumulative")
            ax_rew2.set_ylabel("Cumulative", color="#f59e0b", fontsize=8)
            ax_rew2.tick_params(axis="y", colors="#f59e0b", labelsize=7)
            ax_rew.set_title("Reward per step")
            ax_rew.set_xlabel("Step")
            ax_rew.set_ylabel("Reward")
            ax_rew.grid(True, alpha=0.25)

            # ── Row 0: calibration count ──────────────────────────────────
            ax_cal = fig.add_subplot(gs[0, 2:])
            ax_cal.step(steps, n_cal, color=_C_CAL, lw=2.0, where="post")
            ax_cal.fill_between(steps, n_cal, step="post",
                                color=_C_CAL, alpha=0.12)
            ax_cal.set_ylim(-0.3, 6.5)
            ax_cal.set_yticks(range(7))
            ax_cal.set_title("Calibrated parameters over time")
            ax_cal.set_xlabel("Step")
            ax_cal.set_ylabel("# params calibrated")
            ax_cal.grid(True, alpha=0.25)

            # ── Rows 1-2: per-parameter estimate traces ───────────────────
            for i, (pkey, plabel, pscale, punit) in enumerate(
                    zip(param_keys, param_labels, param_scales, param_units)):
                row = 1 + i // 4
                col = i % 4
                ax  = fig.add_subplot(gs[row, col])

                ests = [s["estimates"].get(pkey, 0.0) / pscale
                        for s in self._steps]
                cals = [pkey in s["calibrated"] for s in self._steps]

                # Colour-coded line segments
                for j in range(len(steps) - 1):
                    c = _C_CAL if cals[j] else _C_EST
                    ax.plot(steps[j:j+2], ests[j:j+2],
                            color=c, lw=1.5, alpha=0.85, solid_capstyle="round")

                # True value reference
                tv = true_p.get(pkey, 0.0)
                if tv > 0:
                    ax.axhline(tv / pscale, color=_C_TRUE, lw=1.2,
                               ls="--", alpha=0.8, label="true")

                unit_str = f" ({punit})" if punit else ""
                ax.set_title(f"{plabel}{unit_str}", pad=4)
                ax.set_xlabel("Step", fontsize=8)
                ax.grid(True, alpha=0.25)
                ax.legend(loc="best")

            path = os.path.join(self.log_dir, "progress.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[logger] progress   → {path}")
