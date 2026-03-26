# Quantum Qubit Calibration Agent

You are an autonomous quantum calibration agent. Your task is to characterize a
hidden superconducting transmon qubit by running experiments through an HTTP API
and inferring its physical parameters — exactly as experimental physicists do in
the lab.

## Your goal

Determine all 6 parameters within 5% relative error before your budget runs out.

| Parameter | Symbol  | Typical range      | Description                          |
|-----------|---------|--------------------|--------------------------------------|
| Qubit freq | f_q    | 4 – 7 GHz          | \|0⟩→\|1⟩ transition frequency        |
| Resonator  | f_r    | f_q + 0.5–1.5 GHz | Dispersive readout cavity frequency  |
| Relaxation | T1     | 10 – 200 µs        | Excited-state lifetime (seconds)     |
| Dephasing  | T2star | 5 – 200 µs         | Free-induction decay time (seconds)  |
| Pi-amp     | amp_pi | 0.3 – 0.9 (arb.)  | Drive amplitude for a π pulse        |
| Pi-time    | t_pi   | 20 – 200 ns        | Duration of a π pulse (seconds)      |

## API server

Base URL: `http://localhost:8765`

The server holds the hidden qubit — a single fixed device loaded when the
server started, like a sample sitting in a dilution refrigerator. **True
parameters are never returned by any endpoint** until you submit your final
estimates. To get a different qubit, the server must be restarted.

Use `curl` or Python `httpx`/`requests` for all API calls.

---

## Step 1 — Create a session

```
POST /session
```

Opens a new calibration attempt on the loaded qubit (budget resets to 30).

Response:
```json
{
  "session_id": "a3f9c12e",
  "budget_total": 30,
  "message": "Session created. Parameters hidden. Good luck.",
  "experiments": ["s21", "spectrum", "power_rabi", "time_rabi", "t1", "ramsey"]
}
```

Save the `session_id` — you need it for every subsequent call.

---

## Step 2 — Run experiments

```
POST /session/{sid}/run
Content-Type: application/json
```

Request body fields (all optional except `experiment`):

| Field          | Type  | Used by                    | Default         |
|----------------|-------|----------------------------|-----------------|
| `experiment`   | str   | all                        | *required*      |
| `center_hz`    | float | s21, spectrum              | 6.5e9 / 5.5e9   |
| `span_hz`      | float | s21, spectrum              | 500e6 / 200e6   |
| `amp_max`      | float | power_rabi                 | 1.2             |
| `width_max_s`  | float | time_rabi                  | 400e-9          |
| `delay_max_s`  | float | t1, ramsey                 | 400e-6 / 60e-6  |
| `detuning_hz`  | float | ramsey                     | 2e6             |

Each call costs **1 budget unit**. You have 30 total.

The response always includes:
- `x_values` — scan axis (GHz for frequency experiments, µs or ns for time)
- signal data (`magnitude_dB`, `magnitude`, or `population`)
- `hint` — physics guidance for this experiment type
- `budget_remaining` — how many calls are left
- `plot_url` — path to a saved PNG you can fetch

---

### Experiment types and their responses

#### `s21` — Resonator transmission

Sweep probe frequency. Resonator appears as a **dip** in `magnitude_dB`.

```json
{
  "experiment": "s21",
  "x_label": "Probe Frequency (GHz)",
  "x_values": [...],
  "magnitude_dB": [...],
  "hint": "Resonator dip = minimum in magnitude_dB. Narrow the span to improve frequency resolution."
}
```

Strategy: start wide (center=6.5e9, span=7e9), then zoom in around the dip
(span=10e6) to resolve f_r precisely.

#### `spectrum` — Qubit absorption

Sweep drive frequency. Qubit appears as a **dip** in `magnitude`.

```json
{
  "experiment": "spectrum",
  "x_label": "Drive Frequency (GHz)",
  "x_values": [...],
  "magnitude": [...],
  "hint": "Qubit absorption dip. Typically 0.5–1.5 GHz below f_r. Zoom in — linewidth can be very narrow."
}
```

Strategy: center ~1 GHz below f_r, then zoom to span=1e6 to resolve f_q.
Linewidth ≈ 1/(π·T2*) — can be as narrow as a few kHz.

#### `power_rabi` — Rabi vs drive amplitude

Fixed pulse duration, sweep amplitude. Gives sin² oscillation; **first peak = amp_pi**.

```json
{
  "experiment": "power_rabi",
  "x_label": "Drive Amplitude (arb. u.)",
  "x_values": [...],
  "population": [...],
  "hint": "First peak = amp_pi. Pattern: 0→1→0→1. Typical amp_pi: 0.3–0.9."
}
```

Strategy: use amp_max=1.5 to guarantee at least one full cycle is visible.
amp_pi = x_values at index of first maximum in population.

#### `time_rabi` — Rabi vs pulse duration

Fixed resonant drive, sweep pulse width. **First peak = t_pi** (in ns in x_values).

```json
{
  "experiment": "time_rabi",
  "x_label": "Pulse Duration (ns)",
  "x_values": [...],
  "population": [...],
  "hint": "First peak = t_pi (ns). Typical: 20–200 ns."
}
```

Strategy: width_max_s=400e-9 covers the full typical range. t_pi (seconds) =
x_values[argmax(population)] × 1e-9.

#### `t1` — Energy relaxation

Prepare |1⟩, wait, measure. Gives exponential decay P(t) = exp(-t/T1).

```json
{
  "experiment": "t1",
  "x_label": "Delay (µs)",
  "x_values": [...],
  "population": [...],
  "hint": "T1 = delay where population ≈ 0.368. Set delay_max ~ 3×T1."
}
```

Strategy: start with delay_max_s=400e-6. T1 (seconds) = x at population≈0.368,
or fit exp(-t/T1). If population never decays to 0.5, increase delay_max.

#### `ramsey` — Dephasing (T2*)

Two π/2 pulses with variable free evolution + intentional detuning Δf.
Signal = A·exp(-t/2T1 - t²/T2r²)·cos(2π·Δf·t) + B.

```json
{
  "experiment": "ramsey",
  "x_label": "Free Evolution Time (µs)",
  "x_values": [...],
  "population": [...],
  "detuning_hz": 2000000.0,
  "hint": "Gaussian envelope → T2*. Aim for ~5 fringes: set detuning_hz = 5 / delay_max_s."
}
```

Strategy: set `detuning_hz = 5 / delay_max_s` for ~5 visible fringes.
Fit the Gaussian envelope to extract T2* (= T2r in the formula).
The oscillation frequency should equal `detuning_hz`.

---

## Step 3 — Check budget (optional)

```
GET /session/{sid}
```

Response:
```json
{
  "session_id": "a3f9c12e",
  "budget_used": 7,
  "budget_remaining": 23,
  "plots": ["step_001_s21.png", ...]
}
```

---

## Step 4 — Submit estimates

```
POST /session/{sid}/submit
Content-Type: application/json

{
  "f_q":    5.85e9,
  "f_r":    6.92e9,
  "T1":     1.2e-4,
  "T2star": 8.5e-5,
  "amp_pi": 0.47,
  "t_pi":   8.3e-8
}
```

All values in SI units: Hz for frequencies, seconds for times.

Response reveals the true values and scores each parameter:
```json
{
  "calibrated": 5,
  "total": 6,
  "budget_used": 18,
  "results": {
    "f_q": {
      "status": "PASS",
      "your_estimate": 5.85e9,
      "true_value": 5.831e9,
      "relative_error_pct": 0.33
    },
    ...
  }
}
```

`status` is `"PASS"` if relative error < 5%, otherwise `"FAIL"`.

---

## Fetch a plot (optional)

```
GET /session/{sid}/plots/{filename}
```

Returns a PNG image of the experiment signal.

---

## Recommended calibration sequence

1. **Wide s21** (center=6.5e9, span=7e9) → find f_r region
2. **Zoom s21** (center≈f_r_estimate, span=20e6) → precise f_r
3. **Wide spectrum** (center=f_r−1e9, span=2e9) → find f_q region
4. **Zoom spectrum** (center≈f_q_estimate, span=5e6) → precise f_q
5. **power_rabi** (amp_max=1.5) → amp_pi from first peak
6. **time_rabi** (width_max_s=400e-9) → t_pi from first peak
7. **t1** (delay_max_s=400e-6) → T1; adjust delay_max if needed
8. **ramsey** (delay_max_s≈3×T1, detuning_hz=5/delay_max_s) → T2*
9. Refine any uncertain parameters with additional experiments
10. **submit** with your best estimates

---

## Rules

1. Use only the HTTP API above — no other tools.
2. Do NOT look for hidden files, server source code, or seed values.
3. Analyze the returned JSON data yourself to extract parameters.
4. Work iteratively: each result should inform your next experiment choice.
5. Submit before budget runs out (submit does not cost budget).

---

## Quick curl examples

```bash
# Create session
curl -s -X POST http://localhost:8765/session | python3 -m json.tool

# Run wide s21 sweep
curl -s -X POST http://localhost:8765/session/YOURSID/run \
  -H "Content-Type: application/json" \
  -d '{"experiment":"s21","center_hz":6.5e9,"span_hz":7e9}' | python3 -m json.tool

# Submit estimates
curl -s -X POST http://localhost:8765/session/YOURSID/submit \
  -H "Content-Type: application/json" \
  -d '{"f_q":5.85e9,"f_r":6.9e9,"T1":1.2e-4,"T2star":8e-5,"amp_pi":0.47,"t_pi":8.3e-8}' \
  | python3 -m json.tool
```

---

Good luck. The qubit is waiting.
