# agents.md — Dominant Multi‑Cycle Oscillator (STFT + Kalman + Hilbert + Savitzky–Golay) for MT5 (MQL5)

## Mission
Implement a MetaTrader 5 indicator in **MQL5** named:
`DominantCycles_STFT_Kalman_Hilbert_SG.mq5`

It must output a **free oscillator** (no clipping, no normalization to [-1,1], no ceiling/floor) built from **3 to 7 simultaneously tracked dominant price cycles**. The cycles must be extracted per bar using **STFT-like spectral estimation (causal trailing window)**, stabilized with **Kalman filters applied to attributes** (frequency/omega, amplitude, phase), refined using **Hilbert analytic signal** and smoothed using **Savitzky–Golay** (endpoint/causal or symmetric with explicit repaint mode).

The result must be suitable for conversion into an EA via iCustom: stable buffers, direction, flip signals, and cycle attributes exposed.

## Hard constraints (non‑negotiable)
1. **Zero unused inputs**: every `input` must affect code paths and outputs measurably. If an input is not used, remove it.
2. **No hidden normalization/clamping**: no rescaling to fixed bounds; no “auto normalize”; no saturation.
3. **No division-by-zero artifacts**: must be robust on dojis (Open==Close), flat candles, and zero volatility segments. Use epsilons only where mathematically necessary and do not create spikes.
4. **Causal by default**: default mode must not use future bars (no repaint). If a true zero-phase/symmetric mode is offered, it must be explicit, labeled “REPAINT/NON‑CAUSAL”, and off by default.
5. **3–7 cycles continuously available** (when enough bars exist): per bar, output K cycles (K input) as time series, plus their omega/period/amplitude/phase and confidence.
6. **Group delay**: compute and expose group delay for the Hilbert FIR and for any symmetric SG mode; apply compensation only in the explicit non‑causal mode. In causal mode, use prediction/phase-referencing instead of repaint.
7. **EA‑ready buffers**: provide buffers for:
   - sum oscillator,
   - each cycle waveform,
   - per-cycle omega_true, period_true, amplitude_true, phase_unwrapped, phase_wrapped,
   - per-cycle SNR/confidence,
   - direction (+1/-1) and flip signal (+1/-1 at turning points) for the chosen plotted output.

## Deliverables
1. `DominantCycles_STFT_Kalman_Hilbert_SG.mq5` (indicator) — compiles with no errors/warnings in MetaEditor.
2. `README.md` — explains algorithm, parameters, buffer indices, causal vs repaint mode, and recommended presets.
3. `ExampleEA_iCustom.mq5` (optional but strongly recommended) — minimal EA showing how to read `dir`, `flip`, SNR gate, and how to trade only on closed bars.

## Algorithm requirements (must be actually implemented, not described only)

### A) Input price series
Support at least: Close, HLC3, OHLC4 (enumeration input).
No bar-range normalization.

### B) Detrending (must not distort cycle amplitude/phase unnecessarily)
Provide two modes:
1) **Kalman Local Linear Trend (LLT)** (causal): state [level, slope], measurement price.
2) **Savitzky–Golay endpoint regression** (causal): polynomial fit over trailing window W, evaluate at last point (zero extra lag), subtract as trend.
Default: Kalman LLT (causal).
Output the detrended residual: `resid = price - trend`.

### C) Spectral estimation (STFT-like, causal trailing window)
Per bar (or per new closed bar), compute spectrum on last N samples of `resid`:
- Use a Hann window (at minimum). Optionally add Blackman; if offered, must be used.
- Compute only bins that correspond to period range [MinPeriod..MaxPeriod] and ensure bins are dense enough.
Implementation choices:
- Efficient Goertzel per bin (preferred) OR radix-2 FFT (acceptable).
Must compute complex coefficient to extract amplitude+phase (not magnitude only).

Peak picking:
- Find top `Kcand` peaks (>=K, e.g., 2K) by power in the band.
- Use local maxima constraint (avoid picking adjacent bins as separate peaks unless distinct).
Frequency refinement:
- Parabolic interpolation on log-power using neighbors OR Jacobsen/Quinn estimator.
- Recompute complex coefficient at refined omega to get refined amplitude and phase.

Confidence/SNR:
- Compute SNR per peak: power_peak / (mean power of remaining band + eps).
Expose SNR in linear and dB.

### D) Multi-cycle tracking & data association
Maintain **K tracks** (K=3..7), each representing one dominant cycle over time.
At each bar:
1) Predict each track’s omega via Kalman (omega + omega_dot).
2) Associate measured peaks to tracks using a cost function (|omega_meas-omega_pred| normalized by sigma, plus penalty for low SNR).
   - Use greedy matching (K<=7) or Hungarian algorithm; must be deterministic and stable.
3) Update each track with measurement (omega_meas, amp_meas, phase_meas), using:
   - Kalman on omega,
   - Kalman on amplitude,
   - Phase update using unwrap around predicted phase (avoid ±π jumps).
4) Handle birth/death:
   - If a track is unmatched N bars, decay confidence and allow replacement by a new peak.
   - If fewer than K good tracks exist, spawn from strongest unassigned peaks.

Tracks must be sorted by current power/confidence; user can choose “top K sum” or “top 2” etc without cycles swapping causing discontinuities (use track IDs internally).

### E) Hilbert analytic signal per track (refinement)
For each track waveform (or bandpassed residual around omega_true):
- Implement a FIR Hilbert transformer (odd length L), windowed (e.g., Hamming), with known group delay (L-1)/2.
- Compute analytic signal: a(t)=x(t)+jH{x(t)}.
- Use it to refine instantaneous phase and amplitude:
  - phase_h = atan2(im,re), unwrap using prediction
  - amp_h = sqrt(re^2+im^2)
Fuse with STFT measurements in the Kalman update (measurement fusion), with weights based on SNR/confidence.

### F) Savitzky–Golay smoothing (attributes)
Use SG to smooth:
- omega_true (or period_true),
- amplitude_true,
- unwrapped phase derivative (optional).
Must be **causal endpoint regression** by default; if symmetric SG is provided, mark as REPAINT and compute its group delay.

### G) Reconstruct cycle waveforms (no normalization)
For each track k:
- `cycle_k[t] = amp_true_k[t] * cos(phase_true_k[t])` in price units.
No clipping.

Sum output (user-controlled):
- Provide plot mode:
  1) Sum of Top K (default),
  2) Sum of Top 2,
  3) Single cycle (select index),
  4) User mask (bitmask 1..7).
All modes must be implemented, and the mask must actually be applied.

### H) Output & visuals
- Separate subwindow indicator.
- Main plotted line: chosen mode output (sum or selected cycles).
- Provide optional plots: individual cycles (up to K) with distinct colors (static or palette).
- Provide robust direction/coloring:
  - direction = sign(output[t] - output[t-1]) on closed bars
  - flip signal when direction changes (+1 upturn, -1 downturn)
- Forecast (optional but real):
  - Provide H-step ahead forecast using phase propagation: phase(t+H)=phase(t)+omega_true*H and amp persistence/decay
  - forecast_out = sum_k amp_k*cos(phase_k + omega_k*H)
  - Plot forecast shifted to the right (only if chart has space); also expose numeric buffer for EA.

### I) Performance
- Must run on 10k–50k bars without freezing.
- Compute heavy STFT only on new closed bars (optional input “calc_on_tick” true/false).
- Precompute window weights and bin sin/cos tables.

## Acceptance tests (must be executed and passed)
1) Compile with 0 errors/warnings.
2) Visual sanity:
   - output is free (no saturation), follows oscillatory structure, flips color frequently in ranging regimes.
3) Parameter sensitivity:
   - Changing MinPeriod/MaxPeriod must change selected frequencies and waveforms.
   - Changing K (3..7) must add/remove cycle buffers and change sum.
   - Turning off Hilbert fusion must measurably change phase smoothness and slightly alter waveform.
4) Robustness:
   - No spikes on sequences of dojis / flat bars.
   - No NaNs/INF in any buffer.
5) EA buffers:
   - `dir` and `flip` are non-zero and consistent with visual slope on closed bars.
   - forecast buffer differs from current value (not identical).

## Coding rules (MQL5 specifics)
- Use dynamic arrays in function parameters (`double &x[]`, `double &P[][4]` etc).
- Avoid illegal casts like `(void)`.
- All buffers must be `ArraySetAsSeries(..., true)` where appropriate.
- Do not write into indexes that are not calculated (respect prev_calculated).
- Provide clear comments and a buffer index map in README.

## Output buffer map (must be documented)
At minimum:
0: OutputMain (sum/mode)
1: ColorIndex (if DRAW_COLOR_LINE)
2: Forecast (numeric, not necessarily shifted)
3: Dir (+1/-1)
4: Flip (+1/-1/0)
Then for k=1..K:
- cycle_k
- omega_k, period_k, amp_k, phase_unw_k, phase_w_k, snr_k, conf_k
Plus trend and residual outputs.

End of agents.md
