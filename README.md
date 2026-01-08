# DominantCycles_STFT_Kalman_Hilbert_SG

Multi-cycle oscillator for MT5 using causal STFT (Goertzel) + Kalman tracking + Hilbert analytic refinement + Savitzky–Golay smoothing. Output is a free oscillator (no clipping/normalization) built from 3–7 dominant cycles. Designed to be EA-friendly via iCustom.

## Overview
- Input price: Close, HLC3, or OHLC4.
- Detrend: Kalman LLT (causal) or SG endpoint regression (causal) → residual.
- Spectral estimate: Goertzel per bin on trailing window with Hann/Blackman, peak pick + parabolic refinement.
- Tracking: Kalman on omega and amplitude, phase unwrap around predicted phase; deterministic association.
- Hilbert: FIR transformer (odd length), group delay exposed; optional fusion of Hilbert amp/phase.
- SG smoothing: endpoint by default; symmetric option is **REPAINT/NON‑CAUSAL** with group delay exposed.
- Output modes: sum top K, sum top 2, single ranked cycle, or bitmask.
- Direction/flip buffers for EA logic.
- Forecast: H-bar ahead phase propagation with optional amplitude decay.

## Causal vs REPAINT
- **Default is causal**: only past data, no future bars.
- **REPAINT/NON‑CAUSAL** (`InpSmoothMode=SMOOTH_REPAINT`) uses symmetric SG smoothing and applies Hilbert group delay compensation. This uses future bars for smoothing and is explicitly non‑causal.
- Hilbert group delay and SG group delay are exposed in buffers for reference.

## Inputs (summary)
- PriceSource: PRICE_SRC_CLOSE, PRICE_SRC_HLC3, PRICE_SRC_OHLC4
- DetrendMode: KALMAN_LLT or SG_ENDPOINT
- STFT: window length, min/max period, oversampling, window type
- Tracking: K cycles (3–7), association cost tuning, SNR gate, miss/replace settings
- Hilbert: length, blend, resonator pole
- SG smoothing: window/order, causal vs repaint
- PlotMode: sum top K, sum top 2, single cycle index, bitmask
- Forecast: horizon, amplitude decay, optional shift

## Buffer map (CopyBuffer indices)
Base indices (fixed for up to 7 cycles):
- 0: OutputMain (sum/mode)
- 1: ColorIndex (for main plot)
- 2: ForecastMain (numeric, unshifted)
- 3: Dir (+1/-1/0)
- 4: Flip (+1/-1/0)

Per-cycle buffers (k = 1..7):
- base = 5 + (k-1)*9
- base+0: cycle_k
- base+1: omega_k (rad/bar)
- base+2: period_k (bars)
- base+3: amp_k
- base+4: phase_unw_k (rad)
- base+5: phase_w_k (rad, wrapped)
- base+6: snr_lin_k
- base+7: snr_db_k
- base+8: conf_k

After cycles:
- 68: Trend
- 69: Residual
- 70: Hilbert group delay
- 71: SG group delay (symmetric mode)

If `InpKCycles` < 7, buffers for higher cycles are set to `EMPTY_VALUE`.

## Plotting
- Main plot: color line by Dir (green up, red down, gray flat)
- Optional cycle plots: enable with `InpShowCycles`
- Optional forecast plot: enable with `InpShowForecastPlot`, shift with `InpForecastShift`

## Recommended presets
### EURUSD M15 (range-friendly)
- InpSTFTWindow: 128
- InpMinPeriodBars: 10
- InpMaxPeriodBars: 60
- InpKCycles: 4
- InpHilbertLen: 31
- InpSGSmoothWindow: 11
- InpSmoothMode: SMOOTH_CAUSAL

### Indices / Crypto (slower swings)
- InpSTFTWindow: 192
- InpMinPeriodBars: 12
- InpMaxPeriodBars: 120
- InpKCycles: 5
- InpHilbertLen: 41
- InpSGSmoothWindow: 13
- InpSmoothMode: SMOOTH_CAUSAL

## Notes
- Default mode is causal (no repaint). If you enable symmetric SG, it will repaint.
- Use `InpPlotMode=PLOT_MASK` with `InpCycleMask` bits 1..7 to select specific cycles.
- For EA usage, prefer reading shift=1 (closed bar) unless `InpCalcOnTick=true`.

## Example EA
- `ExampleEA_iCustom.mq5` reads OutputMain, Dir, Flip, average SNR, and Cycle #1 period.
- Ensure `InpKCycles` in the EA matches the indicator setting.
