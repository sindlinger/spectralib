# Project Map (SpectralV7 OpenCL)

## Visao geral
- Nome do projeto/indicador: SpectralV7 (port Python -> MQL5 OpenCL float64)
- Objetivo: portar STFT/FFT/janelas/filters/SG/waveforms para MQL5 usando OpenCL (float64, sem fallback CPU) em modulos .mqh.
- Saidas principais: API modular (janelas, FFT, STFT/spectral_helper, filtros, waveforms) para futura integracao no indicador DominantCycles.

## Arvore de arquivos
- `v7_port/SpectralCommon.mqh`
- `v7_port/SpectralArrayTools.mqh`
- `v7_port/SpectralSignalTools.mqh`
- `v7_port/SpectralLinalg.mqh`
- `v7_port/SavitzkyGolay.mqh`
- `v7_port/SpectralFFT.mqh`
- `v7_port/SpectralOpenCLFFT.mqh`
- `v7_port/SpectralOpenCLWindows.mqh`
- `v7_port/SpectralWindows.mqh` (CPU ref; nao usar no caminho principal)
- `v7_port/SpectralOpenCL.mqh` (kernels auxiliares, ex.: Lomb-Scargle)
- `v7_port/SpectralImpl.mqh` (spectral_helper/STFT; em port OpenCL)
- `v7_port/SpectralWaveforms.mqh` (waveforms OpenCL float64)
- `v7_port/SpectralWavelets.mqh` (wavelets OpenCL float64)
- `v7_port/SpectralConvolve.mqh` (convolucoes ordem 2/3 OpenCL float64)
- `v7_port/SpectralPyWTCoeffs.mqh` (coeficientes PyWavelets, float64)
- `v7_port/SpectralPyWTCommon.mqh` (enums/helpers PyWavelets)
- `v7_port/SpectralPyWTWavelets.mqh` (construcao de wavelets discretas PyWavelets)
- `v7_port/SpectralPyWTConvolution.mqh` (convolucao PyWavelets em OpenCL)
- `v7_port/SpectralPyWTDWT.mqh` (DWT/IDWT/SWT 1D usando conv OpenCL)
- `v7_port/SpectralPyWTCWT.mqh` (CWT PyWavelets 1D)

## Fluxo de dados (pipeline)
1. Feed (x/y/serie de entrada)
2. Janelas (OpenCL) + extensao (odd/even/const/zeros)
3. STFT/FFT (OpenCL float64)
4. Espectro/magnitudes/metrics
5. Waveforms/transformacoes

## Modulos (contratos)
- `SpectralCommon.mqh`
  - Responsabilidade: tipos Complex64, helpers basicos.
  - Inputs: n/a
  - Outputs: Complex64, funcoes Cx.

- `SpectralOpenCLFFT.mqh`
  - Responsabilidade: FFT complexa float64 em OpenCL (bit-reversal + stages).
  - Inputs: arrays complexos (real/imag intercalado).
  - Outputs: espectro FFT (in-place/out)

- `SpectralOpenCLWindows.mqh`
  - Responsabilidade: gerar janelas (hann/blackman/kaiser/...) em OpenCL.
  - Inputs: tipo de janela + params
  - Outputs: vetor janela float64

- `SpectralImpl.mqh`
  - Responsabilidade: spectral_helper/STFT (segmentacao, janela, detrend, FFT)
  - Inputs: series, fs, nperseg, noverlap, nfft, etc.
  - Outputs: freqs, times, result complex

- `SpectralWaveforms.mqh`
  - Responsabilidade: waveforms (sawtooth, square, gausspulse, chirp, unit_impulse) em OpenCL.
  - Inputs: arrays t, parametros de forma/frequencia.
  - Outputs: arrays de saida (double).

- `SpectralWavelets.mqh`
  - Responsabilidade: wavelets (qmf, morlet, morlet2, ricker, cwt) em OpenCL.
  - Inputs: dados, parametros de wavelet, widths.
  - Outputs: vetores/matrizes reais ou complexas.

- `SpectralConvolve.mqh`
  - Responsabilidade: convolucoes 1D de 2a e 3a ordem (valid) em OpenCL.
  - Inputs: in1 1D, in2 2D/3D (filtros).
  - Outputs: vetor resultado (valid).

- `SpectralPyWTCoeffs.mqh`
  - Responsabilidade: coeficientes (db/sym/coif/bior/dmey) em float64.
  - Inputs: n/a
  - Outputs: arrays constantes.

- `SpectralPyWTCommon.mqh`
  - Responsabilidade: enums e funcoes de tamanho (dwt/swt) PyWavelets.
  - Inputs: n/a
  - Outputs: helpers de comprimento.

- `SpectralPyWTWavelets.mqh`
  - Responsabilidade: construcao de filtros dec/rec (db/sym/coif/bior/rbio/dmey/haar).
  - Inputs: familia + ordem.
  - Outputs: filtros e metadados em struct.

- `SpectralPyWTConvolution.mqh`
  - Responsabilidade: down/upsampling convolution PyWavelets (modos de extensao completos).
  - Inputs: sinal, filtro, modo, step.
  - Outputs: vetor convoluido (OpenCL, float64).

- `SpectralPyWTDWT.mqh`
  - Responsabilidade: DWT/IDWT/SWT 1D (dec_a/dec_d/rec_a/rec_d).
  - Inputs: sinal + wavelet.
  - Outputs: coeficientes e reconstrucao.

- `SpectralPyWTCWT.mqh`
  - Responsabilidade: CWT 1D (conv/fft) com wavelets continuas PyWavelets.
  - Inputs: data, scales, nome wavelet, metodo, precision.
  - Outputs: matriz de coeficientes + frequencias.

- `SavitzkyGolay.mqh`
  - Responsabilidade: coeficientes SG + convolucao (port OpenCL)

- `SpectralArrayTools.mqh`
  - Responsabilidade: extenso/axis helpers (odd/even/const/zeros)

- `SpectralSignalTools.mqh`
  - Responsabilidade: detrend/segment ops (OpenCL)

- `SpectralOpenCL.mqh`
  - Responsabilidade: kernels auxiliares (Lomb-Scargle, etc.)

## Modos e flags importantes
- Sem fallback CPU no caminho principal
- Float64 OpenCL em todos os kernels
- Janelas e FFT devem operar em GPU

## Assumptions (estavel)
- OpenCL disponivel (GPU) e double suportado.
- Funcoes CPU sao apenas referencia, nao usadas no pipeline principal.
