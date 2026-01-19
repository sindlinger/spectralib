# Project Status (SpectralV7 OpenCL)

Data: 2026-01-09
Escopo: Port Python -> MQL5 OpenCL float64 (sem fallback), foco STFT/ISTFT/PSD + wavelets/waveforms.

## Estado geral
- Pipeline STFT/ISTFT/PSD **fechado em GPU** (inclui magnitude/angle/phase + unwrap em GPU).
- CWT FFT e CWT conv **em GPU**.
- Waveforms **em GPU**.
- Wavelet filters (PyWT) **construídos em GPU**.
- Checagens COLA/NOLA **em GPU**.
- Coherence final **em GPU**.
- Scan de pico em PyWT_CentralFrequency **em GPU**.

## Módulos prontos (GPU compute)
- `v7_port/SpectralOpenCLFFT.mqh`
- `v7_port/SpectralOpenCLWindows.mqh`
- `v7_port/SpectralOpenCL.mqh`
- `v7_port/SpectralImpl.mqh` (STFT/ISTFT/PSD/spectrogram)
- `v7_port/SavitzkyGolay.mqh`
- `v7_port/SpectralWaveforms.mqh`
- `v7_port/SpectralWavelets.mqh`
- `v7_port/SpectralConvolve.mqh`
- `v7_port/SpectralPyWTConvolution.mqh`
- `v7_port/SpectralPyWTDWT.mqh`
- `v7_port/SpectralPyWTCWT.mqh`
- `v7_port/SpectralPyWTWavelets.mqh` (construção de filtros em kernel)

## Kernels adicionados recentemente
- `cabs_cplx`, `carg_cplx`, `unwrap_phase_rows`
- `cola_binsums`, `nola_binsums`, `cola_check`, `nola_check`
- `coherence_ratio`
- `max_mag_index`

## Ainda fora da GPU (helpers/dados, baixo custo)
- `v7_port/SpectralCommon.mqh` (tipos/Complex)
- `v7_port/SpectralArrayTools.mqh` (helpers)
- `v7_port/SpectralSignalTools.mqh` (helpers)
- `v7_port/SpectralLinalg.mqh` (helpers)
- `v7_port/SpectralFFT.mqh` e `v7_port/SpectralWindows.mqh` (referência CPU)
- `v7_port/SpectralPyWTCoeffs.mqh` e `v7_port/SpectralPyWTCommon.mqh` (dados/metadados)

## A fazer (integração no “aplicador”/indicador)
1) Criar **aplicador mínimo** que só chama 1 função do pipeline (ex.: `stft_1d` ou `spectrogram_1d` com janela fixa)
2) Expor 1 saída simples e validar no gráfico
3) Adicionar 1 parâmetro por vez (nperseg, noverlap, window)
4) Somente depois ligar múltiplas saídas e recursos avançados

## Observações
- Tudo está em float64 (OpenCL double).
- Sem simplificações: median/unwrap/ratio estão em kernels com lógica integral.
- CPU apenas para leitura final de buffers e controle de fluxo (inevitável em MQL).

