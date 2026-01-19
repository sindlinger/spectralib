# Projeto: SpectraLib + DLL OpenCL (async)

## Objetivo
Mover STFT, Periodograma, Fase e Espectrograma para uma DLL OpenCL assinc (worker + fila pequena) e manter a SpectraLib como front-end leve, com fallback para OpenCL MQL5. Medir metricas CPU/GPU/espera e expor via log, label e Data Window.

## Escopo
- DLL 64-bit (OpenCL) com fila pequena (padrao 3)
- Worker thread GPU (1 por instancia)
- Service MQL5 opcional (background)
- Bridge na SpectraLib (sem mexer em indicadores/EA)

## Fases

### Fase 0 - Preparacao
- [OK] Criar SpectralConfig.mqh (flags globais)
- [OK] Ajustar SpectralImpl/SpectralOpenCLWindows para cleanup/refs
- [OK] Padronizar arquivos ASCII no Service

### Fase 1 - Definir contrato DLL
- Entradas:
  - serie double[]
  - n, nperseg, noverlap, nfft, window, detrend
  - flags: onesided, full matrix, include mag/phase/amp
- Saidas:
  - STFT: complex flat (re+im), mag, phase, amp, dims (nseg, nfreq)
  - Spectrograma: mag/phase/amp + dims
  - Periodograma: espectro + picos + periodos (global/local/sublocal)
  - Fase: fase instantanea (Hilbert)

### Fase 2 - DLL (OpenCLWorker)
- Implementar worker thread + fila pequena (drop oldest)
- Implementar ring buffer opcional (streaming)
- Exportar funcoes:
  - StartWorker(queue, profiling)
  - PushData(job)
  - TryGetResult(job_id or oldest)
  - GetStats(out[...])
  - ResetStats()
  - StopWorker()

### Fase 3 - Bridge na SpectraLib
- Criar SpectralDllBridge.mqh (imports DLL)
- Em SpectralImpl.mqh:
  - stft_1d -> chama DLL quando SPECTRAL_USE_DLL=1
  - spectrogram_1d -> chama DLL
  - periodogram_1d -> chama DLL
- Em SpectralHilbert.mqh:
  - hilbert_analytic_gpu -> chama DLL
- Fallback automatico para OpenCL atual

### Fase 4 - Service (opcional)
- Criar OpenCLWorkerService.mq5 (ASCII)
- Loop com Sleep/IsStopped
- Detectar nova barra (CisNewBar)
- PushData para DLL
- Enviar resultado via EventChartCustom

### Fase 5 - UI/Metricas
- Data Window: buffers numericos no indicador
- Label discreto no grafico
- Log periodico

## Principios
- Nao bloquear o thread do indicador
- Evitar recalculo tick a tick
- Permitir multiplas instancias (com lock opcional)
- GPU profiling opcional

## Entregaveis
- OpenCLWorker.dll
- SpectralDllBridge.mqh
- Ajustes em SpectralImpl.mqh e SpectralHilbert.mqh
- OpenCLWorkerService.mq5 (opcional)
- Indicador com metricas

