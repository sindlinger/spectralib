# AGENTS.md
# Repo: MQL5/Include/spectralib
# Language: PT-BR (ASCII only)

## Objetivo
Implementar uma pipeline assinc com DLL + OpenCL para STFT, Periodograma, Fase e Espectrograma, mantendo a SpectraLib como front-end leve e com fallback. Usar Service MQL5 como gerenciador opcional em background. Exibir metricas (CPU/GPU/espera/bytes/jobs) em log, label e Data Window.

## Regras principais
- Nao quebrar funcoes existentes da SpectraLib.
- Nao criar dependencias rigidas em indicadores/EA; a SpectraLib deve funcionar sozinha.
- Tudo em ASCII.
- A DLL deve ser 64-bit.
- Um worker GPU por instancia (fila pequena), com suporte a multiplas instancias.

## Arquivos chave
- SpectralImpl.mqh: STFT/periodograma/espectrograma (ponto principal de integracao DLL)
- SpectralHilbert.mqh: fase (Hilbert) se for usar na DLL
- SpectralOpenCLFFT.mqh: kernels FFT (referencia para port)
- SpectralConfig.mqh: flags globais (DLL/async/cache/stream)

## Pipeline desejada (alto nivel)
1) MQL5 (indicador/EA) chama SpectraLib
2) SpectraLib decide: DLL (se habilitado) ou fallback OpenCL atual
3) DLL (worker thread) executa GPU e retorna buffer
4) SpectraLib devolve resultado final ao indicador/EA

## Metricas
- CPU_ms (prep)
- GPU_ms (kernel)
- WAIT_ms (fila)
- BYTES_IN/OUT
- JOBS_OK/JOBS_DROP

## Saidas
- STFT: matriz complexa completa + magnitude + fase + amplitude
- Espectrograma: magnitude + fase + amplitude
- Periodograma: espectro completo + periodos global/local/sublocal
- Fase: fase instantanea (Hilbert) e/ou fase por bin da STFT

## Conflitos comuns
- Multi-instancia no mesmo GPU gera contencao (permitir, mas avisar)
- Data Window so existe em indicador, nao em Service

