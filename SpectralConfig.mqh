#ifndef __SPECTRAL_CONFIG_MQH__
#define __SPECTRAL_CONFIG_MQH__

// Config central da SpectraLib (DLL/assinc/stream)

// 0=nao usa DLL, 1=usa DLL (OpenCLWorker)
#define SPECTRAL_USE_DLL 0

// Assincrono via DLL
// 1=assinc (fila + TryGetResult), 0=sincrono (bloqueia)
#define SPECTRAL_DLL_ASYNC 1

// Fila pequena na DLL (se assinc)
#define SPECTRAL_DLL_QUEUE_SIZE 3
#define SPECTRAL_DLL_DROP_OLDEST 1

// Cache/stream
// 1=recalcula apenas em nova barra
#define SPECTRAL_CACHE_ON_NEW_BAR 1
// 1=streaming incremental (ring buffer na DLL)
#define SPECTRAL_STREAMING_MODE 0

// Saidas completas (matrizes)
#define SPECTRAL_STFT_FULL_MATRIX 1
#define SPECTRAL_SPECTRO_FULL_MATRIX 1

// Itens de saida (STFT/Espectrograma)
#define SPECTRAL_INCLUDE_MAG 1
#define SPECTRAL_INCLUDE_PHASE 1
#define SPECTRAL_INCLUDE_AMP 1

// Periodos dominantes
#define SPECTRAL_INCLUDE_PERIOD_GLOBAL 1
#define SPECTRAL_INCLUDE_PERIOD_LOCAL 1
#define SPECTRAL_INCLUDE_PERIOD_SUBLOCAL 1

#endif
