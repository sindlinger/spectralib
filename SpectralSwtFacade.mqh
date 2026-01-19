//+------------------------------------------------------------------+
//| SpectralSwtFacade.mqh                                            |
//| Wrapper SWT (local ou serviço) para uso por indicadores          |
//+------------------------------------------------------------------+
#ifndef __SPECTRAL_SWT_FACADE_MQH__
#define __SPECTRAL_SWT_FACADE_MQH__

#include <spectralib/SpectralPyWTDWT.mqh>
#include <spectralib/SpectralSwtStreamClient.mqh>

inline int _swt_prevpow2(int v){ int n=1; while((n<<1) <= v) n <<= 1; return n; }

// estado interno (arquivo unico)
static SwtAsyncState gSwtSt;
static bool  gSwtStInit=false;
static bool  gSwtReady=false;
static bool  gSwtShow=false;
static ulong gSwtStart=0;
static int   gSwtLastPct=-1;
static bool  gSwtAnnounced=false;
static double gSwtLast0=0.0;
static double gSwtLast1=0.0;
static bool   gSwtHas=false;
// fila de amostras (evita misturar INIT com PUSH)
static double gSwtQSamples[];
static int    gSwtQFlags[];
static int    gSwtQHead=0;
static int    gSwtQCount=0;
static int    gSwtQCap=64;

inline void _swt_state_init()
{
   if(!gSwtStInit){ SwtAsyncReset(gSwtSt); gSwtStInit=true; }
}

inline void _swt_q_init()
{
   if(ArraySize(gSwtQSamples) != gSwtQCap)
   {
      ArrayResize(gSwtQSamples, gSwtQCap);
      ArrayResize(gSwtQFlags, gSwtQCap);
      gSwtQHead = 0;
      gSwtQCount = 0;
   }
}

inline void _swt_q_push(const double sample, const int flag)
{
   _swt_q_init();
   if(gSwtQCount < gSwtQCap)
   {
      int idx = (gSwtQHead + gSwtQCount) % gSwtQCap;
      gSwtQSamples[idx] = sample;
      gSwtQFlags[idx] = flag;
      gSwtQCount++;
   }
   else
   {
      // fila cheia: sobrescreve o mais antigo
      gSwtQSamples[gSwtQHead] = sample;
      gSwtQFlags[gSwtQHead] = flag;
      gSwtQHead = (gSwtQHead + 1) % gSwtQCap;
   }
}

inline bool _swt_q_pop(double &sample, int &flag)
{
   if(gSwtQCount <= 0) return false;
   _swt_q_init();
   sample = gSwtQSamples[gSwtQHead];
   flag = gSwtQFlags[gSwtQHead];
   gSwtQHead = (gSwtQHead + 1) % gSwtQCap;
   gSwtQCount--;
   return true;
}

inline void _swt_log_init_progress(const int timer_ms)
{
   if(!gSwtShow) return;
   _swt_state_init();
   SwtAsyncState st; st = gSwtSt;

   int len = st.init_in_progress ? st.len : st.want_len;
   int level = st.init_in_progress ? st.level : st.want_level;
   int chunk = st.chunk_size;
   if(len <= 0 || chunk <= 0) return;
   int chunks_total = (len + chunk - 1) / chunk;

   if(!gSwtAnnounced && (st.init_in_progress || st.want_init))
   {
      gSwtStart = GetTickCount();
      gSwtLastPct = -1;
      gSwtAnnounced = true;
      PrintFormat("SWT INIT: len=%d level=%d batch=%d batches=%d timer=%dms",
                  len, level, chunk, chunks_total, timer_ms);
   }

   if(st.init_in_progress)
   {
      int pct = (int)MathFloor(100.0 * (double)st.chunk_pos / (double)len);
      if(pct > 100) pct = 100;
      if(pct >= gSwtLastPct + 10)
      {
         ulong elapsed = GetTickCount() - gSwtStart;
         PrintFormat("SWT INIT: %d%% (%d/%d) elapsed=%ums",
                     pct, st.chunk_pos, len, (uint)elapsed);
         gSwtLastPct = pct;
      }
   }

   if(gSwtAnnounced && st.init_ready && !st.init_in_progress)
   {
      ulong elapsed = GetTickCount() - gSwtStart;
      PrintFormat("SWT INIT DONE: elapsed=%ums", (uint)elapsed);
      gSwtAnnounced = false;
      gSwtLastPct = -1;
   }
}

inline void SwtFacadeOnTimer(const double &raw[], const bool useService,
                             const bool showProgress, const int timer_ms,
                             const int runtimeBatchSize)
{
   if(!useService) return;
   _swt_state_init();
   SwtAsyncState st; st = gSwtSt;
   gSwtShow = showProgress;
   // se ainda inicializando, processa init e sai
   if(st.want_init || st.init_in_progress)
   {
      SwtAsyncTick(raw, st);
      gSwtSt = st;
      if(st.has)
        {
         gSwtLast0 = st.last0;
         gSwtLast1 = st.last1;
         gSwtHas = true;
        }
      _swt_log_init_progress(timer_ms);
      return;
   }
   if(!st.init_ready) return;

   int steps = (runtimeBatchSize > 0 ? runtimeBatchSize : 1);
   bool did = false;
   for(int i=0; i<steps; ++i)
   {
      double s=0.0; int f=0;
      if(!_swt_q_pop(s, f)) break;
      SwtAsyncSubmitSample(st, s, f);
      SwtAsyncTick(raw, st);
      did = true;
   }
   if(did)
   {
      gSwtSt = st;
      if(st.has)
        {
         gSwtLast0 = st.last0;
         gSwtLast1 = st.last1;
         gSwtHas = true;
        }
   }
   _swt_log_init_progress(timer_ms);
}

inline bool SwtFacadeDenoiseLast(const double &raw[],
                                 const int rates_total,
                                 const int n_window,
                                 const bool raw_ready,
                                 const int level_in,
                                 const bool useService,
                                 const bool submitOnNewBarOnly,
                                 const int initBatchSize,
                                 const bool showProgress,
                                 const bool isNewBar,
                                 double &out0, double &out1)
{
   int limit = MathMin(rates_total, n_window);
   if(limit < 8) return false;
   int len = _swt_prevpow2(limit);
   if(len < 8) return false;

   int maxLevel = PyWT_SwtMaxLevel(len);
   if(maxLevel < 1) return false;
   int level = level_in;
   if(level < 1) level = 1;
   if(level > maxLevel) level = maxLevel;

   if(useService)
   {
      if(ArraySize(raw) < len) return false;
      _swt_state_init();
      SwtAsyncState st; st = gSwtSt;
      if(!gSwtReady)
      {
         SwtAsyncReset(st);
         gSwtReady = true;
      }
      int chunk = initBatchSize;
      if(chunk < 1) chunk = 1;
      SwtAsyncSetChunkSize(st, chunk);
      SwtAsyncRequestInit(st, len, level);
      if(!submitOnNewBarOnly || isNewBar)
         _swt_q_push(raw[0], isNewBar ? 1 : 0);

      gSwtShow = showProgress;
      // retorna ultimo valor cacheado; o timer atualiza via SwtFacadeOnTimer
      gSwtSt = st;
      if(gSwtHas)
        {
         out0 = gSwtLast0;
         out1 = gSwtLast1;
         return true;
        }
      return false;
   }

   static PyWTDiscreteWavelet w;
   static bool w_ready = false;
   if(!w_ready)
   {
      if(!PyWT_DiscreteWavelet(PYWT_DB,4,w)) return false;
      w_ready = true;
   }

   double in[]; ArrayResize(in, len);
   for(int i=0;i<len;i++)
      in[i] = raw[len-1-i];

   double approx[];
   if(!PyWT_SWT_A(in, w, level, approx)) return false;

   out0 = approx[len-1];
   out1 = approx[len-2];
   return true;
}

#endif // __SPECTRAL_SWT_FACADE_MQH__
