//+------------------------------------------------------------------+
//| SpectralSwtStreamClient.mqh                                      |
//| Wrapper limpo para uso do serviço SWT via pipe (async/chunked)   |
//+------------------------------------------------------------------+
#ifndef __SPECTRAL_SWT_STREAM_CLIENT_MQH__
#define __SPECTRAL_SWT_STREAM_CLIENT_MQH__

#include <spectralib/SpectralServiceBridge.mqh>

struct SwtAsyncState
{
   bool init_ready;
   bool init_in_progress;
   bool want_init;
   int  want_len;
   int  want_level;
   int  len;
   int  level;
   int  chunk_pos;
   int  chunk_size;
   double last0;
   double last1;
   bool has;
   bool pending;
   double pending_sample;
   int pending_newbar;
};

inline void SwtAsyncReset(SwtAsyncState &st)
{
   st.init_ready = false;
   st.init_in_progress = false;
   st.want_init = false;
   st.want_len = 0;
   st.want_level = 0;
   st.len = 0;
   st.level = 0;
   st.chunk_pos = 0;
   st.chunk_size = 2048;
   st.last0 = 0.0;
   st.last1 = 0.0;
   st.has = false;
   st.pending = false;
   st.pending_sample = 0.0;
   st.pending_newbar = 0;
}

inline void SwtAsyncSetChunkSize(SwtAsyncState &st, const int chunk_size)
{
   if(chunk_size > 0) st.chunk_size = chunk_size;
}

inline void SwtAsyncRequestInit(SwtAsyncState &st, const int len, const int level)
{
   if(len < 8) return;
   if(!st.init_ready || st.len != len || st.level != level)
   {
      st.want_init = true;
      st.want_len = len;
      st.want_level = level;
   }
}

inline void SwtAsyncSubmitSample(SwtAsyncState &st, const double sample, const int is_new_bar)
{
   st.pending = true;
   st.pending_sample = sample;
   st.pending_newbar = is_new_bar;
}

inline bool SwtAsyncGetLast(SwtAsyncState &st, double &out0, double &out1)
{
   if(!st.has) return false;
   out0 = st.last0;
   out1 = st.last1;
   return true;
}

inline void SwtAsyncTick(const double &series[], SwtAsyncState &st)
{
   // start init if requested
   if(st.want_init && !st.init_in_progress)
   {
      if(SpectralPipeInitBegin(st.want_len, st.want_level, 2))
      {
         st.init_in_progress = true;
         st.init_ready = false;
         st.len = st.want_len;
         st.level = st.want_level;
         st.chunk_pos = 0;
         st.want_init = false;
      }
      else
      {
         return;
      }
   }

   // chunked init upload
   if(st.init_in_progress)
   {
      int remain = st.len - st.chunk_pos;
      if(remain <= 0)
      {
         double out0=0.0, out1=0.0;
         if(SpectralPipeInitEnd(out0, out1, 5))
         {
            st.last0 = out0;
            st.last1 = out1;
            st.has = true;
            st.init_ready = true;
            st.init_in_progress = false;
         }
         return;
      }

      int n = MathMin(st.chunk_size, remain);
      double chunk[]; ArrayResize(chunk, n);
      for(int i=0;i<n;i++)
      {
         int idx = st.len - 1 - (st.chunk_pos + i); // oldest -> newest
         chunk[i] = series[idx];
      }

      if(SpectralPipeInitChunk(chunk, n, 2))
         st.chunk_pos += n;
      return;
   }

   // regular push/get on timer
   if(st.pending && st.init_ready)
   {
      SpectralPipePushSample(st.pending_sample, st.level, st.pending_newbar, 0);
      double g0=0.0, g1=0.0;
      if(SpectralPipeGetLast(g0, g1, 0))
      {
         st.last0 = g0;
         st.last1 = g1;
         st.has = true;
      }
      st.pending = false;
   }
}

#endif // __SPECTRAL_SWT_STREAM_CLIENT_MQH__
