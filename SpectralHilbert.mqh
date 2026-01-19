#ifndef __SPECTRAL_HILBERT_MQH__
#define __SPECTRAL_HILBERT_MQH__

#include "SpectralCommon.mqh"
#include "SpectralOpenCLFFT.mqh"

inline void _pack_real_to_cplx(const double &x[],double &buf[])
  {
   int N=ArraySize(x);
   ArrayResize(buf,2*N);
   for(int i=0;i<N;i++)
     {
      buf[2*i]=x[i];
      buf[2*i+1]=0.0;
     }
  }

inline bool hilbert_analytic_gpu(const double &x[],Complex64 &out[])
  {
   int N=ArraySize(x);
   if(N<=0) return false;

   static CLFFTPlan plan;
   if(!plan.ready) CLFFTReset(plan);
   if(!CLFFTInit(plan,N)) return false;
   if(plan.batch!=1)
     {
      if(!CLFFTEnsureBatchBuffers(plan,1)) return false;
     }

   double buf[];
   _pack_real_to_cplx(x,buf);
   CLBufferWrite(plan.memA,buf);

   if(!CLFFTExecuteFromMemA_NoRead(plan,false)) return false;
   if(!CLFFTHilbertMask(plan,plan.memFinal,N)) return false;
   if(plan.memFinal!=plan.memA)
     {
      if(!CLFFTCopyCplx(plan,plan.memFinal,plan.memA,N)) return false;
     }
   if(!CLFFTExecuteFromMemA_NoRead(plan,true)) return false;

   double outbuf[];
   ArrayResize(outbuf,2*N);
   CLBufferRead(plan.memFinal,outbuf);
   ArrayResize(out,N);
   for(int i=0;i<N;i++)
     out[i]=Cx(outbuf[2*i],outbuf[2*i+1]);
   return true;
  }

inline bool hilbert_imag_gpu(const double &x[],double &y[])
  {
   Complex64 tmp[];
   if(!hilbert_analytic_gpu(x,tmp)) return false;
   int N=ArraySize(tmp);
   ArrayResize(y,N);
   for(int i=0;i<N;i++) y[i]=tmp[i].im;
   return true;
  }

#endif
