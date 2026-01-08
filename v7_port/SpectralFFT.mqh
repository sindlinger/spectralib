#ifndef __SPECTRAL_FFT_MQH__
#define __SPECTRAL_FFT_MQH__

#include "SpectralCommon.mqh"

// Direct DFT/IDFT (float64). Used for window functions (chebwin/taylor).
inline void DFT(const Complex64 &in[],Complex64 &out[],const bool inverse)
  {
   int N=ArraySize(in);
   ArrayResize(out,N);
   double sign=inverse?1.0:-1.0;
   for(int k=0;k<N;k++)
     {
      double re=0.0, im=0.0;
      for(int n=0;n<N;n++)
        {
         double ang=2.0*PI*(double)k*(double)n/(double)N;
         double cs=MathCos(ang);
         double sn=MathSin(ang)*sign;
         re += in[n].re*cs - in[n].im*sn;
         im += in[n].re*sn + in[n].im*cs;
        }
      if(inverse)
        { re/=N; im/=N; }
      out[k]=Cx(re,im);
     }
  }

inline void FFTReal(const double &in[],Complex64 &out[])
  {
   int N=ArraySize(in);
   Complex64 tmp[];
   ArrayResize(tmp,N);
   for(int i=0;i<N;i++) tmp[i]=Cx(in[i],0.0);
   DFT(tmp,out,false);
  }

#endif
