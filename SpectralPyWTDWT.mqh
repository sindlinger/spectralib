#ifndef __SPECTRAL_PYWT_DWT_MQH__
#define __SPECTRAL_PYWT_DWT_MQH__

#include "SpectralPyWTCommon.mqh"
#include "SpectralPyWTWavelets.mqh"
#include "SpectralPyWTConvolution.mqh"

inline bool PyWT_DecA(const double &in[],const PyWTDiscreteWavelet &w,const int mode,double &output[])
  {
   return PyWT_DownsamplingConvolution(in,w.dec_lo,2,mode,output);
  }

inline bool PyWT_DecD(const double &in[],const PyWTDiscreteWavelet &w,const int mode,double &output[])
  {
   return PyWT_DownsamplingConvolution(in,w.dec_hi,2,mode,output);
  }

inline bool PyWT_RecA(const double &coeffs[],const PyWTDiscreteWavelet &w,double &output[])
  {
   return PyWT_UpsamplingConvolutionFull(coeffs,w.rec_lo,output);
  }

inline bool PyWT_RecD(const double &coeffs[],const PyWTDiscreteWavelet &w,double &output[])
  {
   return PyWT_UpsamplingConvolutionFull(coeffs,w.rec_hi,output);
  }

inline bool PyWT_IDWT(const double &coeffs_a[],const double &coeffs_d[],const PyWTDiscreteWavelet &w,const int mode,double &output[])
  {
   int hasA = (ArraySize(coeffs_a) > 0);
   int hasD = (ArraySize(coeffs_d) > 0);
   if(!hasA && !hasD) return false;
   int input_len = hasA ? ArraySize(coeffs_a) : ArraySize(coeffs_d);
   if(hasA && hasD && ArraySize(coeffs_a) != ArraySize(coeffs_d)) return false;
   int out_len = (mode==PYWT_MODE_PERIODIZATION) ? (2*input_len) : (2*input_len - w.rec_len + 2);
   ArrayResize(output,out_len);
   for(int i=0;i<out_len;i++) output[i]=0.0;

   double tmp[];
   if(hasA)
     {
      if(!PyWT_UpsamplingConvolutionValidSF(coeffs_a,w.rec_lo,mode,tmp)) return false;
      int L=ArraySize(tmp);
      if(L!=out_len) return false;
      for(int i=0;i<out_len;i++) output[i] += tmp[i];
     }
   if(hasD)
     {
      if(!PyWT_UpsamplingConvolutionValidSF(coeffs_d,w.rec_hi,mode,tmp)) return false;
      int L=ArraySize(tmp);
      if(L!=out_len) return false;
      for(int i=0;i<out_len;i++) output[i] += tmp[i];
     }
   return true;
  }

inline bool PyWT_SWT_A(const double &in[],const PyWTDiscreteWavelet &w,const int level,double &output[])
  {
   int N=ArraySize(in);
   if(level < 1) return false;
   if(level > PyWT_SwtMaxLevel(N)) return false;
   if(level==1)
      return PyWT_DownsamplingConvolutionPeriodization(in,w.dec_lo,1,1,output);

   int fstep = 1 << (level-1);
   int F = w.dec_len << (level-1);
   double ef[]; ArrayResize(ef,F);
   for(int i=0;i<F;i++) ef[i]=0.0;
   for(int i=0;i<w.dec_len;i++)
      ef[i*fstep] = w.dec_lo[i];
   return PyWT_DownsamplingConvolutionPeriodization(in,ef,1,1,output);
  }

inline bool PyWT_SWT_D(const double &in[],const PyWTDiscreteWavelet &w,const int level,double &output[])
  {
   int N=ArraySize(in);
   if(level < 1) return false;
   if(level > PyWT_SwtMaxLevel(N)) return false;
   if(level==1)
      return PyWT_DownsamplingConvolutionPeriodization(in,w.dec_hi,1,1,output);

   int fstep = 1 << (level-1);
   int F = w.dec_len << (level-1);
   double ef[]; ArrayResize(ef,F);
   for(int i=0;i<F;i++) ef[i]=0.0;
   for(int i=0;i<w.dec_len;i++)
      ef[i*fstep] = w.dec_hi[i];
   return PyWT_DownsamplingConvolutionPeriodization(in,ef,1,1,output);
  }

#endif // __SPECTRAL_PYWT_DWT_MQH__
