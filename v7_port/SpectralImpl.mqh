#ifndef __SPECTRAL_IMPL_MQH__
#define __SPECTRAL_IMPL_MQH__

#include "SpectralCommon.mqh"
#include "SpectralArrayTools.mqh"
#include "SpectralSignalTools.mqh"
#include "SpectralOpenCLWindows.mqh"
#include "SpectralOpenCLFFT.mqh"
#include "SpectralOpenCL.mqh"

// Lomb-Scargle periodogram (OpenCL float64 only, no CPU fallback)
inline void lombscargle(const double &x[],const double &y[],const double &freqs[],
                        const bool precenter,const bool normalize,double &pgram[])
  {
   int N=ArraySize(x);
   int F=ArraySize(freqs);
   ArrayResize(pgram,F);
   if(N<=0 || F<=0) return;
   double y_in[];
   ArrayResize(y_in,N);
   double mean=0.0;
   if(precenter)
     {
      for(int i=0;i<N;i++) mean+=y[i];
      mean/=N;
     }
   for(int i=0;i<N;i++) y_in[i]=precenter ? (y[i]-mean) : y[i];

   static CLHandle clh;
   if(!clh.ready) CLReset(clh);
   if(!CLLombscargle(clh,x,y_in,freqs,normalize,pgram))
     {
      ArrayResize(pgram,0);
      return;
     }
  }

// median bias
inline double _median_bias(const int n)
  {
   if(n<=1) return 1.0;
   double sum=0.0;
   for(int k=1;k<=((n-1)/2);k++)
     {
      double ii2=2.0*k;
      sum += 1.0/(ii2+1.0) - 1.0/ii2;
     }
   return 1.0 + sum;
  }

// triage_segments: returns window array and nperseg
inline void _triage_segments(const string window,const int input_length,int &nperseg,double &win[])
  {
   if(nperseg<=0) nperseg=256;
   if(nperseg>input_length) nperseg=input_length;
   CLGetWindow(window,nperseg,true,win);
  }

// FFT helper for 1D x (returns 2D array: segments x nfft_complex)
inline void _fft_helper_1d(const double &x[],const double &win[],const int nperseg,
                           const int noverlap,const int nfft,const int sides,
                           const int detrend_type,Complex64 &out[][])
  {
   int N=ArraySize(x);
   int step=nperseg-noverlap;
   int nseg;
   if(nperseg==1 && noverlap==0) nseg=N;
   else nseg=(N-noverlap)/step;
   if(nseg<1) { ArrayResize(out,0,0); return; }

   ArrayResize(out,nseg);
   int nfreq = (sides==1) ? (nfft/2+1) : nfft;
   for(int s=0;s<nseg;s++) ArrayResize(out[s],nfreq);

   // For each segment
   static CLFFTPlan plan;
   if(!plan.ready) CLFFTReset(plan);
   double seg[];
   ArrayResize(seg,nperseg);
   for(int s=0;s<nseg;s++)
     {
      int start=s*step;
      for(int i=0;i<nperseg;i++) seg[i]=x[start+i];
      // detrend
      if(detrend_type!=DETREND_NONE)
        {
         // use 2D detrend helper by wrapping single segment
         double tmp[1][];
         ArrayResize(tmp,1);
         ArrayResize(tmp[0],nperseg);
         for(int i=0;i<nperseg;i++) tmp[0][i]=seg[i];
         detrend_segments(tmp,detrend_type);
         for(int i=0;i<nperseg;i++) seg[i]=tmp[0][i];
        }
      // apply window + zero-pad on GPU, then FFT (OpenCL float64)
      Complex64 spec[];
      if(!CLFFTLoadRealSegment(plan,seg,win,0,nperseg,nfft)) { ArrayResize(out,0,0); return; }
      if(!CLFFTExecuteFromMemA(plan,spec,false)) { ArrayResize(out,0,0); return; }
      if(sides==1)
        {
         for(int k=0;k<nfreq;k++) out[s][k]=spec[k];
        }
      else
        {
         for(int k=0;k<nfft;k++) out[s][k]=spec[k];
        }
     }
  }

// spectral helper (1D)
inline void _spectral_helper_1d(const double &x_in[],const double &y_in[],
                                const double fs,const string window,
                                int nperseg,int noverlap,int nfft,
                                const int detrend_type,const bool return_onesided,
                                const string scaling,const string mode,
                                const string boundary,const bool padded,
                                double &freqs[],double &t[],Complex64 &result[][])
  {
   // boundary extension
   double x[];
   ArrayCopy(x,x_in);
   if(boundary!="")
     {
      int nedge = nperseg/2;
      double xext[];
      if(boundary=="even") even_ext(x,nedge,xext);
      else if(boundary=="odd") odd_ext(x,nedge,xext);
      else if(boundary=="constant") const_ext(x,nedge,xext);
      else if(boundary=="zeros") zero_ext(x,nedge,xext);
      else xext=x;
      x=xext;
     }

   int N=ArraySize(x);
   _triage_segments(window,N,nperseg,freqs); // reuse freqs array as temp for win
   double win[];
   win=freqs; // window output from triage
   int seglen=nperseg;

   if(noverlap<0) noverlap=seglen/2;
   if(noverlap>=seglen) noverlap=seglen-1;
   if(nfft<=0) nfft=seglen;
   if(nfft<seglen) nfft=seglen;

   // padding to fit integer number of segments
   if(padded)
     {
      int step=seglen-noverlap;
      int nseg = (N-noverlap+step-1)/step;
      int total = nseg*step + noverlap;
      if(total>N)
        {
         int add=total-N;
         int oldN=N;
         ArrayResize(x,oldN+add);
         for(int i=0;i<add;i++) x[oldN+i]=0.0;
         N=ArraySize(x);
        }
     }

   int sides = return_onesided?1:0; // 1 for onesided
   _fft_helper_1d(x,win,seglen,noverlap,nfft,return_onesided?1:0,detrend_type,result);

   // freqs
   int nfreq = return_onesided ? (nfft/2+1) : nfft;
   ArrayResize(freqs,nfreq);
   for(int k=0;k<nfreq;k++) freqs[k]= (double)k*fs/(double)nfft;

   // times
   int nseg=ArrayRange(result,0);
   ArrayResize(t,nseg);
   double step=(double)(seglen-noverlap);
   for(int i=0;i<nseg;i++) t[i]=( (double)i*step + (double)seglen/2.0)/fs;

   // scaling
   if(scaling=="density")
     {
      double winpow=0.0;
      for(int i=0;i<seglen;i++) winpow+=win[i]*win[i];
      double scale=1.0/(fs*winpow);
      for(int s=0;s<nseg;s++)
        for(int k=0;k<nfreq;k++)
           result[s][k]=CxScale(result[s][k],MathSqrt(scale));
     }
   else if(scaling=="spectrum")
     {
      double wsum=0.0;
      for(int i=0;i<seglen;i++) wsum+=win[i];
      double scale=1.0/wsum;
      for(int s=0;s<nseg;s++)
        for(int k=0;k<nfreq;k++)
           result[s][k]=CxScale(result[s][k],scale);
     }
  }

#endif
