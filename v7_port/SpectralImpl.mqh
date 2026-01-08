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

   // Batch segments on GPU (detrend + window + padding + FFT)
   static CLFFTPlan plan;
   if(!plan.ready) CLFFTReset(plan);
   if(!CLFFTLoadRealSegmentsDetrendBatch(plan,x,win,0,step,nperseg,nfft,detrend_type,nseg))
     { ArrayResize(out,0,0); return; }
   Complex64 specFlat[];
   if(!CLFFTExecuteBatchFromMemA(plan,nseg,specFlat,false))
     { ArrayResize(out,0,0); return; }
   for(int s=0;s<nseg;s++)
     {
      int base=s*nfft;
      if(sides==1)
        {
         for(int k=0;k<nfreq;k++) out[s][k]=specFlat[base + k];
        }
      else
        {
         for(int k=0;k<nfft;k++) out[s][k]=specFlat[base + k];
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
   for(int k=0;k<nfreq;k++)
     {
      if(return_onesided) freqs[k]=(double)k*fs/(double)nfft;
      else
        {
         int kk=(k<=nfft/2)?k:(k-nfft);
         freqs[k]=(double)kk*fs/(double)nfft;
        }
     }

   // times
   int nseg=ArrayRange(result,0);
   ArrayResize(t,nseg);
   double step=(double)(seglen-noverlap);
   for(int i=0;i<nseg;i++) t[i]=( (double)i*step + (double)seglen/2.0)/fs;
   if(boundary!="")
     {
      double shift=(double)seglen/2.0/fs;
      for(int i=0;i<nseg;i++) t[i]-=shift;
     }

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

// PSD/CSD helper using STFT-scaled spectra (sqrt scaling inside _spectral_helper_1d)
inline bool spectral_helper_psd_1d(const double &x_in[],const double &y_in[],const bool same_data,
                                   const double fs,const string window,int nperseg,int noverlap,int nfft,
                                   const int detrend_type,const bool return_onesided,
                                   const string scaling,const string boundary,const bool padded,
                                   double &freqs[],double &t[],Complex64 &result[][])
  {
   double x[]; ArrayCopy(x,x_in);
   double y[];
   if(!same_data) ArrayCopy(y,y_in);

   int Nx=ArraySize(x);
   int Ny=same_data?Nx:ArraySize(y);
   if(Nx<=0 || Ny<=0) { ArrayResize(result,0,0); return false; }

   // pad to equal length
   if(!same_data && Nx!=Ny)
     {
      int Nmax = (Nx>Ny?Nx:Ny);
      if(Nx<Nmax){ int old=Nx; ArrayResize(x,Nmax); for(int i=old;i<Nmax;i++) x[i]=0.0; Nx=Nmax; }
      if(Ny<Nmax){ int old=Ny; ArrayResize(y,Nmax); for(int i=old;i<Nmax;i++) y[i]=0.0; Ny=Nmax; }
     }

   Complex64 X[][];
   _spectral_helper_1d(x,x,fs,window,nperseg,noverlap,nfft,detrend_type,return_onesided,scaling,"stft",boundary,padded,freqs,t,X);
   if(ArrayRange(X,0)==0) { ArrayResize(result,0,0); return false; }
   Complex64 Y[][];
   if(!same_data)
     {
      double tfreqs[]; double tt[];
      _spectral_helper_1d(y,y,fs,window,nperseg,noverlap,nfft,detrend_type,return_onesided,scaling,"stft",boundary,padded,tfreqs,tt,Y);
      if(ArrayRange(Y,0)==0) { ArrayResize(result,0,0); return false; }
     }

   int nseg=ArrayRange(X,0);
   int nfreq=ArrayRange(X,1);
   ArrayResize(result,nseg);
   for(int s=0;s<nseg;s++) ArrayResize(result[s],nfreq);

   for(int s=0;s<nseg;s++)
     {
      for(int k=0;k<nfreq;k++)
        {
         Complex64 cx=CxConj(X[s][k]);
         Complex64 cy = same_data ? X[s][k] : Y[s][k];
         result[s][k]=CxMul(cx,cy);
        }
     }

   // onesided scaling for PSD/CSD
   if(return_onesided)
     {
      int Nfft = nfft;
      if(Nfft<=0) Nfft = nperseg;
      int last = (Nfft%2)? (nfreq-1) : (nfreq-2);
      for(int s=0;s<nseg;s++)
        {
         for(int k=1;k<=last;k++) result[s][k]=CxScale(result[s][k],2.0);
        }
     }

   return true;
  }

inline double _median_bias_mql(const int n)
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

inline void _median_real(double &vals[],double &med)
  {
   int n=ArraySize(vals);
   if(n<=0) { med=0.0; return; }
   ArraySort(vals,WHOLE_ARRAY,0,MODE_ASCEND);
   if(n%2) med=vals[n/2];
   else med=0.5*(vals[n/2-1]+vals[n/2]);
  }

inline bool csd_1d(const double &x[],const double &y[],const double fs,const string window,int nperseg,int noverlap,int nfft,
                   const int detrend_type,const bool return_onesided,const string scaling,const string average,
                   double &freqs[],Complex64 &Pxy[])
  {
   Complex64 Pseg[][];
   double t[];
   bool same_data=false;
   if(!spectral_helper_psd_1d(x,y,same_data,fs,window,nperseg,noverlap,nfft,detrend_type,return_onesided,scaling,"",false,freqs,t,Pseg))
     { ArrayResize(Pxy,0); return false; }
   int nseg=ArrayRange(Pseg,0);
   int nfreq=ArrayRange(Pseg,1);
   ArrayResize(Pxy,nfreq);
   if(nseg<=1)
     {
      for(int k=0;k<nfreq;k++) Pxy[k]=Pseg[0][k];
      return true;
     }

   string avg=StringToLower(average);
   if(avg=="mean")
     {
      for(int k=0;k<nfreq;k++)
        {
         double re=0.0, im=0.0;
         for(int s=0;s<nseg;s++){ re+=Pseg[s][k].re; im+=Pseg[s][k].im; }
         Pxy[k]=Cx(re/(double)nseg, im/(double)nseg);
        }
     }
   else if(avg=="median")
     {
      double bias=_median_bias_mql(nseg);
      double tmpRe[]; double tmpIm[];
      ArrayResize(tmpRe,nseg);
      ArrayResize(tmpIm,nseg);
      for(int k=0;k<nfreq;k++)
        {
         for(int s=0;s<nseg;s++){ tmpRe[s]=Pseg[s][k].re; tmpIm[s]=Pseg[s][k].im; }
         double medRe, medIm;
         _median_real(tmpRe,medRe);
         _median_real(tmpIm,medIm);
         Pxy[k]=Cx(medRe/bias, medIm/bias);
        }
     }
   else return false;

   return true;
  }

inline bool welch_1d(const double &x[],const double fs,const string window,int nperseg,int noverlap,int nfft,
                     const int detrend_type,const bool return_onesided,const string scaling,const string average,
                     double &freqs[],double &Pxx[])
  {
   Complex64 Pxy[];
   if(!csd_1d(x,x,fs,window,nperseg,noverlap,nfft,detrend_type,return_onesided,scaling,average,freqs,Pxy))
     { ArrayResize(Pxx,0); return false; }
   int nfreq=ArraySize(Pxy);
   ArrayResize(Pxx,nfreq);
   for(int k=0;k<nfreq;k++) Pxx[k]=Pxy[k].re;
   return true;
  }

inline bool periodogram_1d(const double &x[],const double fs,const string window,int nfft,
                           const int detrend_type,const bool return_onesided,const string scaling,
                           double &freqs[],double &Pxx[])
  {
   int N=ArraySize(x);
   if(N<=0) { ArrayResize(Pxx,0); return false; }
   int nperseg = (nfft>0? MathMin(nfft,N) : N);
   return welch_1d(x,fs,window,nperseg,0,nfft,detrend_type,return_onesided,scaling,"mean",freqs,Pxx);
  }

inline bool stft_1d(const double &x[],const double fs,const string window,int nperseg,int noverlap,int nfft,
                    const int detrend_type,const bool return_onesided,const string scaling,
                    const string boundary,const bool padded,
                    double &freqs[],double &t[],Complex64 &Zxx[][])
  {
   _spectral_helper_1d(x,x,fs,window,nperseg,noverlap,nfft,detrend_type,return_onesided,scaling,"stft",boundary,padded,freqs,t,Zxx);
   return (ArrayRange(Zxx,0)>0);
  }

inline bool spectrogram_1d(const double &x[],const double fs,const string window,int nperseg,int noverlap,int nfft,
                           const int detrend_type,const bool return_onesided,const string scaling,
                           const string mode,
                           double &freqs[],double &t[],double &Sxx[][])
  {
   string m=StringToLower(mode);
   if(m=="psd")
     {
      Complex64 Pxy[][];
      if(!spectral_helper_psd_1d(x,x,true,fs,window,nperseg,noverlap,nfft,detrend_type,return_onesided,scaling,"",false,freqs,t,Pxy))
        { ArrayResize(Sxx,0,0); return false; }
      int nseg=ArrayRange(Pxy,0);
      int nfreq=ArrayRange(Pxy,1);
      ArrayResize(Sxx,nseg);
      for(int s=0;s<nseg;s++)
        {
         ArrayResize(Sxx[s],nfreq);
         for(int k=0;k<nfreq;k++) Sxx[s][k]=Pxy[s][k].re;
        }
      return true;
     }
   Complex64 Zxx[][];
   if(!stft_1d(x,fs,window,nperseg,noverlap,nfft,detrend_type,return_onesided,scaling,"",false,freqs,t,Zxx))
     { ArrayResize(Sxx,0,0); return false; }
   int nseg=ArrayRange(Zxx,0);
   int nfreq=ArrayRange(Zxx,1);
   ArrayResize(Sxx,nseg);
   for(int s=0;s<nseg;s++) ArrayResize(Sxx[s],nfreq);
   if(m=="complex") return false;
   if(m=="magnitude")
     {
      for(int s=0;s<nseg;s++) for(int k=0;k<nfreq;k++) Sxx[s][k]=CxAbs(Zxx[s][k]);
      return true;
     }
   if(m=="angle" || m=="phase")
     {
      for(int s=0;s<nseg;s++)
        {
         for(int k=0;k<nfreq;k++) Sxx[s][k]=MathArctan2(Zxx[s][k].im,Zxx[s][k].re);
         if(m=="phase")
           {
            // unwrap along frequency axis
            double prev=Sxx[s][0];
            for(int k=1;k<nfreq;k++)
              {
               double v=Sxx[s][k];
               double dp=v-prev;
               while(dp>PI) { v-=2.0*PI; dp=v-prev; }
               while(dp<-PI) { v+=2.0*PI; dp=v-prev; }
               Sxx[s][k]=v;
               prev=v;
              }
           }
        }
      return true;
     }
   return false;
  }

inline bool spectrogram_complex_1d(const double &x[],const double fs,const string window,int nperseg,int noverlap,int nfft,
                                   const int detrend_type,const bool return_onesided,const string scaling,
                                   double &freqs[],double &t[],Complex64 &Sxx[][])
  {
   return stft_1d(x,fs,window,nperseg,noverlap,nfft,detrend_type,return_onesided,scaling,"",false,freqs,t,Sxx);
  }

inline bool coherence_1d(const double &x[],const double &y[],const double fs,const string window,int nperseg,int noverlap,int nfft,
                         const int detrend_type,const bool return_onesided,const string scaling,const string average,
                         double &freqs[],double &Cxy[])
  {
   Complex64 Pxy[];
   if(!csd_1d(x,y,fs,window,nperseg,noverlap,nfft,detrend_type,return_onesided,scaling,average,freqs,Pxy))
     { ArrayResize(Cxy,0); return false; }
   double Pxx[], Pyy[], f2[];
   if(!welch_1d(x,fs,window,nperseg,noverlap,nfft,detrend_type,return_onesided,scaling,average,f2,Pxx))
     { ArrayResize(Cxy,0); return false; }
   if(!welch_1d(y,fs,window,nperseg,noverlap,nfft,detrend_type,return_onesided,scaling,average,f2,Pyy))
     { ArrayResize(Cxy,0); return false; }
   int nfreq=ArraySize(Pxy);
   ArrayResize(Cxy,nfreq);
   for(int k=0;k<nfreq;k++)
     {
      double denom=Pxx[k]*Pyy[k];
      if(denom<=0.0) Cxy[k]=0.0;
      else
        {
         double num=CxAbs(Pxy[k]);
         Cxy[k]=(num*num)/denom;
        }
     }
   return true;
  }

inline bool istft_1d_complex(const Complex64 &Zxx[][],const double fs,const string window,int nperseg,int noverlap,int nfft,
                             const bool input_onesided,const bool boundary,const string scaling,
                             double &time[],Complex64 &x[])
  {
   int nseg=ArrayRange(Zxx,0);
   int nfreq=ArrayRange(Zxx,1);
   if(nseg<=0 || nfreq<=0) return false;

   int n_default = input_onesided ? 2*(nfreq-1) : nfreq;
   if(nperseg<=0) nperseg=n_default;
   if(nperseg<1) return false;

   if(nfft<=0)
     {
      if(input_onesided && nperseg==n_default+1) nfft=nperseg;
      else nfft=n_default;
     }
   if(nfft<nperseg) return false;

   if(noverlap<0) noverlap=nperseg/2;
   if(noverlap>=nperseg) return false;
   int nstep=nperseg-noverlap;

   if(input_onesided)
     {
      int expected = (nfft%2==0) ? (nfft/2+1) : ((nfft+1)/2);
      if(nfreq!=expected) return false;
     }
   else
     {
      if(nfreq!=nfft) return false;
     }

   double win[];
   CLGetWindow(window,nperseg,true,win);
   double wsum=0.0, wsum2=0.0;
   for(int i=0;i<nperseg;i++){ wsum+=win[i]; wsum2+=win[i]*win[i]; }
   double scale=1.0;
   if(scaling=="spectrum") scale=wsum;
   else if(scaling=="psd") scale=MathSqrt(fs*wsum2);
   else return false;

   // build full spectra (batch)
   int N=nfft;
   int batch=nseg;
   Complex64 inFlat[];
   ArrayResize(inFlat,batch*N);
   for(int s=0;s<batch;s++)
     {
      int base=s*N;
      for(int k=0;k<N;k++) inFlat[base+k]=Cx(0.0,0.0);
      if(input_onesided)
        {
         for(int k=0;k<nfreq;k++) inFlat[base+k]=Zxx[s][k];
         int kmax = (N%2==0)? (nfreq-2) : (nfreq-1);
         for(int k=1;k<=kmax;k++)
           inFlat[base + (N-k)] = CxConj(Zxx[s][k]);
        }
      else
        {
         if(nfreq!=N) return false;
         for(int k=0;k<N;k++) inFlat[base+k]=Zxx[s][k];
        }
     }

   static CLFFTPlan plan;
   if(!plan.ready) CLFFTReset(plan);
   if(!CLFFTInit(plan,N)) return false;
   if(!CLFFTUploadComplexBatch(plan,inFlat,batch)) return false;
   Complex64 outFlat[];
   if(!CLFFTExecuteBatchFromMemA(plan,batch,outFlat,true)) return false;

   int outlen = nperseg + (batch-1)*nstep;
   ArrayResize(x,outlen);
   double norm[];
   ArrayResize(norm,outlen);
   for(int i=0;i<outlen;i++){ x[i]=Cx(0.0,0.0); norm[i]=0.0; }

   for(int s=0;s<batch;s++)
     {
      int start=s*nstep;
      int base=s*N;
      for(int i=0;i<nperseg;i++)
        {
         double wr=win[i];
         Complex64 v=outFlat[base+i];
         v.re*=scale; v.im*=scale;
         x[start+i]=CxAdd(x[start+i],Cx(v.re*wr,v.im*wr));
         norm[start+i]+=wr*wr;
        }
     }

   if(boundary)
     {
      int cut=nperseg/2;
      int newlen=outlen-2*cut;
      if(newlen<=0) return false;
      Complex64 xtmp[]; ArrayResize(xtmp,newlen);
      double ntmp[]; ArrayResize(ntmp,newlen);
      for(int i=0;i<newlen;i++){ xtmp[i]=x[i+cut]; ntmp[i]=norm[i+cut]; }
      ArrayResize(x,newlen);
      ArrayResize(norm,newlen);
      for(int i=0;i<newlen;i++){ x[i]=xtmp[i]; norm[i]=ntmp[i]; }
      outlen=newlen;
     }

   for(int i=0;i<outlen;i++)
     {
      double d=norm[i];
      if(d>1e-10){ x[i].re/=d; x[i].im/=d; }
     }

   ArrayResize(time,outlen);
   for(int i=0;i<outlen;i++) time[i]=(double)i/fs;
   return true;
  }

inline bool istft_1d_real(const Complex64 &Zxx[][],const double fs,const string window,int nperseg,int noverlap,int nfft,
                          const bool input_onesided,const bool boundary,const string scaling,
                          double &time[],double &x[])
  {
   Complex64 xc[];
   if(!istft_1d_complex(Zxx,fs,window,nperseg,noverlap,nfft,input_onesided,boundary,scaling,time,xc))
     { ArrayResize(x,0); return false; }
   int n=ArraySize(xc);
   ArrayResize(x,n);
   for(int i=0;i<n;i++) x[i]=xc[i].re;
   return true;
  }

#endif
