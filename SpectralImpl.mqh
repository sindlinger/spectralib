#ifndef __SPECTRAL_IMPL_MQH__
#define __SPECTRAL_IMPL_MQH__

#include "SpectralCommon.mqh"
#include "SpectralConfig.mqh"
#include "SpectralArrayTools.mqh"
#include "SpectralSignalTools.mqh"
#include "SpectralOpenCLWindows.mqh"
#include "SpectralOpenCLFFT.mqh"
#include "SpectralOpenCL.mqh"

inline void _spectral_log_fail(const string where)
  {
   static string last="";
   if(last==where) return;
   last=where;
   PrintFormat("spectralib: %s failed (err=%d)", where, GetLastError());
  }

static CLFFTPlan gSpectralPlan;
static bool gSpectralPlanInit=false;
static CLHandle gSpectralCLH;
static bool gSpectralCLHInit=false;

inline void _spectral_plan_ensure()
  {
   if(!gSpectralPlanInit)
     {
      CLFFTReset(gSpectralPlan);
      gSpectralPlanInit=true;
     }
  }

inline void _spectral_clh_ensure()
  {
   if(!gSpectralCLHInit)
     {
      CLReset(gSpectralCLH);
      gSpectralCLHInit=true;
     }
  }

inline void spectral_cl_cleanup()
  {
   if(gSpectralPlanInit)
     {
      CLFFTFree(gSpectralPlan);
      CLFFTReset(gSpectralPlan);
      gSpectralPlanInit=false;
     }
   if(gSpectralCLHInit)
     {
      CLFree(gSpectralCLH);
      CLReset(gSpectralCLH);
      gSpectralCLHInit=false;
     }
  }

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

   _spectral_clh_ensure();
   if(!CLLombscargle(gSpectralCLH,x,y_in,freqs,normalize,pgram))
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
inline void _fft_helper_1d(CLFFTPlan &plan,const double &x[],const double &win[],const int nperseg,
                           const int noverlap,const int nfft,const int sides,
                           const int detrend_type,const int boundary_type,const int nedge,const int ext_valid,
                           const int nseg,const int scaling_mode,const double fs,Complex64 &out[][])
  {
   int step=nperseg-noverlap;
   if(nseg<1) { ArrayResize(out,0,0); return; }

   int nfreq = (sides==1) ? (nfft/2+1) : nfft;
   ArrayResize(out,nseg,nfreq);

   // Batch segments on GPU (detrend + window + padding + FFT)
   if(!CLFFTLoadRealSegmentsDetrendBatch(plan,x,win,0,step,nperseg,nfft,detrend_type,nseg,boundary_type,nedge,ext_valid))
     { ArrayResize(out,0,0); return; }
   if(!CLFFTExecuteBatchFromMemA_NoRead(plan,nseg,false))
     { ArrayResize(out,0,0); return; }
   if(scaling_mode!=0)
     {
      double wsum=0.0, winpow=0.0;
      if(!CLFFTComputeWinStats(plan,wsum,winpow)) { ArrayResize(out,0,0); return; }
      double scale=1.0;
      if(scaling_mode==1)
        {
         if(winpow>0.0) scale=MathSqrt(1.0/(fs*winpow));
        }
      else if(scaling_mode==2)
        {
         if(wsum!=0.0) scale=1.0/wsum;
        }
      if(scale!=1.0)
        {
         if(!CLFFTScaleBatchFromFinal(plan,nseg,scale)) { ArrayResize(out,0,0); return; }
        }
     }
   int total=nseg*nfft;
   double buf[]; ArrayResize(buf,2*total);
   CLBufferRead(plan.memFinal,buf);
   Complex64 specFlat[]; ArrayResize(specFlat,total);
   for(int i=0;i<total;i++) specFlat[i]=Cx(buf[2*i],buf[2*i+1]);
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

// FFT helper for 1D x (GPU only; leaves result in plan.memFinal)
inline bool _fft_helper_1d_mem(CLFFTPlan &plan,const double &x[],const double &win[],const int nperseg,
                               const int noverlap,const int nfft,const int sides,
                               const int detrend_type,const int boundary_type,const int nedge,const int ext_valid,
                               const int nseg,const int scaling_mode,const double fs)
  {
   int step=nperseg-noverlap;
   if(nseg<1) return false;

   if(!CLFFTLoadRealSegmentsDetrendBatch(plan,x,win,0,step,nperseg,nfft,detrend_type,nseg,boundary_type,nedge,ext_valid))
     return false;
   if(!CLFFTExecuteBatchFromMemA_NoRead(plan,nseg,false))
     return false;
   if(scaling_mode!=0)
     {
      double wsum=0.0, winpow=0.0;
      if(!CLFFTComputeWinStats(plan,wsum,winpow)) return false;
      double scale=1.0;
      if(scaling_mode==1)
        {
         if(winpow>0.0) scale=MathSqrt(1.0/(fs*winpow));
        }
      else if(scaling_mode==2)
        {
         if(wsum!=0.0) scale=1.0/wsum;
        }
      if(scale!=1.0)
        {
         if(!CLFFTScaleBatchFromFinal(plan,nseg,scale)) return false;
        }
     }
   return true;
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
   // boundary handling (GPU)
   double x[];
   ArrayCopy(x,x_in);
   int Norig=ArraySize(x);
   if(Norig<=0) { ArrayResize(result,0,0); return; }
   int boundary_type=0;
   if(boundary=="even") boundary_type=1;
   else if(boundary=="odd") boundary_type=2;
   else if(boundary=="constant") boundary_type=3;
   else if(boundary=="zeros") boundary_type=4;
   int nedge=(boundary_type==0?0:(nperseg/2));
   if(nedge<0) nedge=0;
   if(nedge>Norig-1) nedge=Norig-1;
   int ext_valid = Norig + 2*nedge;
   int N=ext_valid;
   double win[];
   _triage_segments(window,N,nperseg,win);
   int seglen=nperseg;

   if(noverlap<0) noverlap=seglen/2;
   if(noverlap>=seglen) noverlap=seglen-1;
   if(nfft<=0) nfft=seglen;
   if(nfft<seglen) nfft=seglen;

   // effective length (boundary + optional padding)
   int N_eff=ext_valid;
   int step=seglen-noverlap;
   if(padded)
     {
      int nseg_pad = (N_eff-noverlap+step-1)/step;
      int total = nseg_pad*step + noverlap;
      if(total>N_eff) N_eff=total;
     }

   int nseg;
   if(seglen==1 && noverlap==0) nseg=N_eff;
   else nseg=(N_eff-noverlap)/step;

   int scaling_mode=0;
   if(scaling=="density") scaling_mode=1;
   else if(scaling=="spectrum") scaling_mode=2;

   _spectral_plan_ensure();
   _fft_helper_1d(gSpectralPlan,x,win,seglen,noverlap,nfft,return_onesided?1:0,detrend_type,
                  boundary_type,nedge,ext_valid,nseg,scaling_mode,fs,result);

   // freqs
   if(!CLFFTGenerateFreqs(gSpectralPlan,nfft,fs,return_onesided,freqs))
      { ArrayResize(freqs,0); }

   // times
   nseg=ArrayRange(result,0);
   if(!CLFFTGenerateTimes(gSpectralPlan,nseg,seglen,noverlap,fs,boundary_type,t))
      { ArrayResize(t,0); }

   // scaling applied on GPU
  }

// PSD/CSD helper using STFT-scaled spectra (sqrt scaling inside _spectral_helper_1d)
inline bool spectral_helper_psd_1d(const double &x_in[],const double &y_in[],const bool same_data,
                                   const double fs,const string window,int nperseg,int noverlap,int nfft,
                                   const int detrend_type,const bool return_onesided,
                                   const string scaling,const string boundary,const bool padded,
                                   double &freqs[],double &t[],double &result[][])
  {
   // GPU path: compute PSD/CSD segments in OpenCL and read back
   _spectral_plan_ensure();
#define plan gSpectralPlan

   // prepare input
   double x[]; ArrayCopy(x,x_in);
   double y[];
   if(!same_data) ArrayCopy(y,y_in);
   int Nx=ArraySize(x);
   int Ny=same_data?Nx:ArraySize(y);
   if(Nx<=0 || Ny<=0) { ArrayResize(result,0,0); return false; }
   if(!same_data && Nx!=Ny)
     {
      int Nmax=(Nx>Ny?Nx:Ny);
      if(Nx<Nmax){ int old=Nx; ArrayResize(x,Nmax); for(int i=old;i<Nmax;i++) x[i]=0.0; Nx=Nmax; }
      if(Ny<Nmax){ int old=Ny; ArrayResize(y,Nmax); for(int i=old;i<Nmax;i++) y[i]=0.0; Ny=Nmax; }
     }

   // boundary
   int boundary_type=0;
   if(boundary=="even") boundary_type=1;
   else if(boundary=="odd") boundary_type=2;
   else if(boundary=="constant") boundary_type=3;
   else if(boundary=="zeros") boundary_type=4;
   int nedge=(boundary_type==0?0:(nperseg/2));
   if(nedge<0) nedge=0;
   if(nedge>Nx-1) nedge=Nx-1;
   int ext_valid = Nx + 2*nedge;
   int N=ext_valid;

   double win[];
   _triage_segments(window,N,nperseg,win);
   int seglen=nperseg;
   if(noverlap<0) noverlap=seglen/2;
   if(noverlap>=seglen) noverlap=seglen-1;
   if(nfft<=0) nfft=seglen;
   if(nfft<seglen) nfft=seglen;

   int N_eff=ext_valid;
   int step=seglen-noverlap;
   if(padded)
     {
      int nseg_pad=(N_eff-noverlap+step-1)/step;
      int total=nseg_pad*step + noverlap;
      if(total>N_eff) N_eff=total;
     }
   int nseg;
   if(seglen==1 && noverlap==0) nseg=N_eff;
   else nseg=(N_eff-noverlap)/step;
   int nfreq = return_onesided ? (nfft/2+1) : nfft;

   if(!CLFFTInit(plan,nfft)) return false;
   int scaling_mode=0;
   if(scaling=="density") scaling_mode=1;
   else if(scaling=="spectrum") scaling_mode=2;

   // STFT for X
   if(!CLFFTLoadRealSegmentsDetrendBatch(plan,x,win,0,step,seglen,nfft,detrend_type,nseg,boundary_type,nedge,ext_valid))
     { ArrayResize(result,0,0); return false; }
   if(!CLFFTExecuteBatchFromMemA_NoRead(plan,nseg,false)) { ArrayResize(result,0,0); return false; }
   if(scaling_mode!=0)
     {
      double wsum=0.0, winpow=0.0;
      if(!CLFFTComputeWinStats(plan,wsum,winpow)) { ArrayResize(result,0,0); return false; }
      double sc=1.0;
      if(scaling_mode==1){ if(winpow>0.0) sc=MathSqrt(1.0/(fs*winpow)); }
      else if(scaling_mode==2){ if(wsum!=0.0) sc=1.0/wsum; }
      if(sc!=1.0){ if(!CLFFTScaleBatchFromFinal(plan,nseg,sc)) { ArrayResize(result,0,0); return false; } }
     }
   int totalPack=nseg*nfreq;
   if(!CLFFTPackEnsure(plan,totalPack)) { ArrayResize(result,0,0); return false; }
   if(!CLFFTPackSegments(plan,plan.memFinal,nseg,nfft,nfreq,plan.memPack)) { ArrayResize(result,0,0); return false; }
   int memX=plan.memPack;
   int memY=plan.memPack;

   if(!same_data)
     {
      if(!CLFFTLoadRealSegmentsDetrendBatch(plan,y,win,0,step,seglen,nfft,detrend_type,nseg,boundary_type,nedge,ext_valid))
        { ArrayResize(result,0,0); return false; }
      if(!CLFFTExecuteBatchFromMemA_NoRead(plan,nseg,false)) { ArrayResize(result,0,0); return false; }
      if(scaling_mode!=0)
        {
         double wsum=0.0, winpow=0.0;
         if(!CLFFTComputeWinStats(plan,wsum,winpow)) { ArrayResize(result,0,0); return false; }
         double sc=1.0;
         if(scaling_mode==1){ if(winpow>0.0) sc=MathSqrt(1.0/(fs*winpow)); }
         else if(scaling_mode==2){ if(wsum!=0.0) sc=1.0/wsum; }
         if(sc!=1.0){ if(!CLFFTScaleBatchFromFinal(plan,nseg,sc)) { ArrayResize(result,0,0); return false; } }
        }
      if(!CLFFTEnsureHalfBuffer(plan,totalPack)) { ArrayResize(result,0,0); return false; }
      if(!CLFFTPackSegments(plan,plan.memFinal,nseg,nfft,nfreq,plan.memHalf)) { ArrayResize(result,0,0); return false; }
      memY=plan.memHalf;
     }

   if(!CLFFTPSDCompute(plan,memX,memY,nseg,nfreq,nfft,return_onesided))
     { ArrayResize(result,0,0); return false; }

   // export freqs/t
   if(!CLFFTGenerateFreqs(plan,nfft,fs,return_onesided,freqs)) { ArrayResize(freqs,0); }
   if(!CLFFTGenerateTimes(plan,nseg,seglen,noverlap,fs,boundary_type,t)) { ArrayResize(t,0); }

   // read result segments
   ArrayResize(result,nseg,nfreq);
   double buf[]; ArrayResize(buf,2*totalPack);
   CLBufferRead(plan.memPSD,buf);
   for(int s=0;s<nseg;s++)
     {
      int base=s*nfreq;
      for(int k=0;k<nfreq;k++) result[s][k]=buf[2*(base+k)];
     }
#undef plan
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
   ArraySort(vals);
   if((n%2)==1) med=vals[n/2];
   else med=0.5*(vals[n/2-1]+vals[n/2]);
  }

inline bool csd_1d(const double &x[],const double &y[],const double fs,const string window,int nperseg,int noverlap,int nfft,
                   const int detrend_type,const bool return_onesided,const string scaling,const string average,
                   double &freqs[],Complex64 &Pxy[])
  {
   // GPU path: compute PSD/CSD and reduce on GPU
   _spectral_plan_ensure();
#define plan gSpectralPlan

   double xx[]; ArrayCopy(xx,x);
   double yy[]; ArrayCopy(yy,y);
   int Nx=ArraySize(xx);
   int Ny=ArraySize(yy);
   if(Nx<=0 || Ny<=0) { ArrayResize(Pxy,0); return false; }
   if(Nx!=Ny)
     {
      int Nmax=(Nx>Ny?Nx:Ny);
      if(Nx<Nmax){ int old=Nx; ArrayResize(xx,Nmax); for(int i=old;i<Nmax;i++) xx[i]=0.0; Nx=Nmax; }
      if(Ny<Nmax){ int old=Ny; ArrayResize(yy,Nmax); for(int i=old;i<Nmax;i++) yy[i]=0.0; Ny=Nmax; }
     }

   int boundary_type=0;
   int nedge=0;
   int ext_valid=Nx;
   int N=ext_valid;
   double win[];
   _triage_segments(window,N,nperseg,win);
   int seglen=nperseg;
   if(noverlap<0) noverlap=seglen/2;
   if(noverlap>=seglen) noverlap=seglen-1;
   if(nfft<=0) nfft=seglen;
   if(nfft<seglen) nfft=seglen;
   int N_eff=ext_valid;
   int step=seglen-noverlap;
   int nseg;
   if(seglen==1 && noverlap==0) nseg=N_eff;
   else nseg=(N_eff-noverlap)/step;
   int nfreq = return_onesided ? (nfft/2+1) : nfft;

   if(!CLFFTInit(plan,nfft)) { _spectral_log_fail("csd_1d: CLFFTInit"); ArrayResize(Pxy,0); return false; }
   int scaling_mode=0;
   if(scaling=="density") scaling_mode=1;
   else if(scaling=="spectrum") scaling_mode=2;

   if(!CLFFTLoadRealSegmentsDetrendBatch(plan,xx,win,0,step,seglen,nfft,detrend_type,nseg,boundary_type,nedge,ext_valid))
     { _spectral_log_fail("csd_1d: LoadSegments X"); ArrayResize(Pxy,0); return false; }
   if(!CLFFTExecuteBatchFromMemA_NoRead(plan,nseg,false)) { _spectral_log_fail("csd_1d: ExecBatch X"); ArrayResize(Pxy,0); return false; }
   if(scaling_mode!=0)
     {
      double wsum=0.0, winpow=0.0;
      if(!CLFFTComputeWinStats(plan,wsum,winpow)) { _spectral_log_fail("csd_1d: WinStats X"); ArrayResize(Pxy,0); return false; }
      double sc=1.0;
      if(scaling_mode==1){ if(winpow>0.0) sc=MathSqrt(1.0/(fs*winpow)); }
      else if(scaling_mode==2){ if(wsum!=0.0) sc=1.0/wsum; }
      if(sc!=1.0){ if(!CLFFTScaleBatchFromFinal(plan,nseg,sc)) { _spectral_log_fail("csd_1d: ScaleBatch X"); ArrayResize(Pxy,0); return false; } }
     }
   int totalPack=nseg*nfreq;
   if(!CLFFTPackEnsure(plan,totalPack)) { _spectral_log_fail("csd_1d: PackEnsure X"); ArrayResize(Pxy,0); return false; }
   if(!CLFFTPackSegments(plan,plan.memFinal,nseg,nfft,nfreq,plan.memPack)) { _spectral_log_fail("csd_1d: PackSegments X"); ArrayResize(Pxy,0); return false; }
   int memX=plan.memPack;

   if(!CLFFTLoadRealSegmentsDetrendBatch(plan,yy,win,0,step,seglen,nfft,detrend_type,nseg,boundary_type,nedge,ext_valid))
     { _spectral_log_fail("csd_1d: LoadSegments Y"); ArrayResize(Pxy,0); return false; }
   if(!CLFFTExecuteBatchFromMemA_NoRead(plan,nseg,false)) { _spectral_log_fail("csd_1d: ExecBatch Y"); ArrayResize(Pxy,0); return false; }
   if(scaling_mode!=0)
     {
      double wsum=0.0, winpow=0.0;
      if(!CLFFTComputeWinStats(plan,wsum,winpow)) { _spectral_log_fail("csd_1d: WinStats Y"); ArrayResize(Pxy,0); return false; }
      double sc=1.0;
      if(scaling_mode==1){ if(winpow>0.0) sc=MathSqrt(1.0/(fs*winpow)); }
      else if(scaling_mode==2){ if(wsum!=0.0) sc=1.0/wsum; }
      if(sc!=1.0){ if(!CLFFTScaleBatchFromFinal(plan,nseg,sc)) { _spectral_log_fail("csd_1d: ScaleBatch Y"); ArrayResize(Pxy,0); return false; } }
     }
   if(!CLFFTEnsureHalfBuffer(plan,totalPack)) { _spectral_log_fail("csd_1d: EnsureHalf"); ArrayResize(Pxy,0); return false; }
   if(!CLFFTPackSegments(plan,plan.memFinal,nseg,nfft,nfreq,plan.memHalf)) { _spectral_log_fail("csd_1d: PackSegments Y2"); ArrayResize(Pxy,0); return false; }
   int memY=plan.memHalf;

   if(!CLFFTPSDCompute(plan,memX,memY,nseg,nfreq,nfft,return_onesided)) { _spectral_log_fail("csd_1d: PSDCompute"); ArrayResize(Pxy,0); return false; }

   string avg=average;
   StringToLower(avg);
   if(avg=="mean")
     {
      if(!CLFFTPSDReduceMean(plan,nseg,nfreq,Pxy)) { ArrayResize(Pxy,0); return false; }
     }
   else if(avg=="median")
     {
      double bias=_median_bias_mql(nseg);
      if(!CLFFTPSDReduceMedian(plan,nseg,nfreq,bias,Pxy)) { ArrayResize(Pxy,0); return false; }
     }
   else { ArrayResize(Pxy,0); return false; }

   if(!CLFFTGenerateFreqs(plan,nfft,fs,return_onesided,freqs)) { ArrayResize(freqs,0); }
#undef plan
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

// Compute instability of dominant frequency from complex STFT peaks.
// Returns false if STFT cannot be computed.
inline bool stft_1d_matrixc(const double &x[],const double fs,const string window,int nperseg,int noverlap,int nfft,
                             const int detrend_type,const bool return_onesided,const string scaling,
                             double &freqs[],double &t[],matrixc &Z)
  {
   _spectral_plan_ensure();
#define plan gSpectralPlan

   double xx[]; ArrayCopy(xx,x);
   int Norig=ArraySize(xx);
   if(Norig<=0) { Z.Resize(0,0); return false; }

   int boundary_type=0;
   int nedge=0;
   int ext_valid=Norig;
   int N=ext_valid;

   double win[];
   _triage_segments(window,N,nperseg,win);
   int seglen=nperseg;
   if(noverlap<0) noverlap=seglen/2;
   if(noverlap>=seglen) noverlap=seglen-1;
   if(nfft<=0) nfft=seglen;
   if(nfft<seglen) nfft=seglen;
   int step=seglen-noverlap;
   int nseg;
   if(seglen==1 && noverlap==0) nseg=ext_valid;
   else nseg=(ext_valid-noverlap)/step;
   int nfreq = return_onesided ? (nfft/2+1) : nfft;
   if(nseg<=0 || nfreq<=0) { Z.Resize(0,0); return false; }

   int scaling_mode=0;
   if(scaling=="density") scaling_mode=1;
   else if(scaling=="spectrum") scaling_mode=2;

   if(!_fft_helper_1d_mem(plan,xx,win,seglen,noverlap,nfft,return_onesided?1:0,detrend_type,
                          boundary_type,nedge,ext_valid,nseg,scaling_mode,fs))
     { Z.Resize(0,0); return false; }

   int totalPack=nseg*nfreq;
   if(!CLFFTPackEnsure(plan,totalPack)) { Z.Resize(0,0); return false; }
   if(!CLFFTPackSegments(plan,plan.memFinal,nseg,nfft,nfreq,plan.memPack)) { Z.Resize(0,0); return false; }

   double buf[]; ArrayResize(buf,2*totalPack);
   CLBufferRead(plan.memPack,buf);

   Z.Resize(nseg,nfreq);
   for(int s=0;s<nseg;s++)
     {
      int base=s*nfreq;
      for(int k=0;k<nfreq;k++)
        {
         int idx=(base + k)*2;
         complex z; z.real=buf[idx]; z.imag=buf[idx+1];
         Z[s][k]=z;
        }
     }
   if(!CLFFTGenerateFreqs(plan,nfft,fs,return_onesided,freqs)) { ArrayResize(freqs,0); }
   if(!CLFFTGenerateTimes(plan,nseg,seglen,noverlap,fs,boundary_type,t)) { ArrayResize(t,0); }
#undef plan
   return true;
  }

inline bool stft_peak_instability_1d(const double &x[],const double fs,const string window,int nperseg,int noverlap,int nfft,
                                     const int detrend_type,const bool return_onesided,const string scaling,
                                     const string boundary,const bool padded,
                                     double &instab,double &mean_f)
  {
   instab=0.0; mean_f=0.0;
   double freqs[]; double t[]; matrixc Z;
   if(!stft_1d_matrixc(x,fs,window,nperseg,noverlap,nfft,detrend_type,return_onesided,scaling,freqs,t,Z))
      return false;
   int nseg=(int)Z.Rows();
   int nfreq=(int)Z.Cols();
   if(nseg<=1 || nfreq<=1) return false;
   double peaks[]; ArrayResize(peaks,nseg);
   for(int s=0;s<nseg;s++)
     {
      int kmax=1; double vmax=0.0;
      for(int k=1;k<nfreq;k++)
        {
         complex z = Z[s][k];
         double mag=MathSqrt(z.real*z.real + z.imag*z.imag);
         if(mag>vmax) { vmax=mag; kmax=k; }
        }
      peaks[s] = (kmax < ArraySize(freqs) ? freqs[kmax] : 0.0);
     }
   for(int i=0;i<nseg;i++) mean_f += peaks[i];
   mean_f /= (double)nseg;
   if(mean_f<=1e-12) { instab=0.0; return true; }
   double var=0.0;
   for(int i=0;i<nseg;i++) { double d=peaks[i]-mean_f; var += d*d; }
   var /= (double)nseg;
   instab = MathSqrt(var)/mean_f;
   return true;
  }

inline bool spectrogram_1d(const double &x[],const double fs,const string window,int nperseg,int noverlap,int nfft,
                           const int detrend_type,const bool return_onesided,const string scaling,
                           const string mode,
                           double &freqs[],double &t[],double &Sxx[][])
  {
   string m=mode;
   StringToLower(m);
   if(m=="psd")
     {
      if(!spectral_helper_psd_1d(x,x,true,fs,window,nperseg,noverlap,nfft,detrend_type,return_onesided,scaling,"",false,freqs,t,Sxx))
        { ArrayResize(Sxx,0,0); return false; }
      return true;
     }
   if(m=="complex") return false;

   // GPU STFT -> magnitude/phase on GPU (no CPU simplification)
   _spectral_plan_ensure();

   double xx[]; ArrayCopy(xx,x);
   int Norig=ArraySize(xx);
   if(Norig<=0) { ArrayResize(Sxx,0,0); return false; }

   int boundary_type=0;
   int nedge=0;
   int ext_valid=Norig;
   int N=ext_valid;

   double win[];
   _triage_segments(window,N,nperseg,win);
   int seglen=nperseg;
   if(noverlap<0) noverlap=seglen/2;
   if(noverlap>=seglen) noverlap=seglen-1;
   if(nfft<=0) nfft=seglen;
   if(nfft<seglen) nfft=seglen;
   int step=seglen-noverlap;
   int nseg;
   if(seglen==1 && noverlap==0) nseg=ext_valid;
   else nseg=(ext_valid-noverlap)/step;
   int nfreq = return_onesided ? (nfft/2+1) : nfft;

   int scaling_mode=0;
   if(scaling=="density") scaling_mode=1;
   else if(scaling=="spectrum") scaling_mode=2;

   if(!_fft_helper_1d_mem(gSpectralPlan,xx,win,seglen,noverlap,nfft,return_onesided?1:0,detrend_type,
                          boundary_type,nedge,ext_valid,nseg,scaling_mode,fs))
     { ArrayResize(Sxx,0,0); return false; }

   int totalPack=nseg*nfreq;
   if(!CLFFTPackEnsure(gSpectralPlan,totalPack)) { ArrayResize(Sxx,0,0); return false; }
   if(!CLFFTPackSegments(gSpectralPlan,gSpectralPlan.memFinal,nseg,nfft,nfreq,gSpectralPlan.memPack)) { ArrayResize(Sxx,0,0); return false; }

   if(m=="magnitude")
     {
      if(!CLFFTComputeMag(gSpectralPlan,gSpectralPlan.memPack,totalPack)) { ArrayResize(Sxx,0,0); return false; }
     }
   else if(m=="angle" || m=="phase")
     {
      if(!CLFFTComputePhase(gSpectralPlan,gSpectralPlan.memPack,totalPack)) { ArrayResize(Sxx,0,0); return false; }
      if(m=="phase")
        {
         if(!CLFFTUnwrapPhase(gSpectralPlan,nseg,nfreq)) { ArrayResize(Sxx,0,0); return false; }
        }
     }
   else
     {
      ArrayResize(Sxx,0,0); return false;
     }

   ArrayResize(Sxx,nseg,nfreq);
   double buf[]; ArrayResize(buf,totalPack);
   CLBufferRead(gSpectralPlan.memOutReal,buf);
   for(int s=0;s<nseg;s++)
     {
      int base=s*nfreq;
      for(int k=0;k<nfreq;k++) Sxx[s][k]=buf[base+k];
     }
   if(!CLFFTGenerateFreqs(gSpectralPlan,nfft,fs,return_onesided,freqs)) { ArrayResize(freqs,0); }
   if(!CLFFTGenerateTimes(gSpectralPlan,nseg,seglen,noverlap,fs,boundary_type,t)) { ArrayResize(t,0); }
   return true;
  }

// Flat-output spectrogram (avoids 2D dynamic arrays in calling code)
inline bool spectrogram_1d_flat(const double &x[],const double fs,const string window,int nperseg,int noverlap,int nfft,
                                const int detrend_type,const bool return_onesided,const string scaling,
                                const string mode,
                                double &freqs[],double &t[],double &Sxx_flat[],int &nseg_out,int &nfreq_out)
  {
   string m=mode;
   StringToLower(m);
   if(m=="psd" || m=="complex") return false;

   // GPU STFT -> magnitude/phase on GPU (no CPU simplification)
   _spectral_plan_ensure();

   double xx[]; ArrayCopy(xx,x);
   int Norig=ArraySize(xx);
   if(Norig<=0) { ArrayResize(Sxx_flat,0); nseg_out=0; nfreq_out=0; return false; }

   int boundary_type=0;
   int nedge=0;
   int ext_valid=Norig;
   int N=ext_valid;

   double win[];
   _triage_segments(window,N,nperseg,win);
   int seglen=nperseg;
   if(noverlap<0) noverlap=seglen/2;
   if(noverlap>=seglen) noverlap=seglen-1;
   if(nfft<=0) nfft=seglen;
   if(nfft<seglen) nfft=seglen;
   int step=seglen-noverlap;
   int nseg=0;
   if(seglen==1 && noverlap==0) nseg=ext_valid;
   else nseg=(ext_valid-noverlap)/step;
   int nfreq = return_onesided ? (nfft/2+1) : nfft;

   int scaling_mode=0;
   if(scaling=="density") scaling_mode=1;
   else if(scaling=="spectrum") scaling_mode=2;

   if(!_fft_helper_1d_mem(gSpectralPlan,xx,win,seglen,noverlap,nfft,return_onesided?1:0,detrend_type,
                          boundary_type,nedge,ext_valid,nseg,scaling_mode,fs))
     { ArrayResize(Sxx_flat,0); nseg_out=0; nfreq_out=0; return false; }

   int totalPack=nseg*nfreq;
   if(!CLFFTPackEnsure(gSpectralPlan,totalPack)) { ArrayResize(Sxx_flat,0); nseg_out=0; nfreq_out=0; return false; }
   if(!CLFFTPackSegments(gSpectralPlan,gSpectralPlan.memFinal,nseg,nfft,nfreq,gSpectralPlan.memPack)) { ArrayResize(Sxx_flat,0); nseg_out=0; nfreq_out=0; return false; }

   if(m=="magnitude")
     {
      if(!CLFFTComputeMag(gSpectralPlan,gSpectralPlan.memPack,totalPack)) { ArrayResize(Sxx_flat,0); nseg_out=0; nfreq_out=0; return false; }
     }
   else if(m=="angle" || m=="phase")
     {
      if(!CLFFTComputePhase(gSpectralPlan,gSpectralPlan.memPack,totalPack)) { ArrayResize(Sxx_flat,0); nseg_out=0; nfreq_out=0; return false; }
      if(m=="phase")
        {
         if(!CLFFTUnwrapPhase(gSpectralPlan,nseg,nfreq)) { ArrayResize(Sxx_flat,0); nseg_out=0; nfreq_out=0; return false; }
        }
     }
   else
     {
      ArrayResize(Sxx_flat,0); nseg_out=0; nfreq_out=0; return false;
     }

   ArrayResize(Sxx_flat,totalPack);
   CLBufferRead(gSpectralPlan.memOutReal,Sxx_flat);
   if(!CLFFTGenerateFreqs(gSpectralPlan,nfft,fs,return_onesided,freqs)) { ArrayResize(freqs,0); }
   if(!CLFFTGenerateTimes(gSpectralPlan,nseg,seglen,noverlap,fs,boundary_type,t)) { ArrayResize(t,0); }
   nseg_out=nseg;
   nfreq_out=nfreq;
   return true;
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
   _spectral_plan_ensure();
   int nfreq=ArraySize(Pxy);
   if(nfreq<=0) { ArrayResize(Cxy,0); return false; }
   if(!CLFFTInit(gSpectralPlan,nfreq)) return false;
   if(!CLFFTComputeCoherenceFromArrays(gSpectralPlan,Pxy,Pxx,Pyy,Cxy)) { ArrayResize(Cxy,0); return false; }
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

   int N=nfft;
   int batch=nseg;
   _spectral_plan_ensure();
   if(!CLFFTInit(gSpectralPlan,N)) return false;

   double wsum=0.0, wsum2=0.0;
   if(!CLFFTUpldRealSeries(gSpectralPlan,win,win)) return false;
   if(!CLFFTComputeWinStats(gSpectralPlan,wsum,wsum2)) return false;
   double scale=1.0;
   if(scaling=="spectrum") scale=wsum;
   else if(scaling=="psd") scale=MathSqrt(fs*wsum2);
   else return false;

   if(input_onesided)
     {
      Complex64 inHalf[];
      ArrayResize(inHalf,batch*nfreq);
      for(int s=0;s<batch;s++)
        {
         int base=s*nfreq;
         for(int k=0;k<nfreq;k++) inHalf[base+k]=Zxx[s][k];
        }
      if(!CLFFTExpandOnesidedToMemA(gSpectralPlan,inHalf,batch,nfreq)) return false;
     }
   else
     {
      if(nfreq!=N) return false;
      Complex64 inFlat[];
      ArrayResize(inFlat,batch*N);
      for(int s=0;s<batch;s++)
        {
         int base=s*N;
         for(int k=0;k<N;k++) inFlat[base+k]=Zxx[s][k];
        }
      if(!CLFFTUploadComplexBatch(gSpectralPlan,inFlat,batch)) return false;
     }

   if(!CLFFTExecuteBatchFromMemA_NoRead(gSpectralPlan,batch,true)) return false;

   int outlen = nperseg + (batch-1)*nstep;
   if(!CLFFTOverlapAddFromFinal_NoRead(gSpectralPlan,batch,nperseg,nstep,N,win,scale,outlen)) return false;

   if(boundary)
     {
      int cut=nperseg/2;
      int newlen=outlen-2*cut;
      if(newlen<=0) return false;
      if(!CLFFTCropComplexFromMem(gSpectralPlan,gSpectralPlan.memFinal,cut,newlen)) return false;
      outlen=newlen;
     }

   double buf[];
   ArrayResize(buf,2*outlen);
   CLBufferRead(gSpectralPlan.memFinal,buf);
   ArrayResize(x,outlen);
   for(int i=0;i<outlen;i++) x[i]=Cx(buf[2*i],buf[2*i+1]);

   if(!CLFFTGenerateTimeLinear(gSpectralPlan,outlen,fs,time)) { ArrayResize(time,0); return false; }
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

inline bool check_COLA_1d(const string window,const int nperseg,const int noverlap,const double tol)
  {
   if(nperseg<1) return false;
   if(noverlap>=nperseg || noverlap<0) return false;
   double win[];
   CLGetWindow(window,nperseg,true,win);
   _spectral_plan_ensure();
   if(!CLFFTInit(gSpectralPlan,nperseg)) return false;
   bool ok=false;
   if(!CLFFTCheckCOLA(gSpectralPlan,win,nperseg,noverlap,tol,ok)) return false;
   return ok;
  }

inline bool check_NOLA_1d(const string window,const int nperseg,const int noverlap,const double tol)
  {
   if(nperseg<1) return false;
   if(noverlap>=nperseg || noverlap<0) return false;
   double win[];
   CLGetWindow(window,nperseg,true,win);
   static CLFFTPlan plan;
   if(!plan.ready) CLFFTReset(plan);
   if(!CLFFTInit(plan,nperseg)) return false;
   bool ok=false;
   if(!CLFFTCheckNOLA(plan,win,nperseg,noverlap,tol,ok)) return false;
   return ok;
  }

#endif
