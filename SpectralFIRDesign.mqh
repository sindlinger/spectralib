#ifndef __SPECTRAL_FIR_DESIGN_MQH__
#define __SPECTRAL_FIR_DESIGN_MQH__

#include "SpectralCommon.mqh"
#include "SpectralOpenCLFIR.mqh"
#include "SpectralOpenCLFFT.mqh"
#include "SpectralOpenCLWindows.mqh"
#include "SpectralWindows.mqh"

// FIR design port (float64) from _fir_filter_design.py

// ---- Kaiser helpers ----
inline double kaiser_beta(const double a)
  {
   if(a>50.0) return 0.1102*(a-8.7);
   if(a>21.0) return 0.5842*MathPow(a-21.0,0.4) + 0.07886*(a-21.0);
   return 0.0;
  }

inline double kaiser_atten(const int numtaps,const double width)
  {
   return 2.285*(numtaps-1)*PI*width + 7.95;
  }

inline bool kaiserord(const double ripple,const double width,int &numtaps,double &beta)
  {
   double A=MathAbs(ripple);
   if(A<8.0) return false;
   double n = (A - 7.95) / 2.285 / (PI*width) + 1.0;
   numtaps=(int)MathCeil(n);
   beta=kaiser_beta(A);
   return true;
  }

// ---- internal GPU helpers ----
inline bool _cl_reduce_sum(CLFIRPlan &p,const int mem,const int n,double &out)
  {
   if(n<=0) return false;
   int memOut=CLBufferCreate(p.ctx,sizeof(double),CL_MEM_READ_WRITE);
   if(memOut==INVALID_HANDLE) return false;
   CLSetKernelArgMem(p.kern_sum,0,mem);
   CLSetKernelArg(p.kern_sum,1,n);
   CLSetKernelArgMem(p.kern_sum,2,memOut);
   uint offs[1]={0}; uint work[1]={1};
   bool ok=CLExecute(p.kern_sum,1,offs,work);
   double tmp[1];
   if(ok) CLBufferRead(memOut,tmp);
   CLBufferFree(memOut);
   if(!ok) return false;
   out=tmp[0];
   return true;
  }

inline bool _cl_reduce_min(CLFIRPlan &p,const int mem,const int n,double &out)
  {
   if(n<=0) return false;
   int memOut=CLBufferCreate(p.ctx,sizeof(double),CL_MEM_READ_WRITE);
   if(memOut==INVALID_HANDLE) return false;
   CLSetKernelArgMem(p.kern_min,0,mem);
   CLSetKernelArg(p.kern_min,1,n);
   CLSetKernelArgMem(p.kern_min,2,memOut);
   uint offs[1]={0}; uint work[1]={1};
   bool ok=CLExecute(p.kern_min,1,offs,work);
   double tmp[1];
   if(ok) CLBufferRead(memOut,tmp);
   CLBufferFree(memOut);
   if(!ok) return false;
   out=tmp[0];
   return true;
  }

inline bool _cl_reduce_max(CLFIRPlan &p,const int mem,const int n,double &out)
  {
   if(n<=0) return false;
   int memOut=CLBufferCreate(p.ctx,sizeof(double),CL_MEM_READ_WRITE);
   if(memOut==INVALID_HANDLE) return false;
   CLSetKernelArgMem(p.kern_max,0,mem);
   CLSetKernelArg(p.kern_max,1,n);
   CLSetKernelArgMem(p.kern_max,2,memOut);
   uint offs[1]={0}; uint work[1]={1};
   bool ok=CLExecute(p.kern_max,1,offs,work);
   double tmp[1];
   if(ok) CLBufferRead(memOut,tmp);
   CLBufferFree(memOut);
   if(!ok) return false;
   out=tmp[0];
   return true;
  }

inline bool _cl_vec_scale(CLFIRPlan &p,const int mem,const int n,const double s)
  {
   if(n<=0) return false;
   CLSetKernelArgMem(p.kern_vec_scale,0,mem);
   CLSetKernelArg(p.kern_vec_scale,1,n);
   CLSetKernelArg(p.kern_vec_scale,2,s);
   uint offs[1]={0}; uint work[1]={(uint)n};
   return CLExecute(p.kern_vec_scale,1,offs,work);
  }

inline bool _parse_pass_zero(const string pass_zero,bool &pass_zero_bool)
  {
   string p=pass_zero; StringToLower(p);
   if(p=="" || p=="true" || p=="1" || p=="yes")
     { pass_zero_bool=true; return true; }
   if(p=="false" || p=="0" || p=="no")
     { pass_zero_bool=false; return true; }
   if(p=="bandstop" || p=="lowpass") { pass_zero_bool=true; return true; }
   if(p=="bandpass" || p=="highpass") { pass_zero_bool=false; return true; }
   return false;
  }

inline bool _build_bands_from_cutoff(const double &cutoff_in[],const double nyq,
                                     const bool pass_zero,const bool pass_nyquist,
                                     double &bands_flat[])
  {
   int n=ArraySize(cutoff_in);
   if(n<=0 || nyq<=0) return false;
   double cutoff[];
   ArrayResize(cutoff,n);
   for(int i=0;i<n;i++)
     {
      cutoff[i]=cutoff_in[i]/nyq;
      if(cutoff[i]<=0.0 || cutoff[i]>=1.0) return false;
     }
   for(int i=1;i<n;i++)
     if(cutoff[i]<=cutoff[i-1]) return false;

   int extra=(pass_zero?1:0)+(pass_nyquist?1:0);
   int m=n+extra;
   double cutext[];
   ArrayResize(cutext,m);
   int idx=0;
   if(pass_zero) cutext[idx++]=0.0;
   for(int i=0;i<n;i++) cutext[idx++]=cutoff[i];
   if(pass_nyquist) cutext[idx++]=1.0;
   int steps=m/2;
   ArrayResize(bands_flat,2*steps);
   for(int s=0;s<steps;s++)
     {
      bands_flat[2*s]=cutext[2*s];
      bands_flat[2*s+1]=cutext[2*s+1];
     }
   return true;
  }

// ---- firwin (GPU) ----
inline bool firwin(const int numtaps,const double &cutoff_in[],const double width,
                   const string window,const double &window_params[],
                   const string pass_zero,const bool scale,const double fs,
                   double &h[])
  {
   if(numtaps<1) return false;
   double nyq=0.5*fs;
   bool pass_zero_bool;
   if(!_parse_pass_zero(pass_zero,pass_zero_bool)) return false;
   int ncut=ArraySize(cutoff_in);
   bool pass_nyquist = ((ncut & 1)!=0) ^ pass_zero_bool;
   if(pass_nyquist && (numtaps%2==0)) return false;

   string win=window;
   double params[];
   ArrayResize(params,0);
   if(width>0.0)
     {
      double atten=kaiser_atten(numtaps,width/nyq);
      double beta=kaiser_beta(atten);
      win="kaiser";
      ArrayResize(params,1); params[0]=beta;
     }
   else
     {
      ArrayResize(params,ArraySize(window_params));
      for(int i=0;i<ArraySize(window_params);i++) params[i]=window_params[i];
     }

   double bands[];
   if(!_build_bands_from_cutoff(cutoff_in,nyq,pass_zero_bool,pass_nyquist,bands)) return false;
   int steps=ArraySize(bands)/2;

   // window (GPU generator, returned to CPU)
   double winbuf[];
   if(win=="" || win=="none")
     {
      ArrayResize(winbuf,numtaps);
      for(int i=0;i<numtaps;i++) winbuf[i]=1.0;
     }
   else
     {
      if(win=="kaiser" && ArraySize(params)>0)
        {
         static CLWinHandle winplan;
         if(!CLWindowGenerate(winplan,WIN_KAISER,numtaps,true,params,params,winbuf))
            get_window_params(win,numtaps,false,params,winbuf);
        }
      else
        {
         get_window_params(win,numtaps,false,params,winbuf);
        }
     }

   static CLFIRPlan plan;
   if(!CLFIRInit(plan)) return false;

   // allocate buffers
   if(plan.memBands!=INVALID_HANDLE) CLBufferFree(plan.memBands);
   if(plan.memWin!=INVALID_HANDLE) CLBufferFree(plan.memWin);
   if(plan.memH!=INVALID_HANDLE) CLBufferFree(plan.memH);
   if(plan.memHC!=INVALID_HANDLE) CLBufferFree(plan.memHC);
   plan.memBands=CLBufferCreate(plan.ctx,MathMax(1,ArraySize(bands))*sizeof(double),CL_MEM_READ_ONLY);
   plan.memWin=CLBufferCreate(plan.ctx,MathMax(1,numtaps)*sizeof(double),CL_MEM_READ_ONLY);
   plan.memH=CLBufferCreate(plan.ctx,MathMax(1,numtaps)*sizeof(double),CL_MEM_READ_WRITE);
   plan.memHC=CLBufferCreate(plan.ctx,MathMax(1,numtaps)*sizeof(double),CL_MEM_READ_WRITE);
   if(plan.memBands==INVALID_HANDLE || plan.memWin==INVALID_HANDLE || plan.memH==INVALID_HANDLE || plan.memHC==INVALID_HANDLE)
     return false;
   CLBufferWrite(plan.memBands,bands);
   CLBufferWrite(plan.memWin,winbuf);

   // firwin_core
   CLSetKernelArgMem(plan.kern_firwin,0,plan.memWin);
   CLSetKernelArg(plan.kern_firwin,1,numtaps);
   CLSetKernelArgMem(plan.kern_firwin,2,plan.memBands);
   CLSetKernelArg(plan.kern_firwin,3,steps);
   CLSetKernelArg(plan.kern_firwin,4,(int)(scale?1:0));
   CLSetKernelArgMem(plan.kern_firwin,5,plan.memH);
   CLSetKernelArgMem(plan.kern_firwin,6,plan.memHC);
   uint offs[1]={0}; uint work[1]={(uint)numtaps};
   if(!CLExecute(plan.kern_firwin,1,offs,work)) return false;

   if(scale)
     {
      double sumhc=0.0;
      if(!_cl_reduce_sum(plan,plan.memHC,numtaps,sumhc)) return false;
      if(MathAbs(sumhc)<=0.0) return false;
      if(!_cl_vec_scale(plan,plan.memH,numtaps,1.0/sumhc)) return false;

      // rebuild h (sinc)
      CLSetKernelArgMem(plan.kern_firwin_build,0,plan.memWin);
      CLSetKernelArg(plan.kern_firwin_build,1,numtaps);
      CLSetKernelArgMem(plan.kern_firwin_build,2,plan.memBands);
      CLSetKernelArg(plan.kern_firwin_build,3,steps);
      CLSetKernelArgMem(plan.kern_firwin_build,4,plan.memH);
      if(!CLExecute(plan.kern_firwin_build,1,offs,work)) return false;

      // scale again using first passband
      double left=bands[0];
      double right=bands[1];
      double scale_freq=(left==0.0?0.0:(right==1.0?1.0:0.5*(left+right)));
      CLSetKernelArg(plan.kern_firwin_cos,0,numtaps);
      CLSetKernelArg(plan.kern_firwin_cos,1,scale_freq);
      CLSetKernelArgMem(plan.kern_firwin_cos,2,plan.memHC);
      if(!CLExecute(plan.kern_firwin_cos,1,offs,work)) return false;

      CLSetKernelArgMem(plan.kern_vec_mul,0,plan.memH);
      CLSetKernelArgMem(plan.kern_vec_mul,1,plan.memHC);
      CLSetKernelArg(plan.kern_vec_mul,2,numtaps);
      CLSetKernelArgMem(plan.kern_vec_mul,3,plan.memHC);
      if(!CLExecute(plan.kern_vec_mul,1,offs,work)) return false;

      double sumhc2=0.0;
      if(!_cl_reduce_sum(plan,plan.memHC,numtaps,sumhc2)) return false;
      if(MathAbs(sumhc2)<=0.0) return false;
      if(!_cl_vec_scale(plan,plan.memH,numtaps,1.0/sumhc2)) return false;
     }

   ArrayResize(h,numtaps);
   CLBufferRead(plan.memH,h);
   return true;
  }

inline bool firwin_scalar(const int numtaps,const double cutoff,const double width,
                          const string window,const string pass_zero,const bool scale,const double fs,
                          double &h[])
  {
   double c[1]; c[0]=cutoff;
   double params[]; ArrayResize(params,0);
   return firwin(numtaps,c,width,window,params,pass_zero,scale,fs,h);
  }

// ---- firwin2 (GPU FFT, CPU interp/shift) ----
inline bool firwin2(const int numtaps,const double &freq_in[],const double &gain_in[],
                    const int nfreqs_in,const string window,const bool antisymmetric,
                    const double fs,double &out[])
  {
   int nf=ArraySize(freq_in);
   if(nf!=ArraySize(gain_in) || nf<2) return false;
   double nyq=0.5*fs;
   if(freq_in[0]!=0.0 || freq_in[nf-1]!=nyq) return false;
   for(int i=1;i<nf;i++) if(freq_in[i]<freq_in[i-1]) return false;
   for(int i=0;i<nf-2;i++) if(freq_in[i]==freq_in[i+1] && freq_in[i+1]==freq_in[i+2]) return false;
   if(freq_in[1]==0.0 || freq_in[nf-2]==nyq) return false;

   int ftype=1;
   if(antisymmetric) ftype = (numtaps%2==0)?4:3;
   else ftype = (numtaps%2==0)?2:1;
   if(ftype==2 && gain_in[nf-1]!=0.0) return false;
   if(ftype==3 && (gain_in[0]!=0.0 || gain_in[nf-1]!=0.0)) return false;
   if(ftype==4 && gain_in[0]!=0.0) return false;

   int nfreqs=nfreqs_in;
   if(nfreqs<=0)
     {
      int p=1; while(p<numtaps) p<<=1;
      nfreqs=1+p;
     }
   if(numtaps>=nfreqs) return false;

   double freq[]; ArrayResize(freq,nf);
   for(int i=0;i<nf;i++) freq[i]=freq_in[i];
   for(int i=0;i<nf-1;i++)
     if(freq[i]==freq[i+1])
       {
        double eps=2.2204460492503131e-16*nyq;
        freq[i]-=eps; freq[i+1]+=eps;
       }
   for(int i=1;i<nf;i++) if(freq[i]<=freq[i-1]) return false;

   double x[];
   Linspace(0.0,nyq,nfreqs,x);

   static CLFIRPlan fplan;
   if(!CLFIRInit(fplan)) return false;

   int memX=CLBufferCreate(fplan.ctx,nfreqs*sizeof(double),CL_MEM_READ_ONLY);
   int memFreq=CLBufferCreate(fplan.ctx,nf*sizeof(double),CL_MEM_READ_ONLY);
   int memGain=CLBufferCreate(fplan.ctx,nf*sizeof(double),CL_MEM_READ_ONLY);
   int memFx=CLBufferCreate(fplan.ctx,nfreqs*sizeof(double),CL_MEM_READ_WRITE);
   int memShift=CLBufferCreate(fplan.ctx,2*nfreqs*sizeof(double),CL_MEM_READ_WRITE);
   int memFx2=CLBufferCreate(fplan.ctx,2*nfreqs*sizeof(double),CL_MEM_READ_WRITE);
   if(memX==INVALID_HANDLE || memFreq==INVALID_HANDLE || memGain==INVALID_HANDLE || memFx==INVALID_HANDLE || memShift==INVALID_HANDLE || memFx2==INVALID_HANDLE)
     {
      if(memX!=INVALID_HANDLE) CLBufferFree(memX);
      if(memFreq!=INVALID_HANDLE) CLBufferFree(memFreq);
      if(memGain!=INVALID_HANDLE) CLBufferFree(memGain);
      if(memFx!=INVALID_HANDLE) CLBufferFree(memFx);
      if(memShift!=INVALID_HANDLE) CLBufferFree(memShift);
      if(memFx2!=INVALID_HANDLE) CLBufferFree(memFx2);
      return false;
     }
   CLBufferWrite(memX,x);
   CLBufferWrite(memFreq,freq);
   CLBufferWrite(memGain,gain_in);

   bool ok=true;
   uint offs[1]={0}; uint workx[1]={(uint)nfreqs};

   // interpolate fx on GPU
   CLSetKernelArgMem(fplan.kern_interp1d,0,memX);
   CLSetKernelArg(fplan.kern_interp1d,1,nfreqs);
   CLSetKernelArgMem(fplan.kern_interp1d,2,memFreq);
   CLSetKernelArgMem(fplan.kern_interp1d,3,memGain);
   CLSetKernelArg(fplan.kern_interp1d,4,nf);
   CLSetKernelArgMem(fplan.kern_interp1d,5,memFx);
   if(!CLExecute(fplan.kern_interp1d,1,offs,workx)) ok=false;

   // build shift and fx2 on GPU
   double alpha=0.5*(numtaps-1);
   if(ok)
     {
      CLSetKernelArg(fplan.kern_make_shift,0,memX);
      CLSetKernelArg(fplan.kern_make_shift,1,nfreqs);
      CLSetKernelArg(fplan.kern_make_shift,2,alpha);
      CLSetKernelArg(fplan.kern_make_shift,3,nyq);
      CLSetKernelArg(fplan.kern_make_shift,4,ftype);
      CLSetKernelArgMem(fplan.kern_make_shift,5,memShift);
      if(!CLExecute(fplan.kern_make_shift,1,offs,workx)) ok=false;
     }

   if(ok)
     {
      CLSetKernelArgMem(fplan.kern_cplx_mul_real,0,memShift);
      CLSetKernelArgMem(fplan.kern_cplx_mul_real,1,memFx);
      CLSetKernelArg(fplan.kern_cplx_mul_real,2,nfreqs);
      CLSetKernelArgMem(fplan.kern_cplx_mul_real,3,memFx2);
      if(!CLExecute(fplan.kern_cplx_mul_real,1,offs,workx)) ok=false;
     }

   Complex64 fx2[];
   if(ok)
     {
      double fx2buf[]; ArrayResize(fx2buf,2*nfreqs);
      CLBufferRead(memFx2,fx2buf);
      ArrayResize(fx2,nfreqs);
      for(int i=0;i<nfreqs;i++) fx2[i]=Cx(fx2buf[2*i],fx2buf[2*i+1]);
     }

   CLBufferFree(memX); CLBufferFree(memFreq); CLBufferFree(memGain);
   CLBufferFree(memFx); CLBufferFree(memShift); CLBufferFree(memFx2);
   if(!ok) return false;

   // irfft using OpenCL FFT (expand onesided -> ifft)
   int nfull = 2*(nfreqs-1);
   static CLFFTPlan fftplan;
   if(!CLFFTInit(fftplan,nfull)) return false;
   if(!CLFFTExpandOnesidedToMemA(fftplan,fx2,1,nfreqs)) return false;
   if(!CLFFTExecuteFromMemA_NoRead(fftplan,true)) return false;
   double buf[]; ArrayResize(buf,2*nfull);
   CLBufferRead(fftplan.memFinal,buf);
   double out_full[]; ArrayResize(out_full,nfull);
   for(int i=0;i<nfull;i++) out_full[i]=buf[2*i];

   double wind[];
   if(window=="" || window=="none")
     {
      ArrayResize(wind,numtaps);
      for(int i=0;i<numtaps;i++) wind[i]=1.0;
     }
   else get_window(window,numtaps,false,wind);

   ArrayResize(out,numtaps);
   for(int i=0;i<numtaps;i++) out[i]=out_full[i]*wind[i];
   if(ftype==3) out[numtaps/2]=0.0;
   return true;
  }

// ---- firls (GPU) ----
inline bool firls(const int numtaps,const double &bands_in[],const double &desired_in[],
                  const double &weight_in[],const double fs,double &coeffs[])
  {
   if(numtaps<1 || (numtaps%2)==0) return false;
   int nb=ArraySize(bands_in);
   if(nb<2 || (nb%2)!=0) return false;
   if(ArraySize(desired_in)!=nb) return false;
   double nyq=0.5*fs;
   if(nyq<=0) return false;

   int nbands=nb/2;
   double bands[]; ArrayResize(bands,nb);
   for(int i=0;i<nb;i++)
     {
      bands[i]=bands_in[i]/nyq;
      if(bands[i]<0.0 || bands[i]>1.0) return false;
     }

   double desired[]; ArrayResize(desired,nb);
   for(int i=0;i<nb;i++) desired[i]=desired_in[i];

   double weight[]; ArrayResize(weight,nbands);
   if(ArraySize(weight_in)==nbands)
     {
      for(int i=0;i<nbands;i++) weight[i]=weight_in[i];
     }
   else
     {
      for(int i=0;i<nbands;i++) weight[i]=1.0;
     }

   int M=(numtaps-1)/2;
   int N=M+1;

   static CLFIRPlan plan;
   if(!CLFIRInit(plan)) return false;

   // allocate buffers
   if(plan.memBands!=INVALID_HANDLE) CLBufferFree(plan.memBands);
   if(plan.memDesired!=INVALID_HANDLE) CLBufferFree(plan.memDesired);
   if(plan.memWeight!=INVALID_HANDLE) CLBufferFree(plan.memWeight);
   if(plan.memQ!=INVALID_HANDLE) CLBufferFree(plan.memQ);
   if(plan.memq!=INVALID_HANDLE) CLBufferFree(plan.memq);
   if(plan.memb!=INVALID_HANDLE) CLBufferFree(plan.memb);
   if(plan.memx!=INVALID_HANDLE) CLBufferFree(plan.memx);
   if(plan.memFlag!=INVALID_HANDLE) CLBufferFree(plan.memFlag);
   if(plan.memH!=INVALID_HANDLE) CLBufferFree(plan.memH);

   plan.memBands=CLBufferCreate(plan.ctx,nb*sizeof(double),CL_MEM_READ_ONLY);
   plan.memDesired=CLBufferCreate(plan.ctx,nb*sizeof(double),CL_MEM_READ_ONLY);
   plan.memWeight=CLBufferCreate(plan.ctx,nbands*sizeof(double),CL_MEM_READ_ONLY);
   plan.memq=CLBufferCreate(plan.ctx,numtaps*sizeof(double),CL_MEM_READ_WRITE);
   // tamanho em bytes sem perda (evita truncar em uint)
   ulong bytesQ=(ulong)N*(ulong)N*(ulong)sizeof(double);
   if(bytesQ>(ulong)0xFFFFFFFF) return false;
   plan.memQ=CLBufferCreate(plan.ctx,(uint)bytesQ,CL_MEM_READ_WRITE);
   plan.memb=CLBufferCreate(plan.ctx,N*sizeof(double),CL_MEM_READ_WRITE);
   plan.memx=CLBufferCreate(plan.ctx,N*sizeof(double),CL_MEM_READ_WRITE);
   plan.memFlag=CLBufferCreate(plan.ctx,sizeof(int),CL_MEM_READ_WRITE);
   plan.memH=CLBufferCreate(plan.ctx,numtaps*sizeof(double),CL_MEM_READ_WRITE);
   if(plan.memBands==INVALID_HANDLE || plan.memDesired==INVALID_HANDLE || plan.memWeight==INVALID_HANDLE ||
      plan.memq==INVALID_HANDLE || plan.memQ==INVALID_HANDLE || plan.memb==INVALID_HANDLE || plan.memx==INVALID_HANDLE ||
      plan.memFlag==INVALID_HANDLE || plan.memH==INVALID_HANDLE)
      return false;

   CLBufferWrite(plan.memBands,bands);
   CLBufferWrite(plan.memDesired,desired);
   CLBufferWrite(plan.memWeight,weight);
   int flag0[1]={0};
   CLBufferWrite(plan.memFlag,flag0);

   // q
   CLSetKernelArg(plan.kern_firls_q,0,numtaps);
   CLSetKernelArg(plan.kern_firls_q,1,nbands);
   CLSetKernelArgMem(plan.kern_firls_q,2,plan.memBands);
   CLSetKernelArgMem(plan.kern_firls_q,3,plan.memWeight);
   CLSetKernelArgMem(plan.kern_firls_q,4,plan.memq);
   uint offs[1]={0}; uint workq[1]={(uint)numtaps};
   if(!CLExecute(plan.kern_firls_q,1,offs,workq)) return false;

   // Q
   CLSetKernelArg(plan.kern_firls_Q,0,N);
   CLSetKernelArgMem(plan.kern_firls_Q,1,plan.memq);
   CLSetKernelArgMem(plan.kern_firls_Q,2,plan.memQ);
   uint workQ[1]={(uint)(N*N)};
   if(!CLExecute(plan.kern_firls_Q,1,offs,workQ)) return false;

   // b
   CLSetKernelArg(plan.kern_firls_b,0,M);
   CLSetKernelArg(plan.kern_firls_b,1,nbands);
   CLSetKernelArgMem(plan.kern_firls_b,2,plan.memBands);
   CLSetKernelArgMem(plan.kern_firls_b,3,plan.memDesired);
   CLSetKernelArgMem(plan.kern_firls_b,4,plan.memWeight);
   CLSetKernelArgMem(plan.kern_firls_b,5,plan.memb);
   uint workb[1]={(uint)N};
   if(!CLExecute(plan.kern_firls_b,1,offs,workb)) return false;

   // solve
   CLSetKernelArgMem(plan.kern_chol_solve,0,plan.memQ);
   CLSetKernelArgMem(plan.kern_chol_solve,1,plan.memb);
   CLSetKernelArgMem(plan.kern_chol_solve,2,plan.memx);
   CLSetKernelArg(plan.kern_chol_solve,3,N);
   CLSetKernelArgMem(plan.kern_chol_solve,4,plan.memFlag);
   uint work1[1]={1};
   if(!CLExecute(plan.kern_chol_solve,1,offs,work1)) return false;

   int flag[1]; CLBufferRead(plan.memFlag,flag);
   if(flag[0]!=0) return false;

   // coeffs
   CLSetKernelArgMem(plan.kern_firls_coeffs,0,plan.memx);
   CLSetKernelArg(plan.kern_firls_coeffs,1,numtaps);
   CLSetKernelArgMem(plan.kern_firls_coeffs,2,plan.memH);
   uint workc[1]={(uint)numtaps};
   if(!CLExecute(plan.kern_firls_coeffs,1,offs,workc)) return false;

   ArrayResize(coeffs,numtaps);
   CLBufferRead(plan.memH,coeffs);
   return true;
  }

// ---- minimum_phase (FFT GPU, scalar ops CPU for fidelity) ----
inline bool _dhtm(const double &mag[],double &recon[])
  {
   int N=ArraySize(mag);
   if(N<=0) return false;
   static CLFIRPlan fplan;
   if(!CLFIRInit(fplan)) return false;

   // sig for modified DHT
   double sig[]; ArrayResize(sig,N);
   for(int i=0;i<N;i++) sig[i]=0.0;
   int mid=N/2;
   for(int i=1;i<mid;i++) sig[i]=1.0;
   for(int i=mid+1;i<N;i++) sig[i]=-1.0;

   // log(mag) on GPU
   int memMag=CLBufferCreate(fplan.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   int memLog=CLBufferCreate(fplan.ctx,N*sizeof(double),CL_MEM_READ_WRITE);
   if(memMag==INVALID_HANDLE || memLog==INVALID_HANDLE) { if(memMag!=INVALID_HANDLE) CLBufferFree(memMag); if(memLog!=INVALID_HANDLE) CLBufferFree(memLog); return false; }
   CLBufferWrite(memMag,mag);
   CLSetKernelArgMem(fplan.kern_vec_log,0,memMag);
   CLSetKernelArg(fplan.kern_vec_log,1,N);
   CLSetKernelArgMem(fplan.kern_vec_log,2,memLog);
   uint offs[1]={0}; uint work[1]={(uint)N};
   if(!CLExecute(fplan.kern_vec_log,1,offs,work)) { CLBufferFree(memMag); CLBufferFree(memLog); return false; }

   double logmag[]; ArrayResize(logmag,N);
   CLBufferRead(memLog,logmag);
   CLBufferFree(memMag); CLBufferFree(memLog);

   Complex64 tmp[]; ArrayResize(tmp,N);
   for(int i=0;i<N;i++) tmp[i]=Cx(logmag[i],0.0);
   static CLFFTPlan plan;
   if(!CLFFTInit(plan,N)) return false;
   Complex64 t1[];
   if(!CLFFTExecute(plan,tmp,t1,true)) return false; // ifft

   Complex64 t2[]; ArrayResize(t2,N);
   for(int i=0;i<N;i++) t2[i]=Cx(t1[i].re*sig[i], t1[i].im*sig[i]);
   Complex64 t3[];
   if(!CLFFTExecute(plan,t2,t3,false)) return false; // fft

   // exp(t3) on GPU and multiply by mag on GPU
   int memC=CLBufferCreate(fplan.ctx,2*N*sizeof(double),CL_MEM_READ_ONLY);
   int memExp=CLBufferCreate(fplan.ctx,2*N*sizeof(double),CL_MEM_READ_WRITE);
   int memMult=CLBufferCreate(fplan.ctx,2*N*sizeof(double),CL_MEM_READ_WRITE);
   int memMag2=CLBufferCreate(fplan.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   if(memC==INVALID_HANDLE || memExp==INVALID_HANDLE || memMult==INVALID_HANDLE || memMag2==INVALID_HANDLE)
     {
      if(memC!=INVALID_HANDLE) CLBufferFree(memC);
      if(memExp!=INVALID_HANDLE) CLBufferFree(memExp);
      if(memMult!=INVALID_HANDLE) CLBufferFree(memMult);
      if(memMag2!=INVALID_HANDLE) CLBufferFree(memMag2);
      return false;
     }
   double cbuf[]; ArrayResize(cbuf,2*N);
   for(int i=0;i<N;i++){ cbuf[2*i]=t3[i].re; cbuf[2*i+1]=t3[i].im; }
   CLBufferWrite(memC,cbuf);
   CLBufferWrite(memMag2,mag);

   CLSetKernelArgMem(fplan.kern_cplx_exp,0,memC);
   CLSetKernelArg(fplan.kern_cplx_exp,1,N);
   CLSetKernelArgMem(fplan.kern_cplx_exp,2,memExp);
   if(!CLExecute(fplan.kern_cplx_exp,1,offs,work)) { CLBufferFree(memC); CLBufferFree(memExp); CLBufferFree(memMult); CLBufferFree(memMag2); return false; }

   CLSetKernelArgMem(fplan.kern_cplx_mul_real,0,memExp);
   CLSetKernelArgMem(fplan.kern_cplx_mul_real,1,memMag2);
   CLSetKernelArg(fplan.kern_cplx_mul_real,2,N);
   CLSetKernelArgMem(fplan.kern_cplx_mul_real,3,memMult);
   if(!CLExecute(fplan.kern_cplx_mul_real,1,offs,work)) { CLBufferFree(memC); CLBufferFree(memExp); CLBufferFree(memMult); CLBufferFree(memMag2); return false; }

   double mbuf[]; ArrayResize(mbuf,2*N);
   CLBufferRead(memMult,mbuf);
   CLBufferFree(memC); CLBufferFree(memExp); CLBufferFree(memMult); CLBufferFree(memMag2);

   Complex64 mult[]; ArrayResize(mult,N);
   for(int i=0;i<N;i++) mult[i]=Cx(mbuf[2*i], mbuf[2*i+1]);

   Complex64 t4[];
   if(!CLFFTExecute(plan,mult,t4,true)) return false; // ifft
   ArrayResize(recon,N);
   for(int i=0;i<N;i++) recon[i]=t4[i].re;
   return true;
  }

inline bool minimum_phase(const double &h[],const string method,const int n_fft_in,double &h_minimum[])
  {
   int N=ArraySize(h);
   if(N<=2) return false;
   string m=method; StringToLower(m);
   if(m!="homomorphic" && m!="hilbert") return false;

   int n_fft=n_fft_in;
   if(n_fft<=0)
     {
      double eps=0.01;
      n_fft=1;
      int need=(int)MathCeil(MathLog(2.0*(N-1)/eps)/MathLog(2.0));
      n_fft=1<<need;
     }
   if(n_fft<N) return false;

   if(m=="hilbert")
     {
      Complex64 hin[]; ArrayResize(hin,n_fft);
      for(int i=0;i<n_fft;i++) hin[i]=Cx((i<N)?h[i]:0.0,0.0);
      static CLFFTPlan plan;
      if(!CLFFTInit(plan,n_fft)) return false;
      Complex64 Hc[];
      if(!CLFFTExecute(plan,hin,Hc,false)) return false;

      double H[]; ArrayResize(H,n_fft);
      int n_half=N/2;
      for(int k=0;k<n_fft;k++)
        {
         double w=2.0*PI*(double)n_half*(double)k/(double)n_fft;
         H[k]=Hc[k].re*MathCos(w) - Hc[k].im*MathSin(w);
        }
      double maxH=H[0], minH=H[0];
      for(int i=1;i<n_fft;i++){ if(H[i]>maxH) maxH=H[i]; if(H[i]<minH) minH=H[i]; }
      double dp=maxH-1.0;
      double ds=0.0-minH;
      double S=4.0/MathPow(MathSqrt(1.0+dp+ds)+MathSqrt(1.0-dp+ds),2.0);
      // apply scale/sqrt on GPU
      static CLFIRPlan fplan3;
      if(!CLFIRInit(fplan3)) return false;
      int memH=CLBufferCreate(fplan3.ctx,n_fft*sizeof(double),CL_MEM_READ_WRITE);
      int memTmp=CLBufferCreate(fplan3.ctx,n_fft*sizeof(double),CL_MEM_READ_WRITE);
      if(memH==INVALID_HANDLE || memTmp==INVALID_HANDLE)
        { if(memH!=INVALID_HANDLE) CLBufferFree(memH); if(memTmp!=INVALID_HANDLE) CLBufferFree(memTmp); return false; }
      CLBufferWrite(memH,H);
      uint offsH[1]={0}; uint workH[1]={(uint)n_fft};
      CLSetKernelArgMem(fplan3.kern_vec_add,0,memH);
      CLSetKernelArg(fplan3.kern_vec_add,1,ds);
      CLSetKernelArg(fplan3.kern_vec_add,2,n_fft);
      CLSetKernelArgMem(fplan3.kern_vec_add,3,memTmp);
      if(!CLExecute(fplan3.kern_vec_add,1,offsH,workH)) { CLBufferFree(memH); CLBufferFree(memTmp); return false; }
      CLSetKernelArgMem(fplan3.kern_vec_scale,0,memTmp);
      CLSetKernelArg(fplan3.kern_vec_scale,1,n_fft);
      CLSetKernelArg(fplan3.kern_vec_scale,2,S);
      if(!CLExecute(fplan3.kern_vec_scale,1,offsH,workH)) { CLBufferFree(memH); CLBufferFree(memTmp); return false; }
      CLSetKernelArgMem(fplan3.kern_vec_sqrt,0,memTmp);
      CLSetKernelArg(fplan3.kern_vec_sqrt,1,n_fft);
      CLSetKernelArgMem(fplan3.kern_vec_sqrt,2,memH);
      if(!CLExecute(fplan3.kern_vec_sqrt,1,offsH,workH)) { CLBufferFree(memH); CLBufferFree(memTmp); return false; }
      CLSetKernelArgMem(fplan3.kern_vec_add,0,memH);
      CLSetKernelArg(fplan3.kern_vec_add,1,1e-10);
      CLSetKernelArg(fplan3.kern_vec_add,2,n_fft);
      CLSetKernelArgMem(fplan3.kern_vec_add,3,memH);
      if(!CLExecute(fplan3.kern_vec_add,1,offsH,workH)) { CLBufferFree(memH); CLBufferFree(memTmp); return false; }
      CLBufferRead(memH,H);
      CLBufferFree(memH); CLBufferFree(memTmp);
      double recon[];
      if(!_dhtm(H,recon)) return false;
      int n_out = N/2 + (N%2);
      ArrayResize(h_minimum,n_out);
      for(int i=0;i<n_out;i++) h_minimum[i]=recon[i];
      return true;
     }

   // homomorphic
   Complex64 hin[]; ArrayResize(hin,n_fft);
   for(int i=0;i<n_fft;i++) hin[i]=Cx((i<N)?h[i]:0.0,0.0);
   static CLFFTPlan plan2;
   if(!CLFFTInit(plan2,n_fft)) return false;
   Complex64 Hc2[];
   if(!CLFFTExecute(plan2,hin,Hc2,false)) return false;

   static CLFIRPlan fplan2;
   if(!CLFIRInit(fplan2)) return false;

   double mag[]; ArrayResize(mag,n_fft);
   double minpos=0.0;
   // magnitude on GPU
   int memC=CLBufferCreate(fplan2.ctx,2*n_fft*sizeof(double),CL_MEM_READ_ONLY);
   int memMag=CLBufferCreate(fplan2.ctx,n_fft*sizeof(double),CL_MEM_READ_WRITE);
   if(memC==INVALID_HANDLE || memMag==INVALID_HANDLE) { if(memC!=INVALID_HANDLE) CLBufferFree(memC); if(memMag!=INVALID_HANDLE) CLBufferFree(memMag); return false; }
   double cbuf[]; ArrayResize(cbuf,2*n_fft);
   for(int i=0;i<n_fft;i++){ cbuf[2*i]=Hc2[i].re; cbuf[2*i+1]=Hc2[i].im; }
   CLBufferWrite(memC,cbuf);
   uint offs2[1]={0}; uint work2[1]={(uint)n_fft};
   CLSetKernelArgMem(fplan2.kern_vec_abs,0,memC);
   CLSetKernelArg(fplan2.kern_vec_abs,1,n_fft);
   CLSetKernelArgMem(fplan2.kern_vec_abs,2,memMag);
   if(!CLExecute(fplan2.kern_vec_abs,1,offs2,work2)) { CLBufferFree(memC); CLBufferFree(memMag); return false; }
   CLBufferRead(memMag,mag);
   CLBufferFree(memC);

   for(int i=0;i<n_fft;i++)
     if(mag[i]>0.0 && (minpos==0.0 || mag[i]<minpos)) minpos=mag[i];
   if(minpos<=0.0) minpos=1.0;
   double add=1e-7*minpos;
   // mag = 0.5*log(mag+add) on GPU
   CLSetKernelArgMem(fplan2.kern_vec_add,0,memMag);
   CLSetKernelArg(fplan2.kern_vec_add,1,add);
   CLSetKernelArg(fplan2.kern_vec_add,2,n_fft);
   CLSetKernelArgMem(fplan2.kern_vec_add,3,memMag);
   if(!CLExecute(fplan2.kern_vec_add,1,offs2,work2)) { CLBufferFree(memMag); return false; }
   CLSetKernelArgMem(fplan2.kern_vec_log,0,memMag);
   CLSetKernelArg(fplan2.kern_vec_log,1,n_fft);
   CLSetKernelArgMem(fplan2.kern_vec_log,2,memMag);
   if(!CLExecute(fplan2.kern_vec_log,1,offs2,work2)) { CLBufferFree(memMag); return false; }
   CLSetKernelArgMem(fplan2.kern_vec_scale,0,memMag);
   CLSetKernelArg(fplan2.kern_vec_scale,1,n_fft);
   CLSetKernelArg(fplan2.kern_vec_scale,2,0.5);
   if(!CLExecute(fplan2.kern_vec_scale,1,offs2,work2)) { CLBufferFree(memMag); return false; }
   CLBufferRead(memMag,mag);
   CLBufferFree(memMag);

   Complex64 temp[]; ArrayResize(temp,n_fft);
   for(int i=0;i<n_fft;i++) temp[i]=Cx(mag[i],0.0);
   Complex64 ifft1[];
   if(!CLFFTExecute(plan2,temp,ifft1,true)) return false;

   double win[]; ArrayResize(win,n_fft);
   for(int i=0;i<n_fft;i++) win[i]=0.0;
   win[0]=1.0;
   int stop=(N+1)/2;
   for(int i=1;i<stop;i++) win[i]=2.0;
   if(N%2!=0) win[stop]=1.0;
   // apply window on GPU
   int memC2=CLBufferCreate(fplan2.ctx,2*n_fft*sizeof(double),CL_MEM_READ_WRITE);
   int memWin=CLBufferCreate(fplan2.ctx,n_fft*sizeof(double),CL_MEM_READ_ONLY);
   int memOut=CLBufferCreate(fplan2.ctx,2*n_fft*sizeof(double),CL_MEM_READ_WRITE);
   if(memC2==INVALID_HANDLE || memWin==INVALID_HANDLE || memOut==INVALID_HANDLE)
     {
      if(memC2!=INVALID_HANDLE) CLBufferFree(memC2);
      if(memWin!=INVALID_HANDLE) CLBufferFree(memWin);
      if(memOut!=INVALID_HANDLE) CLBufferFree(memOut);
      return false;
     }
   ArrayResize(cbuf,2*n_fft);
   for(int i=0;i<n_fft;i++){ cbuf[2*i]=ifft1[i].re; cbuf[2*i+1]=ifft1[i].im; }
   CLBufferWrite(memC2,cbuf);
   CLBufferWrite(memWin,win);
   CLSetKernelArgMem(fplan2.kern_cplx_mul_real,0,memC2);
   CLSetKernelArgMem(fplan2.kern_cplx_mul_real,1,memWin);
   CLSetKernelArg(fplan2.kern_cplx_mul_real,2,n_fft);
   CLSetKernelArgMem(fplan2.kern_cplx_mul_real,3,memOut);
   if(!CLExecute(fplan2.kern_cplx_mul_real,1,offs2,work2)) { CLBufferFree(memC2); CLBufferFree(memWin); CLBufferFree(memOut); return false; }
   CLBufferRead(memOut,cbuf);
   CLBufferFree(memC2); CLBufferFree(memWin); CLBufferFree(memOut);
   for(int i=0;i<n_fft;i++) { ifft1[i].re=cbuf[2*i]; ifft1[i].im=cbuf[2*i+1]; }

   Complex64 fft2[];
   if(!CLFFTExecute(plan2,ifft1,fft2,false)) return false;
   // exp(fft2) on GPU
   memC2=CLBufferCreate(fplan2.ctx,2*n_fft*sizeof(double),CL_MEM_READ_WRITE);
   memOut=CLBufferCreate(fplan2.ctx,2*n_fft*sizeof(double),CL_MEM_READ_WRITE);
   if(memC2==INVALID_HANDLE || memOut==INVALID_HANDLE)
     {
      if(memC2!=INVALID_HANDLE) CLBufferFree(memC2);
      if(memOut!=INVALID_HANDLE) CLBufferFree(memOut);
      return false;
     }
   ArrayResize(cbuf,2*n_fft);
   for(int i=0;i<n_fft;i++){ cbuf[2*i]=fft2[i].re; cbuf[2*i+1]=fft2[i].im; }
   CLBufferWrite(memC2,cbuf);
   CLSetKernelArgMem(fplan2.kern_cplx_exp,0,memC2);
   CLSetKernelArg(fplan2.kern_cplx_exp,1,n_fft);
   CLSetKernelArgMem(fplan2.kern_cplx_exp,2,memOut);
   if(!CLExecute(fplan2.kern_cplx_exp,1,offs2,work2)) { CLBufferFree(memC2); CLBufferFree(memOut); return false; }
   CLBufferRead(memOut,cbuf);
   CLBufferFree(memC2); CLBufferFree(memOut);
   for(int i=0;i<n_fft;i++) { fft2[i].re=cbuf[2*i]; fft2[i].im=cbuf[2*i+1]; }
   Complex64 ifft2[];
   if(!CLFFTExecute(plan2,fft2,ifft2,true)) return false;

   int n_out = N/2 + (N%2);
   ArrayResize(h_minimum,n_out);
   for(int i=0;i<n_out;i++) h_minimum[i]=ifft2[i].re;
   return true;
  }

#endif
