#ifndef __SPECTRAL_WAVEFORMS_MQH__
#define __SPECTRAL_WAVEFORMS_MQH__

#include "SpectralCommon.mqh"
#include "SpectralOpenCLCommon.mqh"

struct CLWaveHandle
  {
   int ctx;
   int prog;
   int kern_saw;
   int kern_square;
   int kern_gauss_FF;
   int kern_gauss_FT;
   int kern_gauss_TF;
   int kern_gauss_TT;
   int kern_chirp_lin;
   int kern_chirp_quad;
   int kern_chirp_log;
   int kern_chirp_hyp;
   int kern_unit;
   int memT;
   int memW;
   int memOut1;
   int memOut2;
   int memOut3;
   int lenT;
   int lenW;
   bool ready;
  };

inline void CLWaveReset(CLWaveHandle &h)
  {
   h.ctx=INVALID_HANDLE; h.prog=INVALID_HANDLE;
   h.kern_saw=INVALID_HANDLE; h.kern_square=INVALID_HANDLE;
   h.kern_gauss_FF=INVALID_HANDLE; h.kern_gauss_FT=INVALID_HANDLE;
   h.kern_gauss_TF=INVALID_HANDLE; h.kern_gauss_TT=INVALID_HANDLE;
   h.kern_chirp_lin=INVALID_HANDLE; h.kern_chirp_quad=INVALID_HANDLE;
   h.kern_chirp_log=INVALID_HANDLE; h.kern_chirp_hyp=INVALID_HANDLE;
   h.kern_unit=INVALID_HANDLE;
   h.memT=INVALID_HANDLE; h.memW=INVALID_HANDLE;
   h.memOut1=INVALID_HANDLE; h.memOut2=INVALID_HANDLE; h.memOut3=INVALID_HANDLE;
   h.lenT=0; h.lenW=0; h.ready=false;
  }

inline void CLWaveFree(CLWaveHandle &h)
  {
   if(h.memT!=INVALID_HANDLE) { CLBufferFree(h.memT); h.memT=INVALID_HANDLE; }
   if(h.memW!=INVALID_HANDLE) { CLBufferFree(h.memW); h.memW=INVALID_HANDLE; }
   if(h.memOut1!=INVALID_HANDLE) { CLBufferFree(h.memOut1); h.memOut1=INVALID_HANDLE; }
   if(h.memOut2!=INVALID_HANDLE) { CLBufferFree(h.memOut2); h.memOut2=INVALID_HANDLE; }
   if(h.memOut3!=INVALID_HANDLE) { CLBufferFree(h.memOut3); h.memOut3=INVALID_HANDLE; }
   if(h.kern_saw!=INVALID_HANDLE) { CLKernelFree(h.kern_saw); h.kern_saw=INVALID_HANDLE; }
   if(h.kern_square!=INVALID_HANDLE) { CLKernelFree(h.kern_square); h.kern_square=INVALID_HANDLE; }
   if(h.kern_gauss_FF!=INVALID_HANDLE) { CLKernelFree(h.kern_gauss_FF); h.kern_gauss_FF=INVALID_HANDLE; }
   if(h.kern_gauss_FT!=INVALID_HANDLE) { CLKernelFree(h.kern_gauss_FT); h.kern_gauss_FT=INVALID_HANDLE; }
   if(h.kern_gauss_TF!=INVALID_HANDLE) { CLKernelFree(h.kern_gauss_TF); h.kern_gauss_TF=INVALID_HANDLE; }
   if(h.kern_gauss_TT!=INVALID_HANDLE) { CLKernelFree(h.kern_gauss_TT); h.kern_gauss_TT=INVALID_HANDLE; }
   if(h.kern_chirp_lin!=INVALID_HANDLE) { CLKernelFree(h.kern_chirp_lin); h.kern_chirp_lin=INVALID_HANDLE; }
   if(h.kern_chirp_quad!=INVALID_HANDLE) { CLKernelFree(h.kern_chirp_quad); h.kern_chirp_quad=INVALID_HANDLE; }
   if(h.kern_chirp_log!=INVALID_HANDLE) { CLKernelFree(h.kern_chirp_log); h.kern_chirp_log=INVALID_HANDLE; }
   if(h.kern_chirp_hyp!=INVALID_HANDLE) { CLKernelFree(h.kern_chirp_hyp); h.kern_chirp_hyp=INVALID_HANDLE; }
   if(h.kern_unit!=INVALID_HANDLE) { CLKernelFree(h.kern_unit); h.kern_unit=INVALID_HANDLE; }
   if(h.prog!=INVALID_HANDLE) { CLProgramFree(h.prog); h.prog=INVALID_HANDLE; }
   if(h.ctx!=INVALID_HANDLE) { CLContextFree(h.ctx); h.ctx=INVALID_HANDLE; }
   h.lenT=0; h.lenW=0; h.ready=false;
  }

inline bool CLWaveInit(CLWaveHandle &h)
  {
   if(h.ready) return true;
   CLWaveReset(h);
   h.ctx=CLCreateContextGPUFloat64("SpectralWaveforms");
   if(h.ctx==INVALID_HANDLE) return false;
   string code=
   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"#ifndef M_PI\n"
"#define M_PI 3.1415926535897932384626433832795\n"
"#endif\n"
   "__kernel void sawtooth_kernel(__global const double* t, __global const double* w, int wlen, double w0, __global double* out, int n){\n"
   "  int i=get_global_id(0); if(i>=n) return;\n"
   "  double wi = (wlen==1)? w0 : w[i];\n"
   "  double ti = t[i];\n"
   "  double outv;\n"
   "  if(wi<0.0 || wi>1.0){ outv = nan((ulong)0x7ff8000000000000UL); }\n"
   "  else { double tmod=fmod(ti, 2.0*M_PI);\n"
   "    if(tmod < (wi*2.0*M_PI)) outv = tmod/(M_PI*wi) - 1.0;\n"
   "    else outv = (M_PI*(wi+1.0) - tmod)/(M_PI*(1.0-wi));\n"
   "  }\n"
   "  out[i]=outv;\n"
   "}\n"
   "__kernel void square_kernel(__global const double* t, __global const double* w, int wlen, double w0, __global double* out, int n){\n"
   "  int i=get_global_id(0); if(i>=n) return;\n"
   "  double wi = (wlen==1)? w0 : w[i];\n"
   "  double ti = t[i];\n"
   "  double outv;\n"
   "  if(wi<0.0 || wi>1.0){ outv = nan((ulong)0x7ff8000000000000UL); }\n"
   "  else { double tmod=fmod(ti, 2.0*M_PI);\n"
   "    if(tmod < (wi*2.0*M_PI)) outv = 1.0;\n"
   "    else outv = -1.0;\n"
   "  }\n"
   "  out[i]=outv;\n"
   "}\n"
   "__kernel void gausspulse_FF(__global const double* t, double a, double fc, __global double* yI, int n){\n"
   "  int i=get_global_id(0); if(i>=n) return;\n"
   "  double ti=t[i]; double yenv=exp(-a*ti*ti); yI[i]=yenv*cos(2.0*M_PI*fc*ti);\n"
   "}\n"
   "__kernel void gausspulse_FT(__global const double* t, double a, double fc, __global double* yI, __global double* yenv, int n){\n"
   "  int i=get_global_id(0); if(i>=n) return;\n"
   "  double ti=t[i]; double env=exp(-a*ti*ti); yenv[i]=env; yI[i]=env*cos(2.0*M_PI*fc*ti);\n"
   "}\n"
   "__kernel void gausspulse_TF(__global const double* t, double a, double fc, __global double* yI, __global double* yQ, int n){\n"
   "  int i=get_global_id(0); if(i>=n) return;\n"
   "  double ti=t[i]; double env=exp(-a*ti*ti); yI[i]=env*cos(2.0*M_PI*fc*ti); yQ[i]=env*sin(2.0*M_PI*fc*ti);\n"
   "}\n"
   "__kernel void gausspulse_TT(__global const double* t, double a, double fc, __global double* yI, __global double* yQ, __global double* yenv, int n){\n"
   "  int i=get_global_id(0); if(i>=n) return;\n"
   "  double ti=t[i]; double env=exp(-a*ti*ti); yenv[i]=env; yI[i]=env*cos(2.0*M_PI*fc*ti); yQ[i]=env*sin(2.0*M_PI*fc*ti);\n"
   "}\n"
   "__kernel void chirp_lin(__global const double* t, double f0, double t1, double f1, double phi, __global double* out, int n){\n"
   "  int i=get_global_id(0); if(i>=n) return;\n"
   "  double ti=t[i]; double beta=(f1-f0)/t1; double temp=2.0*M_PI*(f0*ti + 0.5*beta*ti*ti); out[i]=cos(temp + phi);\n"
   "}\n"
   "__kernel void chirp_quad(__global const double* t, double f0, double t1, double f1, double phi, int vertex_zero, __global double* out, int n){\n"
   "  int i=get_global_id(0); if(i>=n) return;\n"
   "  double ti=t[i]; double beta=(f1-f0)/(t1*t1); double temp;\n"
   "  if(vertex_zero!=0) temp=2.0*M_PI*(f0*ti + beta*(ti*ti*ti)/3.0);\n"
   "  else temp=2.0*M_PI*(f1*ti + beta*(((t1-ti)*(t1-ti)*(t1-ti)) - (t1*t1*t1))/3.0);\n"
   "  out[i]=cos(temp + phi);\n"
   "}\n"
   "__kernel void chirp_log(__global const double* t, double f0, double t1, double f1, double phi, __global double* out, int n){\n"
   "  int i=get_global_id(0); if(i>=n) return;\n"
   "  double ti=t[i]; double temp;\n"
   "  if(f0==f1) temp=2.0*M_PI*f0*ti;\n"
   "  else { double beta=t1/log(f1/f0); temp=2.0*M_PI*beta*f0*(pow(f1/f0, ti/t1) - 1.0); }\n"
   "  out[i]=cos(temp + phi);\n"
   "}\n"
   "__kernel void chirp_hyp(__global const double* t, double f0, double t1, double f1, double phi, __global double* out, int n){\n"
   "  int i=get_global_id(0); if(i>=n) return;\n"
   "  double ti=t[i]; double temp;\n"
   "  if(f0==f1) temp=2.0*M_PI*f0*ti;\n"
   "  else { double sing=-f1*t1/(f0-f1); temp=2.0*M_PI*(-sing*f0)*log(fabs(1.0 - ti/sing)); }\n"
   "  out[i]=cos(temp + phi);\n"
   "}\n"
   "__kernel void unit_impulse_kernel(const int n, const int pos, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=n) return; out[i]=(i==pos)?1.0:0.0;\n"
   "}\n";

   string build_log="";
   h.prog=CLProgramCreate(h.ctx,code,build_log);
   if(h.prog==INVALID_HANDLE)
     {
      PrintFormat("SpectralWaveforms: CLProgramCreate failed (err=%d)", GetLastError());
      if(build_log!="") Print("SpectralWaveforms build log:\n", build_log);
      CLWaveFree(h);
      return false;
     }
   h.kern_saw=CLKernelCreate(h.prog,"sawtooth_kernel");
   h.kern_square=CLKernelCreate(h.prog,"square_kernel");
   h.kern_gauss_FF=CLKernelCreate(h.prog,"gausspulse_FF");
   h.kern_gauss_FT=CLKernelCreate(h.prog,"gausspulse_FT");
   h.kern_gauss_TF=CLKernelCreate(h.prog,"gausspulse_TF");
   h.kern_gauss_TT=CLKernelCreate(h.prog,"gausspulse_TT");
   h.kern_chirp_lin=CLKernelCreate(h.prog,"chirp_lin");
   h.kern_chirp_quad=CLKernelCreate(h.prog,"chirp_quad");
   h.kern_chirp_log=CLKernelCreate(h.prog,"chirp_log");
   h.kern_chirp_hyp=CLKernelCreate(h.prog,"chirp_hyp");
   h.kern_unit=CLKernelCreate(h.prog,"unit_impulse_kernel");
   if(h.kern_saw==INVALID_HANDLE || h.kern_square==INVALID_HANDLE ||
      h.kern_gauss_FF==INVALID_HANDLE || h.kern_gauss_FT==INVALID_HANDLE ||
      h.kern_gauss_TF==INVALID_HANDLE || h.kern_gauss_TT==INVALID_HANDLE ||
      h.kern_chirp_lin==INVALID_HANDLE || h.kern_chirp_quad==INVALID_HANDLE ||
      h.kern_chirp_log==INVALID_HANDLE || h.kern_chirp_hyp==INVALID_HANDLE ||
      h.kern_unit==INVALID_HANDLE)
     { CLWaveFree(h); return false; }
   h.ready=true;
   return true;
  }

inline bool _wave_alloc(CLWaveHandle &h,const int n,const int wlen,const bool need_w)
  {
   if(!CLWaveInit(h)) return false;
   if(h.memT==INVALID_HANDLE || h.lenT!=n)
     {
      if(h.memT!=INVALID_HANDLE) CLBufferFree(h.memT);
      h.memT=CLBufferCreate(h.ctx,n*sizeof(double),CL_MEM_READ_ONLY);
      if(h.memT==INVALID_HANDLE) return false;
      h.lenT=n;
     }
   if(need_w)
     {
      if(h.memW==INVALID_HANDLE || h.lenW!=wlen)
        {
         if(h.memW!=INVALID_HANDLE) CLBufferFree(h.memW);
         h.memW=CLBufferCreate(h.ctx,wlen*sizeof(double),CL_MEM_READ_ONLY);
         if(h.memW==INVALID_HANDLE) return false;
         h.lenW=wlen;
        }
     }
   return true;
  }

inline bool WaveSawtooth(const double &t[],const double &width[],double &out[])
  {
   int n=ArraySize(t);
   int wlen=ArraySize(width);
   if(n<=0 || wlen<=0) return false;
   if(wlen!=1 && wlen!=n) return false;
   static CLWaveHandle h; if(!h.ready) CLWaveReset(h);
   if(!_wave_alloc(h,n,wlen,true)) return false;
   CLBufferWrite(h.memT,t);
   CLBufferWrite(h.memW,width);
   if(h.memOut1!=INVALID_HANDLE) CLBufferFree(h.memOut1);
   h.memOut1=CLBufferCreate(h.ctx,n*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memOut1==INVALID_HANDLE) return false;

   CLSetKernelArgMem(h.kern_saw,0,h.memT);
   CLSetKernelArgMem(h.kern_saw,1,h.memW);
   CLSetKernelArg(h.kern_saw,2,wlen);
   CLSetKernelArg(h.kern_saw,3,width[0]);
   CLSetKernelArgMem(h.kern_saw,4,h.memOut1);
   CLSetKernelArg(h.kern_saw,5,n);
   uint offs[1]={0}; uint work[1]={(uint)n};
   if(!CLExecute(h.kern_saw,1,offs,work)) return false;
   ArrayResize(out,n);
   CLBufferRead(h.memOut1,out);
   return true;
  }

inline bool WaveSquare(const double &t[],const double &duty[],double &out[])
  {
   int n=ArraySize(t);
   int wlen=ArraySize(duty);
   if(n<=0 || wlen<=0) return false;
   if(wlen!=1 && wlen!=n) return false;
   static CLWaveHandle h; if(!h.ready) CLWaveReset(h);
   if(!_wave_alloc(h,n,wlen,true)) return false;
   CLBufferWrite(h.memT,t);
   CLBufferWrite(h.memW,duty);
   if(h.memOut1!=INVALID_HANDLE) CLBufferFree(h.memOut1);
   h.memOut1=CLBufferCreate(h.ctx,n*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memOut1==INVALID_HANDLE) return false;

   CLSetKernelArgMem(h.kern_square,0,h.memT);
   CLSetKernelArgMem(h.kern_square,1,h.memW);
   CLSetKernelArg(h.kern_square,2,wlen);
   CLSetKernelArg(h.kern_square,3,duty[0]);
   CLSetKernelArgMem(h.kern_square,4,h.memOut1);
   CLSetKernelArg(h.kern_square,5,n);
   uint offs[1]={0}; uint work[1]={(uint)n};
   if(!CLExecute(h.kern_square,1,offs,work)) return false;
   ArrayResize(out,n);
   CLBufferRead(h.memOut1,out);
   return true;
  }

inline bool WaveGausspulse(const double &t[],double fc,double bw,double bwr,bool retquad,bool retenv,
                           double &yI[],double &yQ[],double &yenv[])
  {
   if(fc<0.0 || bw<=0.0 || bwr>=0.0) return false;
   int n=ArraySize(t);
   if(n<=0) return false;
   double ref=MathPow(10.0,bwr/20.0);
   double a= -((PI*fc*bw)*(PI*fc*bw)) / (4.0*MathLog(ref));

   static CLWaveHandle h; if(!h.ready) CLWaveReset(h);
   if(!_wave_alloc(h,n,0,false)) return false;
   CLBufferWrite(h.memT,t);

   if(h.memOut1!=INVALID_HANDLE) CLBufferFree(h.memOut1);
   if(h.memOut2!=INVALID_HANDLE) CLBufferFree(h.memOut2);
   if(h.memOut3!=INVALID_HANDLE) CLBufferFree(h.memOut3);

   h.memOut1=CLBufferCreate(h.ctx,n*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memOut1==INVALID_HANDLE) return false;
   if(retquad) { h.memOut2=CLBufferCreate(h.ctx,n*sizeof(double),CL_MEM_WRITE_ONLY); if(h.memOut2==INVALID_HANDLE) return false; }
   if(retenv) { h.memOut3=CLBufferCreate(h.ctx,n*sizeof(double),CL_MEM_WRITE_ONLY); if(h.memOut3==INVALID_HANDLE) return false; }

   uint offs[1]={0}; uint work[1]={(uint)n};
   if(!retquad && !retenv)
     {
      CLSetKernelArgMem(h.kern_gauss_FF,0,h.memT);
      CLSetKernelArg(h.kern_gauss_FF,1,a);
      CLSetKernelArg(h.kern_gauss_FF,2,fc);
      CLSetKernelArgMem(h.kern_gauss_FF,3,h.memOut1);
      CLSetKernelArg(h.kern_gauss_FF,4,n);
      if(!CLExecute(h.kern_gauss_FF,1,offs,work)) return false;
     }
   else if(!retquad && retenv)
     {
      CLSetKernelArgMem(h.kern_gauss_FT,0,h.memT);
      CLSetKernelArg(h.kern_gauss_FT,1,a);
      CLSetKernelArg(h.kern_gauss_FT,2,fc);
      CLSetKernelArgMem(h.kern_gauss_FT,3,h.memOut1);
      CLSetKernelArgMem(h.kern_gauss_FT,4,h.memOut3);
      CLSetKernelArg(h.kern_gauss_FT,5,n);
      if(!CLExecute(h.kern_gauss_FT,1,offs,work)) return false;
     }
   else if(retquad && !retenv)
     {
      CLSetKernelArgMem(h.kern_gauss_TF,0,h.memT);
      CLSetKernelArg(h.kern_gauss_TF,1,a);
      CLSetKernelArg(h.kern_gauss_TF,2,fc);
      CLSetKernelArgMem(h.kern_gauss_TF,3,h.memOut1);
      CLSetKernelArgMem(h.kern_gauss_TF,4,h.memOut2);
      CLSetKernelArg(h.kern_gauss_TF,5,n);
      if(!CLExecute(h.kern_gauss_TF,1,offs,work)) return false;
     }
   else
     {
      CLSetKernelArgMem(h.kern_gauss_TT,0,h.memT);
      CLSetKernelArg(h.kern_gauss_TT,1,a);
      CLSetKernelArg(h.kern_gauss_TT,2,fc);
      CLSetKernelArgMem(h.kern_gauss_TT,3,h.memOut1);
      CLSetKernelArgMem(h.kern_gauss_TT,4,h.memOut2);
      CLSetKernelArgMem(h.kern_gauss_TT,5,h.memOut3);
      CLSetKernelArg(h.kern_gauss_TT,6,n);
      if(!CLExecute(h.kern_gauss_TT,1,offs,work)) return false;
     }

   ArrayResize(yI,n); CLBufferRead(h.memOut1,yI);
   if(retquad) { ArrayResize(yQ,n); CLBufferRead(h.memOut2,yQ); } else ArrayResize(yQ,0);
   if(retenv) { ArrayResize(yenv,n); CLBufferRead(h.memOut3,yenv); } else ArrayResize(yenv,0);
   return true;
  }

inline bool WaveGausspulseCutoff(double fc,double bw,double bwr,double tpr,double &tc)
  {
   if(fc<0.0 || bw<=0.0 || bwr>=0.0 || tpr>=0.0) return false;
   double ref=MathPow(10.0,bwr/20.0);
   double a= -((PI*fc*bw)*(PI*fc*bw)) / (4.0*MathLog(ref));
   double tref=MathPow(10.0,tpr/20.0);
   tc=MathSqrt(-MathLog(tref)/a);
   return true;
  }

inline bool WaveChirp(const double &t[],double f0,double t1,double f1,const string method,double phi_deg,bool vertex_zero,double &out[])
  {
   int n=ArraySize(t);
   if(n<=0) return false;
   string m=method;
   StringToLower(m);
   if((m=="logarithmic" || m=="log" || m=="lo") && (f0*f1<=0.0)) return false;
   if((m=="hyperbolic" || m=="hyp") && (f0==0.0 || f1==0.0)) return false;
   double phi=phi_deg*PI/180.0;

   static CLWaveHandle h; if(!h.ready) CLWaveReset(h);
   if(!_wave_alloc(h,n,0,false)) return false;
   CLBufferWrite(h.memT,t);
   if(h.memOut1!=INVALID_HANDLE) CLBufferFree(h.memOut1);
   h.memOut1=CLBufferCreate(h.ctx,n*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memOut1==INVALID_HANDLE) return false;

   uint offs[1]={0}; uint work[1]={(uint)n};
   if(m=="linear" || m=="lin" || m=="li")
     {
      CLSetKernelArgMem(h.kern_chirp_lin,0,h.memT);
      CLSetKernelArg(h.kern_chirp_lin,1,f0);
      CLSetKernelArg(h.kern_chirp_lin,2,t1);
      CLSetKernelArg(h.kern_chirp_lin,3,f1);
      CLSetKernelArg(h.kern_chirp_lin,4,phi);
      CLSetKernelArgMem(h.kern_chirp_lin,5,h.memOut1);
      CLSetKernelArg(h.kern_chirp_lin,6,n);
      if(!CLExecute(h.kern_chirp_lin,1,offs,work)) return false;
     }
   else if(m=="quadratic" || m=="quad" || m=="q")
     {
      CLSetKernelArgMem(h.kern_chirp_quad,0,h.memT);
      CLSetKernelArg(h.kern_chirp_quad,1,f0);
      CLSetKernelArg(h.kern_chirp_quad,2,t1);
      CLSetKernelArg(h.kern_chirp_quad,3,f1);
      CLSetKernelArg(h.kern_chirp_quad,4,phi);
      CLSetKernelArg(h.kern_chirp_quad,5,(int)(vertex_zero?1:0));
      CLSetKernelArgMem(h.kern_chirp_quad,6,h.memOut1);
      CLSetKernelArg(h.kern_chirp_quad,7,n);
      if(!CLExecute(h.kern_chirp_quad,1,offs,work)) return false;
     }
   else if(m=="logarithmic" || m=="log" || m=="lo")
     {
      CLSetKernelArgMem(h.kern_chirp_log,0,h.memT);
      CLSetKernelArg(h.kern_chirp_log,1,f0);
      CLSetKernelArg(h.kern_chirp_log,2,t1);
      CLSetKernelArg(h.kern_chirp_log,3,f1);
      CLSetKernelArg(h.kern_chirp_log,4,phi);
      CLSetKernelArgMem(h.kern_chirp_log,5,h.memOut1);
      CLSetKernelArg(h.kern_chirp_log,6,n);
      if(!CLExecute(h.kern_chirp_log,1,offs,work)) return false;
     }
   else if(m=="hyperbolic" || m=="hyp")
     {
      CLSetKernelArgMem(h.kern_chirp_hyp,0,h.memT);
      CLSetKernelArg(h.kern_chirp_hyp,1,f0);
      CLSetKernelArg(h.kern_chirp_hyp,2,t1);
      CLSetKernelArg(h.kern_chirp_hyp,3,f1);
      CLSetKernelArg(h.kern_chirp_hyp,4,phi);
      CLSetKernelArgMem(h.kern_chirp_hyp,5,h.memOut1);
      CLSetKernelArg(h.kern_chirp_hyp,6,n);
      if(!CLExecute(h.kern_chirp_hyp,1,offs,work)) return false;
     }
   else return false;

   ArrayResize(out,n);
   CLBufferRead(h.memOut1,out);
   return true;
  }

inline bool WaveUnitImpulse1D(const int n,const int idx,double &out[])
  {
   if(n<=0) return false;
   if(idx<0 || idx>=n) return false;
   static CLWaveHandle h; if(!h.ready) CLWaveReset(h);
   if(!CLWaveInit(h)) return false;
   if(h.memOut1!=INVALID_HANDLE) CLBufferFree(h.memOut1);
   h.memOut1=CLBufferCreate(h.ctx,n*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memOut1==INVALID_HANDLE) return false;

   CLSetKernelArg(h.kern_unit,0,n);
   CLSetKernelArg(h.kern_unit,1,idx);
   CLSetKernelArgMem(h.kern_unit,2,h.memOut1);
   uint offs[1]={0}; uint work[1]={(uint)n};
   if(!CLExecute(h.kern_unit,1,offs,work)) return false;
   ArrayResize(out,n);
   CLBufferRead(h.memOut1,out);
   return true;
  }

inline bool WaveUnitImpulseND(const int &shape[],const int &idx_in[],const bool idxMid,double &out[])
  {
   int nd=ArraySize(shape);
   if(nd<=0) return false;
   long total=1;
   for(int i=0;i<nd;i++)
     {
      if(shape[i]<=0) return false;
      total*=shape[i];
     }
   if(total>2147483647L) return false;
   int idx[];
   ArrayResize(idx,nd);
   if(idxMid)
     {
      for(int i=0;i<nd;i++) idx[i]=shape[i]/2;
     }
   else
     {
      int ilen=ArraySize(idx_in);
      if(ilen==1)
        {
         for(int i=0;i<nd;i++) idx[i]=idx_in[0];
        }
      else if(ilen==nd)
        {
         for(int i=0;i<nd;i++) idx[i]=idx_in[i];
        }
      else return false;
     }
   for(int i=0;i<nd;i++) if(idx[i]<0 || idx[i]>=shape[i]) return false;

   long pos=0;
   long stride=1;
   for(int i=nd-1;i>=0;i--)
     {
      pos += idx[i]*stride;
      stride*=shape[i];
     }
   return WaveUnitImpulse1D((int)total,(int)pos,out);
  }

#endif
