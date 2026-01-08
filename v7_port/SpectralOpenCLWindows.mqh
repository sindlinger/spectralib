#ifndef __SPECTRAL_OPENCL_WINDOWS_MQH__
#define __SPECTRAL_OPENCL_WINDOWS_MQH__

#include "SpectralCommon.mqh"
#include "SpectralOpenCLFFT.mqh"
#include "SpectralWindows.mqh"

enum CLWindowType
  {
   WIN_BOXCAR=0,
   WIN_TRIANG=1,
   WIN_PARZEN=2,
   WIN_BOHMAN=3,
   WIN_BLACKMAN=4,
   WIN_NUTTALL=5,
   WIN_BLACKMANHARRIS=6,
   WIN_FLATTOP=7,
   WIN_BARTLETT=8,
   WIN_HANN=9,
   WIN_TUKEY=10,
   WIN_BARTHANN=11,
   WIN_GENERAL_HAMMING=12,
   WIN_HAMMING=13,
   WIN_KAISER=14,
   WIN_GAUSSIAN=15,
   WIN_GENERAL_GAUSSIAN=16,
   WIN_COSINE=17,
   WIN_EXPONENTIAL=18,
   WIN_GENERAL_COSINE=19,
   WIN_CHEBWIN=20,
   WIN_TAYLOR=21
  };

struct CLWinHandle
  {
   int ctx;
   int prog;
   int kern;
   int memParams;
   int memCoeffs;
   int memOut;
   bool ready;
  };

inline void CLWinReset(CLWinHandle &h)
  {
   h.ctx=INVALID_HANDLE; h.prog=INVALID_HANDLE; h.kern=INVALID_HANDLE;
   h.memParams=INVALID_HANDLE; h.memCoeffs=INVALID_HANDLE; h.memOut=INVALID_HANDLE;
   h.ready=false;
  }

inline void CLWinFree(CLWinHandle &h)
  {
   if(h.memParams!=INVALID_HANDLE) { CLBufferFree(h.memParams); h.memParams=INVALID_HANDLE; }
   if(h.memCoeffs!=INVALID_HANDLE) { CLBufferFree(h.memCoeffs); h.memCoeffs=INVALID_HANDLE; }
   if(h.memOut!=INVALID_HANDLE) { CLBufferFree(h.memOut); h.memOut=INVALID_HANDLE; }
   if(h.kern!=INVALID_HANDLE) { CLKernelFree(h.kern); h.kern=INVALID_HANDLE; }
   if(h.prog!=INVALID_HANDLE) { CLProgramFree(h.prog); h.prog=INVALID_HANDLE; }
   if(h.ctx!=INVALID_HANDLE) { CLContextFree(h.ctx); h.ctx=INVALID_HANDLE; }
   h.ready=false;
  }

inline bool CLWinInit(CLWinHandle &h)
  {
   if(h.ready) return true;
   CLWinReset(h);
   h.ctx=CLContextCreate(CL_USE_GPU_DOUBLE_ONLY);
   if(h.ctx==INVALID_HANDLE) return false;
   string code=
   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
   "double bessel_i0(double x){\n"
   "  double ax=fabs(x);\n"
   "  if(ax<3.75){ double y=x/3.75; y*=y; return 1.0 + y*(3.5156229 + y*(3.0899424 + y*(1.2067492 + y*(0.2659732 + y*(0.0360768 + y*0.0045813))))); }\n"
   "  double y=3.75/ax; return (exp(ax)/sqrt(ax))*(0.39894228 + y*(0.01328592 + y*(0.00225319 + y*(-0.00157565 + y*(0.00916281 + y*(-0.02057706 + y*(0.02635537 + y*(-0.01647633 + y*0.00392377))))))));\n"
   "}\n"
   "__kernel void win_core(int type, int M, int sym, __global const double* params, int ncoeff,\n"
   "  __global const double* coeffs, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=M) return;\n"
   "  double N=(double)M; double w=0.0; double half=(N-1.0)/2.0;\n"
   "  if(type==0){ w=1.0; }\n"
   "  else if(type==1){ w=1.0 - fabs((i-half)/((N+1.0)/2.0)); }\n"
   "  else if(type==2){ double x=fabs((i-half)/(half+1.0)); if(x<=0.5) w=1.0-6.0*x*x+6.0*x*x*x; else if(x<=1.0) w=2.0*pow(1.0-x,3.0); else w=0.0; }\n"
   "  else if(type==3){ double x=fabs((i-half)/half); w=(1.0-x)*cos(M_PI*x) + (1.0/M_PI)*sin(M_PI*x); }\n"
   "  else if(type==4){ double ang=2.0*M_PI*i/(N-1.0); w=0.42-0.5*cos(ang)+0.08*cos(2.0*ang); }\n"
   "  else if(type==5){ double ang=2.0*M_PI*i/(N-1.0); w=0.355768-0.487396*cos(ang)+0.144232*cos(2.0*ang)-0.012604*cos(3.0*ang); }\n"
   "  else if(type==6){ double ang=2.0*M_PI*i/(N-1.0); w=0.35875-0.48829*cos(ang)+0.14128*cos(2.0*ang)-0.01168*cos(3.0*ang); }\n"
   "  else if(type==7){ double ang=2.0*M_PI*i/(N-1.0); w=1.0-1.93*cos(ang)+1.29*cos(2.0*ang)-0.388*cos(3.0*ang)+0.0322*cos(4.0*ang); }\n"
   "  else if(type==8){ w=1.0 - fabs((i-half)/half); }\n"
   "  else if(type==9){ double ang=2.0*M_PI*i/(N-1.0); w=0.5-0.5*cos(ang); }\n"
   "  else if(type==10){ double alpha=params[0]; if(alpha<=0.0) w=1.0; else if(alpha>=1.0){ double ang=2.0*M_PI*i/(N-1.0); w=0.5-0.5*cos(ang);} else { double edge=alpha*(N-1.0)/2.0; if(i<edge){ double ang=M_PI*(2.0*i/alpha/(N-1.0)-1.0); w=0.5*(1.0+cos(ang)); } else if(i<=(N-1.0)*(1.0-alpha/2.0)) w=1.0; else { double ang=M_PI*(2.0*i/alpha/(N-1.0)-2.0/alpha+1.0); w=0.5*(1.0+cos(ang)); }} }\n"
   "  else if(type==11){ double x=fabs((i-half)/half); w=0.62-0.48*x+0.38*cos(M_PI*x); }\n"
   "  else if(type==12){ double alpha=params[0]; double ang=2.0*M_PI*i/(N-1.0); w=alpha-(1.0-alpha)*cos(ang); }\n"
   "  else if(type==13){ double ang=2.0*M_PI*i/(N-1.0); w=0.54-0.46*cos(ang); }\n"
   "  else if(type==14){ double beta=params[0]; double r=2.0*i/(N-1.0)-1.0; w=bessel_i0(beta*sqrt(1.0-r*r))/bessel_i0(beta); }\n"
   "  else if(type==15){ double std=params[0]; double x=(i-half)/std; w=exp(-0.5*x*x); }\n"
   "  else if(type==16){ double p=params[0]; double sig=params[1]; double x=fabs((i-half)/sig); w=exp(-0.5*pow(x,2.0*p)); }\n"
   "  else if(type==17){ w=sin(M_PI/N*(i+0.5)); }\n"
   "  else if(type==18){ double tau=params[0]; double center=params[1]; if(center<0.0) center=(N-1.0)/2.0; w=exp(-fabs(i-center)/tau); }\n"
   "  else if(type==19){ double delta=2.0*M_PI/(N-1.0); double fac=-M_PI + delta*i; double temp=0.0; for(int k=0;k<ncoeff;k++){ temp += coeffs[k]*cos((double)k*fac);} w=temp; }\n"
   "  out[i]=w; }\n";

   h.prog=CLProgramCreate(h.ctx,code);
   if(h.prog==INVALID_HANDLE) { CLWinFree(h); return false; }
   h.kern=CLKernelCreate(h.prog,"win_core");
   if(h.kern==INVALID_HANDLE) { CLWinFree(h); return false; }
   h.ready=true;
   return true;
  }

inline bool CLWindowGenerate(CLWinHandle &h,const int type,const int M,const bool sym,
                             const double &params[],const double &coeffs[],double &out[])
  {
   if(!CLWinInit(h)) return false;
   int Mx=M;
   bool trunc=false;
   if(!sym) { Mx=M+1; trunc=true; }
   int ncoeff=ArraySize(coeffs);
   int nparams=ArraySize(params);

   if(h.memParams!=INVALID_HANDLE) CLBufferFree(h.memParams);
   if(h.memCoeffs!=INVALID_HANDLE) CLBufferFree(h.memCoeffs);
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memParams=CLBufferCreate(h.ctx,MathMax(1,nparams)*sizeof(double),CL_MEM_READ_ONLY);
   h.memCoeffs=CLBufferCreate(h.ctx,MathMax(1,ncoeff)*sizeof(double),CL_MEM_READ_ONLY);
   h.memOut=CLBufferCreate(h.ctx,Mx*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memParams==INVALID_HANDLE || h.memCoeffs==INVALID_HANDLE || h.memOut==INVALID_HANDLE)
     return false;
   if(nparams>0) CLBufferWrite(h.memParams,params);
   if(ncoeff>0) CLBufferWrite(h.memCoeffs,coeffs);

   CLSetKernelArg(h.kern,0,type);
   CLSetKernelArg(h.kern,1,Mx);
   CLSetKernelArg(h.kern,2,(int)(sym?1:0));
   CLSetKernelArgMem(h.kern,3,h.memParams);
   CLSetKernelArg(h.kern,4,ncoeff);
   CLSetKernelArgMem(h.kern,5,h.memCoeffs);
   CLSetKernelArgMem(h.kern,6,h.memOut);
   uint offs[1]={0}; uint work[1]={(uint)Mx};
   if(!CLExecute(h.kern,1,offs,work)) return false;
   double tmp[];
   ArrayResize(tmp,Mx);
   CLBufferRead(h.memOut,tmp);
   if(trunc)
     {
      ArrayResize(out,M);
      for(int i=0;i<M;i++) out[i]=tmp[i];
     }
   else
     {
      ArrayResize(out,Mx);
      ArrayCopy(out,tmp);
     }
   return true;
  }

#endif
