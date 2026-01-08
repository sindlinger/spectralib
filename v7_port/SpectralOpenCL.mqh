#ifndef __SPECTRAL_OPENCL_MQH__
#define __SPECTRAL_OPENCL_MQH__

#include "SpectralCommon.mqh"

// Simple OpenCL helper for float64 kernels.
// MQL5 OpenCL API is used (CLContextCreate, CLProgramCreate, CLKernelCreate, etc.)

struct CLHandle
  {
   int ctx;
   int prog;
   int kern;
   int memX;
   int memY;
   int memF;
   int memP;
   bool ready;
  };

inline void CLReset(CLHandle &h)
  {
   h.ctx=INVALID_HANDLE;
   h.prog=INVALID_HANDLE;
   h.kern=INVALID_HANDLE;
   h.memX=INVALID_HANDLE;
   h.memY=INVALID_HANDLE;
   h.memF=INVALID_HANDLE;
   h.memP=INVALID_HANDLE;
   h.ready=false;
  }

inline void CLFree(CLHandle &h)
  {
   if(h.memX!=INVALID_HANDLE) { CLBufferFree(h.memX); h.memX=INVALID_HANDLE; }
   if(h.memY!=INVALID_HANDLE) { CLBufferFree(h.memY); h.memY=INVALID_HANDLE; }
   if(h.memF!=INVALID_HANDLE) { CLBufferFree(h.memF); h.memF=INVALID_HANDLE; }
   if(h.memP!=INVALID_HANDLE) { CLBufferFree(h.memP); h.memP=INVALID_HANDLE; }
   if(h.kern!=INVALID_HANDLE) { CLKernelFree(h.kern); h.kern=INVALID_HANDLE; }
   if(h.prog!=INVALID_HANDLE) { CLProgramFree(h.prog); h.prog=INVALID_HANDLE; }
   if(h.ctx!=INVALID_HANDLE)  { CLContextFree(h.ctx); h.ctx=INVALID_HANDLE; }
   h.ready=false;
  }

inline bool CLLombInit(CLHandle &h)
  {
   if(h.ready) return true;
   CLReset(h);
   h.ctx=CLContextCreate(CL_USE_GPU_DOUBLE_ONLY);
   if(h.ctx==INVALID_HANDLE) return false;

   string code=
   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
   "__kernel void lombscargle_d64(const int x_shape, const int freqs_shape,\n"
   "  __global const double* x, __global const double* y,\n"
   "  __global const double* freqs, __global double* pgram,\n"
   "  __global const double* y_dot){\n"
   "  int tid = (int)(get_global_id(0));\n"
   "  int stride = (int)get_global_size(0);\n"
   "  double yD = (y_dot[0]==0.0)?1.0:(2.0/y_dot[0]);\n"
   "  for(int k=tid;k<freqs_shape;k+=stride){\n"
   "    double freq=freqs[k];\n"
   "    double xc=0.0, xs=0.0, cc=0.0, ss=0.0, cs=0.0;\n"
   "    for(int j=0;j<x_shape;j++){\n"
   "      double s = sin(freq*x[j]);\n"
   "      double c = cos(freq*x[j]);\n"
   "      xc += y[j]*c;\n"
   "      xs += y[j]*s;\n"
   "      cc += c*c; ss += s*s; cs += c*s;\n"
   "    }\n"
   "    double tau = atan2(2.0*cs, cc-ss)/(2.0*freq);\n"
   "    double s_tau = sin(freq*tau);\n"
   "    double c_tau = cos(freq*tau);\n"
   "    double c_tau2 = c_tau*c_tau;\n"
   "    double s_tau2 = s_tau*s_tau;\n"
   "    double cs_tau = 2.0*c_tau*s_tau;\n"
   "    double term1 = (c_tau*xc + s_tau*xs);\n"
   "    double term2 = (c_tau*xs - s_tau*xc);\n"
   "    double denom1 = (c_tau2*cc + cs_tau*cs + s_tau2*ss);\n"
   "    double denom2 = (c_tau2*ss - cs_tau*cs + s_tau2*cc);\n"
   "    double p = 0.5*((term1*term1)/denom1 + (term2*term2)/denom2);\n"
   "    pgram[k]=p*yD;\n"
   "  }\n"
   "}\n";

   h.prog=CLProgramCreate(h.ctx,code);
   if(h.prog==INVALID_HANDLE) { CLFree(h); return false; }
   h.kern=CLKernelCreate(h.prog,"lombscargle_d64");
   if(h.kern==INVALID_HANDLE) { CLFree(h); return false; }
   h.ready=true;
   return true;
  }

inline bool CLLombscargle(CLHandle &h,const double &x[],const double &y[],const double &freqs[],double &pgram[])
  {
   int nx=ArraySize(x);
   int nf=ArraySize(freqs);
   if(nx<=0 || nf<=0) return false;
   if(!CLLombInit(h)) return false;
   if(h.memX!=INVALID_HANDLE) CLBufferFree(h.memX);
   if(h.memY!=INVALID_HANDLE) CLBufferFree(h.memY);
   if(h.memF!=INVALID_HANDLE) CLBufferFree(h.memF);
   if(h.memP!=INVALID_HANDLE) CLBufferFree(h.memP);
   h.memX=CLBufferCreate(h.ctx,nx*sizeof(double),CL_MEM_READ_ONLY);
   h.memY=CLBufferCreate(h.ctx,nx*sizeof(double),CL_MEM_READ_ONLY);
   h.memF=CLBufferCreate(h.ctx,nf*sizeof(double),CL_MEM_READ_ONLY);
   h.memP=CLBufferCreate(h.ctx,nf*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memX==INVALID_HANDLE || h.memY==INVALID_HANDLE || h.memF==INVALID_HANDLE || h.memP==INVALID_HANDLE)
     { CLFree(h); return false; }
   CLBufferWrite(h.memX,x);
   CLBufferWrite(h.memY,y);
   CLBufferWrite(h.memF,freqs);
   double ydotArr[1];
   double ydot=0.0;
   for(int i=0;i<nx;i++) ydot+=y[i]*y[i];
   ydotArr[0]=ydot;
   int memYdot=CLBufferCreate(h.ctx,sizeof(double),CL_MEM_READ_ONLY);
   CLBufferWrite(memYdot,ydotArr);

   CLSetKernelArg(h.kern,0,nx);
   CLSetKernelArg(h.kern,1,nf);
   CLSetKernelArgMem(h.kern,2,h.memX);
   CLSetKernelArgMem(h.kern,3,h.memY);
   CLSetKernelArgMem(h.kern,4,h.memF);
   CLSetKernelArgMem(h.kern,5,h.memP);
   CLSetKernelArgMem(h.kern,6,memYdot);

   uint offs[1]={0};
   uint work[1]={(uint)nf};
   bool ok=CLExecute(h.kern,1,offs,work);
   CLBufferFree(memYdot);
   if(!ok) return false;
   ArrayResize(pgram,nf);
   CLBufferRead(h.memP,pgram);
   return true;
  }

#endif
