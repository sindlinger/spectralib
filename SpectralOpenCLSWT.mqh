#ifndef __SPECTRAL_OPENCL_SWT_MQH__
#define __SPECTRAL_OPENCL_SWT_MQH__

#include "SpectralOpenCLCommon.mqh"

// OpenCL SWT (approx) - periodization
struct CLSWTPlan
  {
   int ctx;
   int prog;
   int kern_swt_a;
   int memIn;
   int memF;
   int memOut;
   int lenIn;
   int lenF;
   bool ready;
  };

// Stream plan (persistent GPU buffer)
struct CLSWTStreamPlan
  {
   int ctx;
   int prog;
   int kern_swt_last;
   int memIn;
   int memF;
   int memOut2;
   int lenIn;
   int lenF;
   int head; // index do ultimo (barra 0)
   bool ready;
  };

inline void CLSWTReset(CLSWTPlan &p)
  {
   p.ctx=INVALID_HANDLE; p.prog=INVALID_HANDLE; p.kern_swt_a=INVALID_HANDLE;
   p.memIn=INVALID_HANDLE; p.memF=INVALID_HANDLE; p.memOut=INVALID_HANDLE;
   p.lenIn=0; p.lenF=0; p.ready=false;
  }

inline void CLSWTStreamReset(CLSWTStreamPlan &p)
  {
   p.ctx=INVALID_HANDLE; p.prog=INVALID_HANDLE;
   p.kern_swt_last=INVALID_HANDLE;
   p.memIn=INVALID_HANDLE; p.memF=INVALID_HANDLE; p.memOut2=INVALID_HANDLE;
   p.lenIn=0; p.lenF=0; p.head=0; p.ready=false;
  }

inline void CLSWTStreamFree(CLSWTStreamPlan &p)
  {
   if(p.memIn!=INVALID_HANDLE) { CLBufferFree(p.memIn); p.memIn=INVALID_HANDLE; }
   if(p.memF!=INVALID_HANDLE)  { CLBufferFree(p.memF); p.memF=INVALID_HANDLE; }
   if(p.memOut2!=INVALID_HANDLE){ CLBufferFree(p.memOut2); p.memOut2=INVALID_HANDLE; }
   if(p.kern_swt_last!=INVALID_HANDLE){ CLKernelFree(p.kern_swt_last); p.kern_swt_last=INVALID_HANDLE; }
   if(p.prog!=INVALID_HANDLE){ CLProgramFree(p.prog); p.prog=INVALID_HANDLE; }
   if(p.ctx!=INVALID_HANDLE){ CLContextFree(p.ctx); p.ctx=INVALID_HANDLE; }
   p.ready=false;
  }

inline void CLSWTFree(CLSWTPlan &p)
  {
   if(p.memIn!=INVALID_HANDLE) { CLBufferFree(p.memIn); p.memIn=INVALID_HANDLE; }
   if(p.memF!=INVALID_HANDLE)  { CLBufferFree(p.memF); p.memF=INVALID_HANDLE; }
   if(p.memOut!=INVALID_HANDLE){ CLBufferFree(p.memOut); p.memOut=INVALID_HANDLE; }
   if(p.kern_swt_a!=INVALID_HANDLE){ CLKernelFree(p.kern_swt_a); p.kern_swt_a=INVALID_HANDLE; }
   if(p.prog!=INVALID_HANDLE){ CLProgramFree(p.prog); p.prog=INVALID_HANDLE; }
   if(p.ctx!=INVALID_HANDLE){ CLContextFree(p.ctx); p.ctx=INVALID_HANDLE; }
   p.ready=false;
  }

inline bool CLSWTInit(CLSWTPlan &p)
  {
   if(p.ready) return true;
   CLSWTReset(p);
   p.ctx=CLCreateContextGPUFloat64("SpectralOpenCLSWT");
   if(p.ctx==INVALID_HANDLE) return false;

   string code=
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void swt_a_per(__global const double* input, int N,\n"
"                        __global const double* filt, int L, int fstep,\n"
"                        __global double* output)\n"
"{\n"
"    int k = (int)get_global_id(0);\n"
"    if(k >= N) return;\n"
"    double sum = 0.0;\n"
"    for(int t=0; t<L; ++t){\n"
"        int j = t * fstep;\n"
"        int idx = k - j;\n"
"        int m = idx % N; if(m < 0) m += N;\n"
"        sum += filt[t] * input[m];\n"
"    }\n"
"    output[k] = sum;\n"
"}\n";

   string build_log="";
   p.prog=CLProgramCreate(p.ctx,code,build_log);
   if(p.prog==INVALID_HANDLE)
     {
      PrintFormat("SWT OpenCL build failed (err=%d): %s", GetLastError(), build_log);
      CLSWTFree(p);
      return false;
     }
   p.kern_swt_a=CLKernelCreate(p.prog,"swt_a_per");
   if(p.kern_swt_a==INVALID_HANDLE){ CLSWTFree(p); return false; }
   p.ready=true;
   return true;
  }

inline bool CLSWTStreamInit(CLSWTStreamPlan &p)
  {
   if(p.ready) return true;
   CLSWTStreamReset(p);
   p.ctx=CLCreateContextGPUFloat64("SpectralOpenCLSWTStream");
   if(p.ctx==INVALID_HANDLE) return false;

   string code=
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void swt_a_last_ring(__global const double* input, int N,\n"
"                              __global const double* filt, int L, int fstep,\n"
"                              int head, __global double* out2)\n"
"{\n"
"    int k0 = N-1; int k1 = N-2; double sum0=0.0; double sum1=0.0;\n"
"    int base = head + 1; // oldest\n"
"    for(int t=0; t<L; ++t){\n"
"        int j = t * fstep;\n"
"        int idx0 = k0 - j; int idx1 = k1 - j;\n"
"        int m0 = idx0 % N; if(m0 < 0) m0 += N; m0 = (base + m0) % N;\n"
"        int m1 = idx1 % N; if(m1 < 0) m1 += N; m1 = (base + m1) % N;\n"
"        double f = filt[t];\n"
"        sum0 += f * input[m0];\n"
"        sum1 += f * input[m1];\n"
"    }\n"
"    out2[0]=sum0; out2[1]=sum1;\n"
"}\n"
;

   string build_log="";
   p.prog=CLProgramCreate(p.ctx,code,build_log);
   if(p.prog==INVALID_HANDLE)
     {
      PrintFormat("SWT Stream OpenCL build failed (err=%d): %s", GetLastError(), build_log);
      CLSWTStreamFree(p);
      return false;
     }
   p.kern_swt_last=CLKernelCreate(p.prog,"swt_a_last_ring");
   if(p.kern_swt_last==INVALID_HANDLE)
     { CLSWTStreamFree(p); return false; }
   p.ready=true;
   return true;
  }

inline bool CLSWTStreamAttach(CLSWTStreamPlan &p, const double &in[], const double &filter[])
  {
   int N=ArraySize(in);
   int L=ArraySize(filter);
   if(N<=0 || L<=0) return false;
   if(!CLSWTStreamInit(p)) return false;

   if(p.memIn==INVALID_HANDLE || p.lenIn!=N)
     {
      if(p.memIn!=INVALID_HANDLE) CLBufferFree(p.memIn);
      p.memIn=CLBufferCreate(p.ctx,N*sizeof(double),CL_MEM_READ_WRITE);
      if(p.memIn==INVALID_HANDLE) return false;
      p.lenIn=N;
     }
   if(p.memF==INVALID_HANDLE || p.lenF!=L)
     {
      if(p.memF!=INVALID_HANDLE) CLBufferFree(p.memF);
      p.memF=CLBufferCreate(p.ctx,L*sizeof(double),CL_MEM_READ_ONLY);
      if(p.memF==INVALID_HANDLE) return false;
      p.lenF=L;
     }
   if(p.memOut2==INVALID_HANDLE)
     {
      p.memOut2=CLBufferCreate(p.ctx,2*sizeof(double),CL_MEM_WRITE_ONLY);
      if(p.memOut2==INVALID_HANDLE) return false;
     }

   CLBufferWrite(p.memIn,in);
   CLBufferWrite(p.memF,filter);
   p.head = N-1; // ultimo = barra 0
   return true;
  }

inline bool CLSWTStreamUpdate(CLSWTStreamPlan &p, const int fstep, const double sample, const int is_new_bar, double &out0, double &out1)
  {
   if(!p.ready || p.memIn==INVALID_HANDLE || p.memF==INVALID_HANDLE || p.memOut2==INVALID_HANDLE) return false;
   int N=p.lenIn; int L=p.lenF;
   if(N<=0 || L<=0) return false;

   if(is_new_bar) { p.head = (p.head + 1) % N; }
   int idx = p.head;

   // update only 1 sample directly in GPU buffer
   double sval[]; ArrayResize(sval,1);
   sval[0]=sample;
   uint byte_off = (uint)(idx * sizeof(double));
   if(CLBufferWrite(p.memIn, sval, byte_off, 0, 1) == 0) return false;

   CLSetKernelArgMem(p.kern_swt_last,0,p.memIn);
   CLSetKernelArg(p.kern_swt_last,1,N);
   CLSetKernelArgMem(p.kern_swt_last,2,p.memF);
   CLSetKernelArg(p.kern_swt_last,3,L);
   CLSetKernelArg(p.kern_swt_last,4,fstep);
   CLSetKernelArg(p.kern_swt_last,5,p.head);
   CLSetKernelArgMem(p.kern_swt_last,6,p.memOut2);
   uint offs0[1]={0}; uint work0[1]={1};
   if(!CLExecute(p.kern_swt_last,1,offs0,work0)) return false;
   int st1 = CLExecutionStatus(p.kern_swt_last);
   if(st1 != 0)
     {
      static bool logged1=false;
      if(!logged1)
        {
         PrintFormat("SWT GPU: kern_swt_last status=%d err=%d", st1, GetLastError());
         logged1 = true;
        }
      return false;
     }

   double tmp[]; ArrayResize(tmp,2);
   CLBufferRead(p.memOut2,tmp);
   out0 = tmp[0];
   out1 = tmp[1];
   return true;
  }

inline bool CLSWT_A_Periodization(const double &in[], const double &filter[], const int fstep, double &out[])
  {
   int N=ArraySize(in);
   int L=ArraySize(filter);
   if(N<=0 || L<=0 || fstep<=0) return false;
   ArrayResize(out,N);
   static CLSWTPlan p; if(!p.ready) CLSWTReset(p);
   if(!CLSWTInit(p)) return false;

   if(p.memIn==INVALID_HANDLE || p.lenIn!=N)
     {
      if(p.memIn!=INVALID_HANDLE) CLBufferFree(p.memIn);
      if(p.memOut!=INVALID_HANDLE) CLBufferFree(p.memOut);
      p.memIn=CLBufferCreate(p.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
      p.memOut=CLBufferCreate(p.ctx,N*sizeof(double),CL_MEM_WRITE_ONLY);
      if(p.memIn==INVALID_HANDLE || p.memOut==INVALID_HANDLE) return false;
      p.lenIn=N;
     }
   if(p.memF==INVALID_HANDLE || p.lenF!=L)
     {
      if(p.memF!=INVALID_HANDLE) CLBufferFree(p.memF);
      p.memF=CLBufferCreate(p.ctx,L*sizeof(double),CL_MEM_READ_ONLY);
      if(p.memF==INVALID_HANDLE) return false;
      p.lenF=L;
     }

   CLBufferWrite(p.memIn,in);
   CLBufferWrite(p.memF,filter);
   CLSetKernelArgMem(p.kern_swt_a,0,p.memIn);
   CLSetKernelArg(p.kern_swt_a,1,N);
   CLSetKernelArgMem(p.kern_swt_a,2,p.memF);
   CLSetKernelArg(p.kern_swt_a,3,L);
   CLSetKernelArg(p.kern_swt_a,4,fstep);
   CLSetKernelArgMem(p.kern_swt_a,5,p.memOut);
   uint offs[1]={0}; uint work[1]={(uint)N};
   if(!CLExecute(p.kern_swt_a,1,offs,work)) return false;
   int st2 = CLExecutionStatus(p.kern_swt_a);
   if(st2 != 0)
     {
      static bool logged2=false;
      if(!logged2)
        {
         PrintFormat("SWT GPU: kern_swt_a status=%d err=%d", st2, GetLastError());
         logged2 = true;
        }
      return false;
     }
   CLBufferRead(p.memOut,out);
   return true;
  }

#endif // __SPECTRAL_OPENCL_SWT_MQH__
