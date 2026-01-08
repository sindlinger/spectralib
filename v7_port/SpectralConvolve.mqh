#ifndef __SPECTRAL_CONVOLVE_MQH__
#define __SPECTRAL_CONVOLVE_MQH__

#include "SpectralCommon.mqh"

struct CLConvolveHandle
  {
   int ctx;
   int prog;
   int kern_conv2;
   int kern_conv3;
   int memIn1;
   int memIn2;
   int memOut;
   int lenIn1;
   int lenIn2;
   bool ready;
  };

inline void CLConvolveReset(CLConvolveHandle &h)
  {
   h.ctx=INVALID_HANDLE; h.prog=INVALID_HANDLE;
   h.kern_conv2=INVALID_HANDLE; h.kern_conv3=INVALID_HANDLE;
   h.memIn1=INVALID_HANDLE; h.memIn2=INVALID_HANDLE; h.memOut=INVALID_HANDLE;
   h.lenIn1=0; h.lenIn2=0; h.ready=false;
  }

inline void CLConvolveFree(CLConvolveHandle &h)
  {
   if(h.memIn1!=INVALID_HANDLE) { CLBufferFree(h.memIn1); h.memIn1=INVALID_HANDLE; }
   if(h.memIn2!=INVALID_HANDLE) { CLBufferFree(h.memIn2); h.memIn2=INVALID_HANDLE; }
   if(h.memOut!=INVALID_HANDLE) { CLBufferFree(h.memOut); h.memOut=INVALID_HANDLE; }
   if(h.kern_conv2!=INVALID_HANDLE) { CLKernelFree(h.kern_conv2); h.kern_conv2=INVALID_HANDLE; }
   if(h.kern_conv3!=INVALID_HANDLE) { CLKernelFree(h.kern_conv3); h.kern_conv3=INVALID_HANDLE; }
   if(h.prog!=INVALID_HANDLE) { CLProgramFree(h.prog); h.prog=INVALID_HANDLE; }
   if(h.ctx!=INVALID_HANDLE) { CLContextFree(h.ctx); h.ctx=INVALID_HANDLE; }
   h.lenIn1=0; h.lenIn2=0; h.ready=false;
  }

inline bool CLConvolveInit(CLConvolveHandle &h)
  {
   if(h.ready) return true;
   CLConvolveReset(h);
   h.ctx=CLContextCreate(CL_USE_GPU_DOUBLE_ONLY);
   if(h.ctx==INVALID_HANDLE) return false;
   string code=
   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
   "__kernel void convolve1d2o(__global const double* in1, __global const double* in2, int W, int H, int outN, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=outN) return; double temp=0.0;\n"
   "  for(int x=0;x<W;x++){ for(int y=0;y<H;y++){\n"
   "    temp += in1[i + W - x - 1] * in1[i + H - y - 1] * in2[H*x + y];\n"
   "  }} out[i]=temp; }\n"
   "__kernel void convolve1d3o(__global const double* in1, __global const double* in2, int W, int H, int D, int outN, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=outN) return; double temp=0.0;\n"
   "  for(int x=0;x<W;x++){ for(int y=0;y<H;y++){ for(int z=0;z<D;z++){\n"
   "    temp += in1[i + W - x - 1] * in1[i + H - y - 1] * in1[i + D - z - 1] * in2[(H*x + y)*D + z];\n"
   "  }}} out[i]=temp; }\n";

   h.prog=CLProgramCreate(h.ctx,code);
   if(h.prog==INVALID_HANDLE) { CLConvolveFree(h); return false; }
   h.kern_conv2=CLKernelCreate(h.prog,"convolve1d2o");
   h.kern_conv3=CLKernelCreate(h.prog,"convolve1d3o");
   if(h.kern_conv2==INVALID_HANDLE || h.kern_conv3==INVALID_HANDLE)
     { CLConvolveFree(h); return false; }
   h.ready=true;
   return true;
  }

inline bool Convolve1D2O(const double &in1[],const double &in2[][],double &out[])
  {
   int N=ArraySize(in1);
   int W=ArrayRange(in2,0);
   int H=ArrayRange(in2,1);
   if(N<=0 || W<=0 || H<=0) return false;
   int maxwh = (W>H?W:H);
   int outN = N - maxwh + 1;
   if(outN<=0) return false;

   // flatten in2 as [H*x + y]
   double in2flat[]; ArrayResize(in2flat,W*H);
   for(int x=0;x<W;x++) for(int y=0;y<H;y++) in2flat[H*x + y]=in2[x][y];

   static CLConvolveHandle h; if(!h.ready) CLConvolveReset(h);
   if(!CLConvolveInit(h)) return false;

   if(h.memIn1!=INVALID_HANDLE) CLBufferFree(h.memIn1);
   if(h.memIn2!=INVALID_HANDLE) CLBufferFree(h.memIn2);
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memIn1=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   h.memIn2=CLBufferCreate(h.ctx,(W*H)*sizeof(double),CL_MEM_READ_ONLY);
   h.memOut=CLBufferCreate(h.ctx,outN*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memIn1==INVALID_HANDLE || h.memIn2==INVALID_HANDLE || h.memOut==INVALID_HANDLE) return false;

   CLBufferWrite(h.memIn1,in1);
   CLBufferWrite(h.memIn2,in2flat);

   CLSetKernelArgMem(h.kern_conv2,0,h.memIn1);
   CLSetKernelArgMem(h.kern_conv2,1,h.memIn2);
   CLSetKernelArg(h.kern_conv2,2,W);
   CLSetKernelArg(h.kern_conv2,3,H);
   CLSetKernelArg(h.kern_conv2,4,outN);
   CLSetKernelArgMem(h.kern_conv2,5,h.memOut);
   uint offs[1]={0}; uint work[1]={(uint)outN};
   if(!CLExecute(h.kern_conv2,1,offs,work)) return false;
   ArrayResize(out,outN);
   CLBufferRead(h.memOut,out);
   return true;
  }

inline bool Convolve1D3O(const double &in1[],const double &in2[][][],double &out[])
  {
   int N=ArraySize(in1);
   int W=ArrayRange(in2,0);
   int H=ArrayRange(in2,1);
   int D=ArrayRange(in2,2);
   if(N<=0 || W<=0 || H<=0 || D<=0) return false;
   int maxwhd=W; if(H>maxwhd) maxwhd=H; if(D>maxwhd) maxwhd=D;
   int outN = N - maxwhd + 1;
   if(outN<=0) return false;

   double in2flat[]; ArrayResize(in2flat,W*H*D);
   for(int x=0;x<W;x++)
     for(int y=0;y<H;y++)
       for(int z=0;z<D;z++)
         in2flat[(H*x + y)*D + z]=in2[x][y][z];

   static CLConvolveHandle h; if(!h.ready) CLConvolveReset(h);
   if(!CLConvolveInit(h)) return false;

   if(h.memIn1!=INVALID_HANDLE) CLBufferFree(h.memIn1);
   if(h.memIn2!=INVALID_HANDLE) CLBufferFree(h.memIn2);
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memIn1=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   h.memIn2=CLBufferCreate(h.ctx,(W*H*D)*sizeof(double),CL_MEM_READ_ONLY);
   h.memOut=CLBufferCreate(h.ctx,outN*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memIn1==INVALID_HANDLE || h.memIn2==INVALID_HANDLE || h.memOut==INVALID_HANDLE) return false;

   CLBufferWrite(h.memIn1,in1);
   CLBufferWrite(h.memIn2,in2flat);

   CLSetKernelArgMem(h.kern_conv3,0,h.memIn1);
   CLSetKernelArgMem(h.kern_conv3,1,h.memIn2);
   CLSetKernelArg(h.kern_conv3,2,W);
   CLSetKernelArg(h.kern_conv3,3,H);
   CLSetKernelArg(h.kern_conv3,4,D);
   CLSetKernelArg(h.kern_conv3,5,outN);
   CLSetKernelArgMem(h.kern_conv3,6,h.memOut);
   uint offs[1]={0}; uint work[1]={(uint)outN};
   if(!CLExecute(h.kern_conv3,1,offs,work)) return false;
   ArrayResize(out,outN);
   CLBufferRead(h.memOut,out);
   return true;
  }

#endif
