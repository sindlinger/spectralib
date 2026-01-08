#ifndef __SPECTRAL_OPENCL_FFT_MQH__
#define __SPECTRAL_OPENCL_FFT_MQH__

#include "SpectralCommon.mqh"

struct CLFFTPlan
  {
   int ctx;
   int prog;
   int kern_bitrev;
   int kern_stage;
   int kern_scale;
   int kern_dft;
   int memA;
   int memB;
   int N;
   bool ready;
  };

inline void CLFFTReset(CLFFTPlan &p)
  {
   p.ctx=INVALID_HANDLE;
   p.prog=INVALID_HANDLE;
   p.kern_bitrev=INVALID_HANDLE;
   p.kern_stage=INVALID_HANDLE;
   p.kern_scale=INVALID_HANDLE;
   p.kern_dft=INVALID_HANDLE;
   p.memA=INVALID_HANDLE;
   p.memB=INVALID_HANDLE;
   p.N=0;
   p.ready=false;
  }

inline void CLFFTFree(CLFFTPlan &p)
  {
   if(p.memA!=INVALID_HANDLE) { CLBufferFree(p.memA); p.memA=INVALID_HANDLE; }
   if(p.memB!=INVALID_HANDLE) { CLBufferFree(p.memB); p.memB=INVALID_HANDLE; }
   if(p.kern_bitrev!=INVALID_HANDLE) { CLKernelFree(p.kern_bitrev); p.kern_bitrev=INVALID_HANDLE; }
   if(p.kern_stage!=INVALID_HANDLE) { CLKernelFree(p.kern_stage); p.kern_stage=INVALID_HANDLE; }
   if(p.kern_scale!=INVALID_HANDLE) { CLKernelFree(p.kern_scale); p.kern_scale=INVALID_HANDLE; }
   if(p.kern_dft!=INVALID_HANDLE) { CLKernelFree(p.kern_dft); p.kern_dft=INVALID_HANDLE; }
   if(p.prog!=INVALID_HANDLE) { CLProgramFree(p.prog); p.prog=INVALID_HANDLE; }
   if(p.ctx!=INVALID_HANDLE) { CLContextFree(p.ctx); p.ctx=INVALID_HANDLE; }
   p.N=0; p.ready=false;
  }

inline bool CLFFTInit(CLFFTPlan &p,const int N)
  {
   if(p.ready && p.N==N) return true;
   CLFFTFree(p);
   CLFFTReset(p);
   if(N<=1) return false;
   p.ctx=CLContextCreate(CL_USE_GPU_DOUBLE_ONLY);
   if(p.ctx==INVALID_HANDLE) return false;

   string code=
   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
   "inline uint bitrev(uint x, uint bits){\n"
   "  uint y=0; for(uint i=0;i<bits;i++){ y=(y<<1) | (x & 1); x>>=1; } return y; }\n"
   "__kernel void bit_reverse(__global const double2* in, __global double2* out, int N, int bits){\n"
   "  int i=get_global_id(0); if(i>=N) return; uint r=bitrev((uint)i,(uint)bits); out[r]=in[i]; }\n"
   "__kernel void fft_stage(__global const double2* in, __global double2* out, int N, int m, int inverse){\n"
   "  int i=get_global_id(0); int half=m>>1; int total=N>>1; if(i>=total) return;\n"
   "  int j=i%half; int block=i/half; int k=block*m + j;\n"
   "  double angle = (inverse? 2.0 : -2.0) * M_PI * (double)j / (double)m;\n"
   "  double c=cos(angle); double s=sin(angle);\n"
   "  double2 a=in[k]; double2 b=in[k+half];\n"
   "  double2 t = (double2)(b.x*c - b.y*s, b.x*s + b.y*c);\n"
   "  out[k] = (double2)(a.x + t.x, a.y + t.y);\n"
   "  out[k+half] = (double2)(a.x - t.x, a.y - t.y);\n"
   "}\n"
   "__kernel void fft_scale(__global double2* data, int N, double invN){\n"
   "  int i=get_global_id(0); if(i>=N) return; data[i].x*=invN; data[i].y*=invN; }\n"
   "__kernel void dft_complex(__global const double2* in, __global double2* out, int N, int inverse){\n"
   "  int k=get_global_id(0); if(k>=N) return; double sign = (inverse!=0)? 1.0 : -1.0;\n"
   "  double2 sum=(double2)(0.0,0.0);\n"
   "  for(int n=0;n<N;n++){\n"
   "    double ang = sign * 2.0 * M_PI * ((double)k * (double)n) / (double)N;\n"
   "    double c=cos(ang); double s=sin(ang);\n"
   "    double2 v=in[n]; sum.x += v.x*c - v.y*s; sum.y += v.x*s + v.y*c;\n"
   "  }\n"
   "  if(inverse!=0){ sum.x/= (double)N; sum.y/=(double)N; }\n"
   "  out[k]=sum; }\n";

   p.prog=CLProgramCreate(p.ctx,code);
   if(p.prog==INVALID_HANDLE) { CLFFTFree(p); return false; }
   p.kern_bitrev=CLKernelCreate(p.prog,"bit_reverse");
   p.kern_stage=CLKernelCreate(p.prog,"fft_stage");
   p.kern_scale=CLKernelCreate(p.prog,"fft_scale");
   p.kern_dft=CLKernelCreate(p.prog,"dft_complex");
   if(p.kern_bitrev==INVALID_HANDLE || p.kern_stage==INVALID_HANDLE || p.kern_scale==INVALID_HANDLE || p.kern_dft==INVALID_HANDLE)
     { CLFFTFree(p); return false; }
   p.memA=CLBufferCreate(p.ctx,N*sizeof(double)*2,CL_MEM_READ_WRITE);
   p.memB=CLBufferCreate(p.ctx,N*sizeof(double)*2,CL_MEM_READ_WRITE);
   if(p.memA==INVALID_HANDLE || p.memB==INVALID_HANDLE)
     { CLFFTFree(p); return false; }
   p.N=N;
   p.ready=true;
   return true;
  }

inline void _pack_complex(const Complex64 &in[],double &buf[])
  {
   int N=ArraySize(in);
   ArrayResize(buf,2*N);
   for(int i=0;i<N;i++)
     { buf[2*i]=in[i].re; buf[2*i+1]=in[i].im; }
  }

inline void _unpack_complex(const double &buf[],Complex64 &out[])
  {
   int N=ArraySize(buf)/2;
   ArrayResize(out,N);
   for(int i=0;i<N;i++)
     { out[i]=Cx(buf[2*i],buf[2*i+1]); }
  }

inline bool CLFFTExecute(CLFFTPlan &p,const Complex64 &in[],Complex64 &out[],const bool inverse)
  {
   int N=ArraySize(in);
   if(!CLFFTInit(p,N)) return false;
   bool pow2 = ((N & (N-1))==0);
   double buf[];
   _pack_complex(in,buf);
   CLBufferWrite(p.memA,buf);

   if(!pow2)
     {
      CLSetKernelArgMem(p.kern_dft,0,p.memA);
      CLSetKernelArgMem(p.kern_dft,1,p.memB);
      CLSetKernelArg(p.kern_dft,2,N);
      CLSetKernelArg(p.kern_dft,3,(int)(inverse?1:0));
      uint offs0[1]={0}; uint work0[1]={(uint)N};
      if(!CLExecute(p.kern_dft,1,offs0,work0)) return false;
      CLBufferRead(p.memB,buf);
      _unpack_complex(buf,out);
      return true;
     }

   // bit-reversal into memB
   int bits=0; int tmp=N;
   while(tmp>1){ bits++; tmp>>=1; }
   CLSetKernelArgMem(p.kern_bitrev,0,p.memA);
   CLSetKernelArgMem(p.kern_bitrev,1,p.memB);
   CLSetKernelArg(p.kern_bitrev,2,N);
   CLSetKernelArg(p.kern_bitrev,3,bits);
   uint offs[1]={0}; uint work[1]={(uint)N};
   if(!CLExecute(p.kern_bitrev,1,offs,work)) return false;

   // stages ping-pong
   int m=2;
   bool toggle=false;
   while(m<=N)
     {
      int half=m>>1;
      int total=N>>1;
      int inMem= toggle ? p.memA : p.memB;
      int outMem= toggle ? p.memB : p.memA;
      CLSetKernelArgMem(p.kern_stage,0,inMem);
      CLSetKernelArgMem(p.kern_stage,1,outMem);
      CLSetKernelArg(p.kern_stage,2,N);
      CLSetKernelArg(p.kern_stage,3,m);
      CLSetKernelArg(p.kern_stage,4,(int)(inverse?1:0));
      uint work2[1]={(uint)total};
      if(!CLExecute(p.kern_stage,1,offs,work2)) return false;
      toggle=!toggle;
      m<<=1;
     }
   int finalMem = toggle ? p.memB : p.memA;
   if(inverse)
     {
      double invN=1.0/(double)N;
      CLSetKernelArgMem(p.kern_scale,0,finalMem);
      CLSetKernelArg(p.kern_scale,1,N);
      CLSetKernelArg(p.kern_scale,2,invN);
      if(!CLExecute(p.kern_scale,1,offs,work)) return false;
     }
   CLBufferRead(finalMem,buf);
   _unpack_complex(buf,out);
   return true;
  }

inline void CLFFTRealForward(CLFFTPlan &p,const double &x[],Complex64 &out[])
  {
   int N=ArraySize(x);
   Complex64 tmp[];
   ArrayResize(tmp,N);
   for(int i=0;i<N;i++) tmp[i]=Cx(x[i],0.0);
   CLFFTExecute(p,tmp,out,false);
  }

inline void CLFFTRealInverse(CLFFTPlan &p,const Complex64 &Xhalf[],double &out[])
  {
   int Nh=ArraySize(Xhalf);
   int N=(Nh-1)*2;
   Complex64 full[];
   ArrayResize(full,N);
   // fill
   for(int k=0;k<Nh;k++) full[k]=Xhalf[k];
   for(int k=1;k<Nh-1;k++) full[N-k]=Cx(Xhalf[k].re,-Xhalf[k].im);
   Complex64 tmp[];
   CLFFTExecute(p,full,tmp,true);
   ArrayResize(out,N);
   for(int i=0;i<N;i++) out[i]=tmp[i].re;
  }

#endif
