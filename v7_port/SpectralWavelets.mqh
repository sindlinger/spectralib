#ifndef __SPECTRAL_WAVELETS_MQH__
#define __SPECTRAL_WAVELETS_MQH__

#include "SpectralCommon.mqh"

struct CLWaveletHandle
  {
   int ctx;
   int prog;
   int kern_qmf;
   int kern_morlet;
   int kern_morlet2;
   int kern_ricker;
   int kern_cwt_ricker;
   int kern_cwt_morlet2;
   int memA;
   int memB;
   int memOut;
   int memOut2;
   int lenA;
   bool ready;
  };

inline void CLWaveletReset(CLWaveletHandle &h)
  {
   h.ctx=INVALID_HANDLE; h.prog=INVALID_HANDLE;
   h.kern_qmf=INVALID_HANDLE; h.kern_morlet=INVALID_HANDLE; h.kern_morlet2=INVALID_HANDLE;
   h.kern_ricker=INVALID_HANDLE; h.kern_cwt_ricker=INVALID_HANDLE; h.kern_cwt_morlet2=INVALID_HANDLE;
   h.memA=INVALID_HANDLE; h.memB=INVALID_HANDLE; h.memOut=INVALID_HANDLE; h.memOut2=INVALID_HANDLE;
   h.lenA=0; h.ready=false;
  }

inline void CLWaveletFree(CLWaveletHandle &h)
  {
   if(h.memA!=INVALID_HANDLE) { CLBufferFree(h.memA); h.memA=INVALID_HANDLE; }
   if(h.memB!=INVALID_HANDLE) { CLBufferFree(h.memB); h.memB=INVALID_HANDLE; }
   if(h.memOut!=INVALID_HANDLE) { CLBufferFree(h.memOut); h.memOut=INVALID_HANDLE; }
   if(h.memOut2!=INVALID_HANDLE) { CLBufferFree(h.memOut2); h.memOut2=INVALID_HANDLE; }
   if(h.kern_qmf!=INVALID_HANDLE) { CLKernelFree(h.kern_qmf); h.kern_qmf=INVALID_HANDLE; }
   if(h.kern_morlet!=INVALID_HANDLE) { CLKernelFree(h.kern_morlet); h.kern_morlet=INVALID_HANDLE; }
   if(h.kern_morlet2!=INVALID_HANDLE) { CLKernelFree(h.kern_morlet2); h.kern_morlet2=INVALID_HANDLE; }
   if(h.kern_ricker!=INVALID_HANDLE) { CLKernelFree(h.kern_ricker); h.kern_ricker=INVALID_HANDLE; }
   if(h.kern_cwt_ricker!=INVALID_HANDLE) { CLKernelFree(h.kern_cwt_ricker); h.kern_cwt_ricker=INVALID_HANDLE; }
   if(h.kern_cwt_morlet2!=INVALID_HANDLE) { CLKernelFree(h.kern_cwt_morlet2); h.kern_cwt_morlet2=INVALID_HANDLE; }
   if(h.prog!=INVALID_HANDLE) { CLProgramFree(h.prog); h.prog=INVALID_HANDLE; }
   if(h.ctx!=INVALID_HANDLE) { CLContextFree(h.ctx); h.ctx=INVALID_HANDLE; }
   h.lenA=0; h.ready=false;
  }

inline bool CLWaveletInit(CLWaveletHandle &h)
  {
   if(h.ready) return true;
   CLWaveletReset(h);
   h.ctx=CLContextCreate(CL_USE_GPU_DOUBLE_ONLY);
   if(h.ctx==INVALID_HANDLE) return false;
   string code=
   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
   "__kernel void qmf_kernel(__global const double* coef, __global double* out, int n){\n"
   "  int i=get_global_id(0); if(i>=n) return; int sign=(i & 1)? -1: 1; out[i]=coef[n-1-i]*((double)sign); }\n"
   "__kernel void morlet_kernel(int M, double w, double s, int complete, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=M) return; double end=s*2.0*M_PI; double start=-s*2.0*M_PI;\n"
   "  double delta=(end-start)/(double)(M-1); double x=start + delta*(double)i;\n"
   "  double2 temp=(double2)(cos(w*x), sin(w*x)); if(complete!=0){ temp.x -= exp(-0.5*(w*w)); }\n"
   "  double ga=exp(-0.5*x*x); double scale=pow(M_PI,-0.25);\n"
   "  out[i]=(double2)(temp.x*ga*scale, temp.y*ga*scale); }\n"
   "__kernel void morlet2_kernel(int M, double s, double w, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=M) return; double x=((double)i - (double)(M-1)*0.5)/s;\n"
   "  double2 temp=(double2)(cos(w*x), sin(w*x)); double ga=exp(-0.5*x*x);\n"
   "  double scale=pow(M_PI,-0.25)*sqrt(1.0/s); out[i]=(double2)(temp.x*ga*scale, temp.y*ga*scale); }\n"
   "__kernel void ricker_kernel(int M, double a, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=M) return; double vec=(double)i - (double)(M-1)*0.5; double xsq=vec*vec;\n"
   "  double wsq=a*a; double A=2.0/(sqrt(3.0*a)*pow(M_PI,0.25)); double mod=1.0 - xsq/wsq;\n"
   "  double ga=exp(-xsq/(2.0*wsq)); out[i]=A*mod*ga; }\n"
   "__kernel void cwt_ricker(__global const double* data, int N, double width, int row, __global double* out){\n"
   "  int n=get_global_id(0); if(n>=N) return; int wi=(int)floor(width); if(wi<1) wi=1;\n"
   "  int L=10*wi; if(L>N) L=N; double half=((double)(L-1))*0.5; double wsq=width*width;\n"
   "  double A=2.0/(sqrt(3.0*width)*pow(M_PI,0.25)); double sum=0.0; int start=n - (int)floor(L*0.5);\n"
   "  for(int k=0;k<L;k++){ int idx=start + k; if(idx>=0 && idx<N){ double x=half - (double)k;\n"
   "    double xsq=x*x; double mod=1.0 - xsq/wsq; double ga=exp(-xsq/(2.0*wsq)); double wv=A*mod*ga;\n"
   "    sum += data[idx]*wv; }} out[row*N + n]=sum; }\n"
   "__kernel void cwt_morlet2(__global const double* data, int N, double width, double w, int row, __global double2* out){\n"
   "  int n=get_global_id(0); if(n>=N) return; int wi=(int)floor(width); if(wi<1) wi=1;\n"
   "  int L=10*wi; if(L>N) L=N; double half=((double)(L-1))*0.5; double sumr=0.0; double sumi=0.0;\n"
   "  double scale=pow(M_PI,-0.25)*sqrt(1.0/width); int start=n - (int)floor(L*0.5);\n"
   "  for(int k=0;k<L;k++){ int idx=start + k; if(idx>=0 && idx<N){ double x=(half - (double)k)/width;\n"
   "    double ga=exp(-0.5*x*x); double wr=cos(w*x); double wi2=-sin(w*x);\n"
   "    double wvr=wr*ga*scale; double wvi=wi2*ga*scale; sumr += data[idx]*wvr; sumi += data[idx]*wvi; }}\n"
   "  out[row*N + n]=(double2)(sumr,sumi); }\n";

   h.prog=CLProgramCreate(h.ctx,code);
   if(h.prog==INVALID_HANDLE) { CLWaveletFree(h); return false; }
   h.kern_qmf=CLKernelCreate(h.prog,"qmf_kernel");
   h.kern_morlet=CLKernelCreate(h.prog,"morlet_kernel");
   h.kern_morlet2=CLKernelCreate(h.prog,"morlet2_kernel");
   h.kern_ricker=CLKernelCreate(h.prog,"ricker_kernel");
   h.kern_cwt_ricker=CLKernelCreate(h.prog,"cwt_ricker");
   h.kern_cwt_morlet2=CLKernelCreate(h.prog,"cwt_morlet2");
   if(h.kern_qmf==INVALID_HANDLE || h.kern_morlet==INVALID_HANDLE || h.kern_morlet2==INVALID_HANDLE || h.kern_ricker==INVALID_HANDLE || h.kern_cwt_ricker==INVALID_HANDLE || h.kern_cwt_morlet2==INVALID_HANDLE)
     { CLWaveletFree(h); return false; }
   h.ready=true;
   return true;
  }

inline void _unpack_complex_buf(const double &buf[],Complex64 &out[])
  {
   int N=ArraySize(buf)/2;
   ArrayResize(out,N);
   for(int i=0;i<N;i++) out[i]=Cx(buf[2*i],buf[2*i+1]);
  }

inline bool WaveletQMF(const double &hk[],double &out[])
  {
   int n=ArraySize(hk);
   if(n<=0) return false;
   static CLWaveletHandle h; if(!h.ready) CLWaveletReset(h);
   if(!CLWaveletInit(h)) return false;
   if(h.memA!=INVALID_HANDLE) CLBufferFree(h.memA);
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memA=CLBufferCreate(h.ctx,n*sizeof(double),CL_MEM_READ_ONLY);
   h.memOut=CLBufferCreate(h.ctx,n*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memA==INVALID_HANDLE || h.memOut==INVALID_HANDLE) return false;
   CLBufferWrite(h.memA,hk);
   CLSetKernelArgMem(h.kern_qmf,0,h.memA);
   CLSetKernelArgMem(h.kern_qmf,1,h.memOut);
   CLSetKernelArg(h.kern_qmf,2,n);
   uint offs[1]={0}; uint work[1]={(uint)n};
   if(!CLExecute(h.kern_qmf,1,offs,work)) return false;
   ArrayResize(out,n);
   CLBufferRead(h.memOut,out);
   return true;
  }

inline bool WaveletMorlet(const int M,const double w,const double s,const bool complete,Complex64 &out[])
  {
   if(M<=0) return false;
   static CLWaveletHandle h; if(!h.ready) CLWaveletReset(h);
   if(!CLWaveletInit(h)) return false;
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memOut=CLBufferCreate(h.ctx,M*sizeof(double)*2,CL_MEM_WRITE_ONLY);
   if(h.memOut==INVALID_HANDLE) return false;
   CLSetKernelArg(h.kern_morlet,0,M);
   CLSetKernelArg(h.kern_morlet,1,w);
   CLSetKernelArg(h.kern_morlet,2,s);
   CLSetKernelArg(h.kern_morlet,3,(int)(complete?1:0));
   CLSetKernelArgMem(h.kern_morlet,4,h.memOut);
   uint offs[1]={0}; uint work[1]={(uint)M};
   if(!CLExecute(h.kern_morlet,1,offs,work)) return false;
   double buf[]; ArrayResize(buf,2*M);
   CLBufferRead(h.memOut,buf);
   _unpack_complex_buf(buf,out);
   return true;
  }

inline bool WaveletMorlet2(const int M,const double s,const double w,Complex64 &out[])
  {
   if(M<=0) return false;
   static CLWaveletHandle h; if(!h.ready) CLWaveletReset(h);
   if(!CLWaveletInit(h)) return false;
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memOut=CLBufferCreate(h.ctx,M*sizeof(double)*2,CL_MEM_WRITE_ONLY);
   if(h.memOut==INVALID_HANDLE) return false;
   CLSetKernelArg(h.kern_morlet2,0,M);
   CLSetKernelArg(h.kern_morlet2,1,s);
   CLSetKernelArg(h.kern_morlet2,2,w);
   CLSetKernelArgMem(h.kern_morlet2,3,h.memOut);
   uint offs[1]={0}; uint work[1]={(uint)M};
   if(!CLExecute(h.kern_morlet2,1,offs,work)) return false;
   double buf[]; ArrayResize(buf,2*M);
   CLBufferRead(h.memOut,buf);
   _unpack_complex_buf(buf,out);
   return true;
  }

inline bool WaveletRicker(const int points,const double a,double &out[])
  {
   if(points<=0) return false;
   static CLWaveletHandle h; if(!h.ready) CLWaveletReset(h);
   if(!CLWaveletInit(h)) return false;
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memOut=CLBufferCreate(h.ctx,points*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memOut==INVALID_HANDLE) return false;
   CLSetKernelArg(h.kern_ricker,0,points);
   CLSetKernelArg(h.kern_ricker,1,a);
   CLSetKernelArgMem(h.kern_ricker,2,h.memOut);
   uint offs[1]={0}; uint work[1]={(uint)points};
   if(!CLExecute(h.kern_ricker,1,offs,work)) return false;
   ArrayResize(out,points);
   CLBufferRead(h.memOut,out);
   return true;
  }

inline bool CWT_Ricker(const double &data[],const double &widths[],double &out[][])
  {
   int N=ArraySize(data);
   int W=ArraySize(widths);
   if(N<=0 || W<=0) return false;
   static CLWaveletHandle h; if(!h.ready) CLWaveletReset(h);
   if(!CLWaveletInit(h)) return false;

   if(h.memA!=INVALID_HANDLE) CLBufferFree(h.memA);
   h.memA=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   if(h.memA==INVALID_HANDLE) return false;
   CLBufferWrite(h.memA,data);

   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memOut=CLBufferCreate(h.ctx,(long)W*(long)N*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memOut==INVALID_HANDLE) return false;

   uint offs[1]={0}; uint work[1]={(uint)N};
   for(int wi=0;wi<W;wi++)
     {
      double width=widths[wi];
      CLSetKernelArgMem(h.kern_cwt_ricker,0,h.memA);
      CLSetKernelArg(h.kern_cwt_ricker,1,N);
      CLSetKernelArg(h.kern_cwt_ricker,2,width);
      CLSetKernelArg(h.kern_cwt_ricker,3,wi);
      CLSetKernelArgMem(h.kern_cwt_ricker,4,h.memOut);
      if(!CLExecute(h.kern_cwt_ricker,1,offs,work)) return false;
     }

   double buf[]; ArrayResize(buf,(long)W*(long)N);
   CLBufferRead(h.memOut,buf);
   ArrayResize(out,W);
   for(int wi=0;wi<W;wi++)
     {
      ArrayResize(out[wi],N);
      long base=(long)wi*(long)N;
      for(int i=0;i<N;i++) out[wi][i]=buf[base+i];
     }
   return true;
  }

inline bool CWT_Morlet2(const double &data[],const double &widths[],const double w,Complex64 &out[][])
  {
   int N=ArraySize(data);
   int W=ArraySize(widths);
   if(N<=0 || W<=0) return false;
   static CLWaveletHandle h; if(!h.ready) CLWaveletReset(h);
   if(!CLWaveletInit(h)) return false;

   if(h.memA!=INVALID_HANDLE) CLBufferFree(h.memA);
   h.memA=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   if(h.memA==INVALID_HANDLE) return false;
   CLBufferWrite(h.memA,data);

   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memOut=CLBufferCreate(h.ctx,(long)W*(long)N*sizeof(double)*2,CL_MEM_WRITE_ONLY);
   if(h.memOut==INVALID_HANDLE) return false;

   uint offs[1]={0}; uint work[1]={(uint)N};
   for(int wi=0;wi<W;wi++)
     {
      double width=widths[wi];
      CLSetKernelArgMem(h.kern_cwt_morlet2,0,h.memA);
      CLSetKernelArg(h.kern_cwt_morlet2,1,N);
      CLSetKernelArg(h.kern_cwt_morlet2,2,width);
      CLSetKernelArg(h.kern_cwt_morlet2,3,w);
      CLSetKernelArg(h.kern_cwt_morlet2,4,wi);
      CLSetKernelArgMem(h.kern_cwt_morlet2,5,h.memOut);
      if(!CLExecute(h.kern_cwt_morlet2,1,offs,work)) return false;
     }

   double buf[]; ArrayResize(buf,(long)W*(long)N*2);
   CLBufferRead(h.memOut,buf);
   ArrayResize(out,W);
   for(int wi=0;wi<W;wi++)
     {
      ArrayResize(out[wi],N);
      long base=(long)wi*(long)N*2;
      for(int i=0;i<N;i++) out[wi][i]=Cx(buf[base+2*i],buf[base+2*i+1]);
     }
   return true;
  }

#endif
