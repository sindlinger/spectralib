#ifndef __SPECTRAL_WAVELETS_MQH__
#define __SPECTRAL_WAVELETS_MQH__

#include "SpectralCommon.mqh"
#include "SpectralOpenCLCommon.mqh"

struct CLWaveletHandle
  {
   int ctx;
   int prog;
   int kern_qmf;
   int kern_morlet;
   int kern_morlet2;
   int kern_ricker;
   int kern_gaus;
   int kern_mexh;
   int kern_morl;
   int kern_cgau;
   int kern_shan;
   int kern_fbsp;
   int kern_cmor;
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
   h.kern_ricker=INVALID_HANDLE; h.kern_gaus=INVALID_HANDLE; h.kern_mexh=INVALID_HANDLE; h.kern_morl=INVALID_HANDLE;
   h.kern_cgau=INVALID_HANDLE; h.kern_shan=INVALID_HANDLE; h.kern_fbsp=INVALID_HANDLE; h.kern_cmor=INVALID_HANDLE;
   h.kern_cwt_ricker=INVALID_HANDLE; h.kern_cwt_morlet2=INVALID_HANDLE;
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
   if(h.kern_gaus!=INVALID_HANDLE) { CLKernelFree(h.kern_gaus); h.kern_gaus=INVALID_HANDLE; }
   if(h.kern_mexh!=INVALID_HANDLE) { CLKernelFree(h.kern_mexh); h.kern_mexh=INVALID_HANDLE; }
   if(h.kern_morl!=INVALID_HANDLE) { CLKernelFree(h.kern_morl); h.kern_morl=INVALID_HANDLE; }
   if(h.kern_cgau!=INVALID_HANDLE) { CLKernelFree(h.kern_cgau); h.kern_cgau=INVALID_HANDLE; }
   if(h.kern_shan!=INVALID_HANDLE) { CLKernelFree(h.kern_shan); h.kern_shan=INVALID_HANDLE; }
   if(h.kern_fbsp!=INVALID_HANDLE) { CLKernelFree(h.kern_fbsp); h.kern_fbsp=INVALID_HANDLE; }
   if(h.kern_cmor!=INVALID_HANDLE) { CLKernelFree(h.kern_cmor); h.kern_cmor=INVALID_HANDLE; }
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
   h.ctx=CLCreateContextGPUFloat64("SpectralWavelets");
   if(h.ctx==INVALID_HANDLE) return false;
   string code=
   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
   "#define M_PI 3.1415926535897932384626433832795\n"
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
   "__kernel void gaus_kernel(__global const double* x, int N, int number, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=N) return; double xi=x[i]; double x2=xi*xi;\n"
   "  double base=sqrt(M_PI/2.0);\n"
   "  double v=0.0;\n"
   "  if(number==1){ v=-2.0*xi*exp(-x2)/sqrt(base); }\n"
   "  else if(number==2){ v=-2.0*(2.0*x2-1.0)*exp(-x2)/sqrt(3.0*base); }\n"
   "  else if(number==3){ v=-4.0*(-2.0*pow(xi,3.0)+3.0*xi)*exp(-x2)/sqrt(15.0*base); }\n"
   "  else if(number==4){ v=4.0*(-12.0*x2+4.0*pow(xi,4.0)+3.0)*exp(-x2)/sqrt(105.0*base); }\n"
   "  else if(number==5){ v=8.0*(-4.0*pow(xi,5.0)+20.0*pow(xi,3.0)-15.0*xi)*exp(-x2)/sqrt(105.0*9.0*base); }\n"
   "  else if(number==6){ v=-8.0*(8.0*pow(xi,6.0)-60.0*pow(xi,4.0)+90.0*x2-15.0)*exp(-x2)/sqrt(105.0*9.0*11.0*base); }\n"
   "  else if(number==7){ v=-16.0*(-8.0*pow(xi,7.0)+84.0*pow(xi,5.0)-210.0*pow(xi,3.0)+105.0*xi)*exp(-x2)/sqrt(105.0*9.0*11.0*13.0*base); }\n"
   "  else if(number==8){ v=16.0*(16.0*pow(xi,8.0)-224.0*pow(xi,6.0)+840.0*pow(xi,4.0)-840.0*x2+105.0)*exp(-x2)/sqrt(105.0*9.0*11.0*13.0*15.0*base); }\n"
   "  out[i]=v; }\n"
   "__kernel void mexh_kernel(__global const double* x, int N, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=N) return; double xi=x[i]; double x2=xi*xi;\n"
   "  double v=(1.0-x2)*exp(-x2*0.5)*2.0/(sqrt(3.0)*sqrt(sqrt(M_PI))); out[i]=v; }\n"
   "__kernel void morl_kernel(__global const double* x, int N, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=N) return; double xi=x[i]; double v=cos(5.0*xi)*exp(-xi*xi*0.5); out[i]=v; }\n"
   "__kernel void cgau_kernel(__global const double* x, int N, int number, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=N) return; double xi=x[i]; double x2=xi*xi; double ce=cos(xi); double se=sin(xi); double ex=exp(-x2);\n"
   "  double nr=0.0; double ni=0.0; double denom=0.0;\n"
   "  if(number==1){ denom=sqrt(2.0*sqrt(M_PI/2.0)); nr=(-2.0*xi*ce-se)*ex/denom; ni=(2.0*xi*se-ce)*ex/denom; }\n"
   "  else if(number==2){ denom=sqrt(10.0*sqrt(M_PI/2.0)); nr=(4.0*x2*ce+4.0*xi*se-3.0*ce)*ex/denom; ni=(-4.0*x2*se+4.0*xi*ce+3.0*se)*ex/denom; }\n"
   "  else if(number==3){ denom=sqrt(76.0*sqrt(M_PI/2.0)); nr=(-8.0*pow(xi,3.0)*ce-12.0*x2*se+18.0*xi*ce+7.0*se)*ex/denom; ni=(8.0*pow(xi,3.0)*se-12.0*x2*ce-18.0*xi*se+7.0*ce)*ex/denom; }\n"
   "  else if(number==4){ denom=sqrt(764.0*sqrt(M_PI/2.0)); nr=(16.0*pow(xi,4.0)*ce+32.0*pow(xi,3.0)*se-72.0*x2*ce-56.0*xi*se+25.0*ce)*ex/denom; ni=(-16.0*pow(xi,4.0)*se+32.0*pow(xi,3.0)*ce+72.0*x2*se-56.0*xi*ce-25.0*se)*ex/denom; }\n"
   "  else if(number==5){ denom=sqrt(9496.0*sqrt(M_PI/2.0)); nr=(-32.0*pow(xi,5.0)*ce-80.0*pow(xi,4.0)*se+240.0*pow(xi,3.0)*ce+280.0*x2*se-250.0*xi*ce-81.0*se)*ex/denom; ni=(32.0*pow(xi,5.0)*se-80.0*pow(xi,4.0)*ce-240.0*pow(xi,3.0)*se+280.0*x2*ce+250.0*xi*se-81.0*ce)*ex/denom; }\n"
   "  else if(number==6){ denom=sqrt(140152.0*sqrt(M_PI/2.0)); nr=(64.0*pow(xi,6.0)*ce+192.0*pow(xi,5.0)*se-720.0*pow(xi,4.0)*ce-1120.0*pow(xi,3.0)*se+1500.0*x2*ce+972.0*xi*se-331.0*ce)*ex/denom; ni=(-64.0*pow(xi,6.0)*se+192.0*pow(xi,5.0)*ce+720.0*pow(xi,4.0)*se-1120.0*pow(xi,3.0)*ce-1500.0*x2*se+972.0*xi*ce+331.0*se)*ex/denom; }\n"
   "  else if(number==7){ denom=sqrt(2390480.0*sqrt(M_PI/2.0)); nr=(-128.0*pow(xi,7.0)*ce-448.0*pow(xi,6.0)*se+2016.0*pow(xi,5.0)*ce+3920.0*pow(xi,4.0)*se-7000.0*pow(xi,3.0)*ce-6804.0*x2*se+4634.0*xi*ce+1303.0*se)*ex/denom; ni=(128.0*pow(xi,7.0)*se-448.0*pow(xi,6.0)*ce-2016.0*pow(xi,5.0)*se+3920.0*pow(xi,4.0)*ce+7000.0*pow(xi,3.0)*se-6804.0*x2*ce-4634.0*xi*se+1303.0*ce)*ex/denom; }\n"
   "  else if(number==8){ denom=sqrt(46206736.0*sqrt(M_PI/2.0)); nr=(256.0*pow(xi,8.0)*ce+1024.0*pow(xi,7.0)*se-5376.0*pow(xi,6.0)*ce-12544.0*pow(xi,5.0)*se+28000.0*pow(xi,4.0)*ce+36288.0*pow(xi,3.0)*se-37072.0*x2*ce-20848.0*xi*se+5937.0*ce)*ex/denom; ni=(-256.0*pow(xi,8.0)*se+1024.0*pow(xi,7.0)*ce+5376.0*pow(xi,6.0)*se-12544.0*pow(xi,5.0)*ce-28000.0*pow(xi,4.0)*se+36288.0*pow(xi,3.0)*ce+37072.0*x2*se-20848.0*xi*ce-5937.0*se)*ex/denom; }\n"
   "  out[i]=(double2)(nr,ni); }\n"
   "__kernel void shan_kernel(__global const double* x, int N, double FB, double FC, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=N) return; double xi=x[i]; double c=cos(2.0*M_PI*FC*xi); double s=sin(2.0*M_PI*FC*xi);\n"
   "  double scale=sqrt(FB); double vr=c*scale; double vi=s*scale; if(xi!=0.0){ double t=sin(xi*FB*M_PI)/(xi*FB*M_PI); vr*=t; vi*=t; }\n"
   "  out[i]=(double2)(vr,vi); }\n"
   "__kernel void fbsp_kernel(__global const double* x, int N, int M, double FB, double FC, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=N) return; double xi=x[i]; double c=cos(2.0*M_PI*FC*xi); double s=sin(2.0*M_PI*FC*xi);\n"
   "  double scale=sqrt(FB); double vr=c*scale; double vi=s*scale; if(xi!=0.0){ double t=sin(M_PI*xi*FB/(double)M)/(M_PI*xi*FB/(double)M); double p=pow(t,(double)M); vr*=p; vi*=p; }\n"
   "  out[i]=(double2)(vr,vi); }\n"
   "__kernel void cmor_kernel(__global const double* x, int N, double FB, double FC, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=N) return; double xi=x[i]; double c=cos(2.0*M_PI*FC*xi); double s=sin(2.0*M_PI*FC*xi);\n"
   "  double ex=exp(-(xi*xi)/FB); double scale=1.0/sqrt(M_PI*FB); out[i]=(double2)(c*ex*scale, s*ex*scale); }\n"
   "__kernel void cwt_ricker(__global const double* data, int N, double width, int row, __global double* out){\n"
   "  int n=get_global_id(0); if(n>=N) return; int wi=(int)floor(width); if(wi<1) wi=1;\n"
   "  int L=10*wi; if(L>N) L=N; double hlf=((double)(L-1))*0.5; double wsq=width*width;\n"
   "  double A=2.0/(sqrt(3.0*width)*pow(M_PI,0.25)); double sum=0.0; int start=n - (int)floor(L*0.5);\n"
   "  for(int k=0;k<L;k++){ int idx=start + k; if(idx>=0 && idx<N){ double x=hlf - (double)k;\n"
   "    double xsq=x*x; double mod=1.0 - xsq/wsq; double ga=exp(-xsq/(2.0*wsq)); double wv=A*mod*ga;\n"
   "    sum += data[idx]*wv; }} out[row*N + n]=sum; }\n"
   "__kernel void cwt_morlet2(__global const double* data, int N, double width, double w, int row, __global double2* out){\n"
   "  int n=get_global_id(0); if(n>=N) return; int wi=(int)floor(width); if(wi<1) wi=1;\n"
   "  int L=10*wi; if(L>N) L=N; double hlf=((double)(L-1))*0.5; double sumr=0.0; double sumi=0.0;\n"
   "  double scale=pow(M_PI,-0.25)*sqrt(1.0/width); int start=n - (int)floor(L*0.5);\n"
   "  for(int k=0;k<L;k++){ int idx=start + k; if(idx>=0 && idx<N){ double x=(hlf - (double)k)/width;\n"
   "    double ga=exp(-0.5*x*x); double wr=cos(w*x); double wi2=-sin(w*x);\n"
   "    double wvr=wr*ga*scale; double wvi=wi2*ga*scale; sumr += data[idx]*wvr; sumi += data[idx]*wvi; }}\n"
   "  out[row*N + n]=(double2)(sumr,sumi); }\n";

   h.prog=CLProgramCreate(h.ctx,code);
   if(h.prog==INVALID_HANDLE) { CLWaveletFree(h); return false; }
   h.kern_qmf=CLKernelCreate(h.prog,"qmf_kernel");
   h.kern_morlet=CLKernelCreate(h.prog,"morlet_kernel");
   h.kern_morlet2=CLKernelCreate(h.prog,"morlet2_kernel");
   h.kern_ricker=CLKernelCreate(h.prog,"ricker_kernel");
   h.kern_gaus=CLKernelCreate(h.prog,"gaus_kernel");
   h.kern_mexh=CLKernelCreate(h.prog,"mexh_kernel");
   h.kern_morl=CLKernelCreate(h.prog,"morl_kernel");
   h.kern_cgau=CLKernelCreate(h.prog,"cgau_kernel");
   h.kern_shan=CLKernelCreate(h.prog,"shan_kernel");
   h.kern_fbsp=CLKernelCreate(h.prog,"fbsp_kernel");
   h.kern_cmor=CLKernelCreate(h.prog,"cmor_kernel");
   h.kern_cwt_ricker=CLKernelCreate(h.prog,"cwt_ricker");
   h.kern_cwt_morlet2=CLKernelCreate(h.prog,"cwt_morlet2");
   if(h.kern_qmf==INVALID_HANDLE || h.kern_morlet==INVALID_HANDLE || h.kern_morlet2==INVALID_HANDLE || h.kern_ricker==INVALID_HANDLE ||
      h.kern_gaus==INVALID_HANDLE || h.kern_mexh==INVALID_HANDLE || h.kern_morl==INVALID_HANDLE || h.kern_cgau==INVALID_HANDLE ||
      h.kern_shan==INVALID_HANDLE || h.kern_fbsp==INVALID_HANDLE || h.kern_cmor==INVALID_HANDLE ||
      h.kern_cwt_ricker==INVALID_HANDLE || h.kern_cwt_morlet2==INVALID_HANDLE)
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

inline bool WaveletGaus(const double &x[],const int number,double &out[])
  {
   int N=ArraySize(x);
   if(N<=0) return false;
   static CLWaveletHandle h; if(!h.ready) CLWaveletReset(h);
   if(!CLWaveletInit(h)) return false;
   if(h.memA!=INVALID_HANDLE) CLBufferFree(h.memA);
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memA=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   h.memOut=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memA==INVALID_HANDLE || h.memOut==INVALID_HANDLE) return false;
   CLBufferWrite(h.memA,x);
   CLSetKernelArgMem(h.kern_gaus,0,h.memA);
   CLSetKernelArg(h.kern_gaus,1,N);
   CLSetKernelArg(h.kern_gaus,2,number);
   CLSetKernelArgMem(h.kern_gaus,3,h.memOut);
   uint offs[1]={0}; uint work[1]={(uint)N};
   if(!CLExecute(h.kern_gaus,1,offs,work)) return false;
   ArrayResize(out,N);
   CLBufferRead(h.memOut,out);
   return true;
  }

inline bool WaveletMexh(const double &x[],double &out[])
  {
   int N=ArraySize(x);
   if(N<=0) return false;
   static CLWaveletHandle h; if(!h.ready) CLWaveletReset(h);
   if(!CLWaveletInit(h)) return false;
   if(h.memA!=INVALID_HANDLE) CLBufferFree(h.memA);
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memA=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   h.memOut=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memA==INVALID_HANDLE || h.memOut==INVALID_HANDLE) return false;
   CLBufferWrite(h.memA,x);
   CLSetKernelArgMem(h.kern_mexh,0,h.memA);
   CLSetKernelArg(h.kern_mexh,1,N);
   CLSetKernelArgMem(h.kern_mexh,2,h.memOut);
   uint offs[1]={0}; uint work[1]={(uint)N};
   if(!CLExecute(h.kern_mexh,1,offs,work)) return false;
   ArrayResize(out,N);
   CLBufferRead(h.memOut,out);
   return true;
  }

inline bool WaveletMorl(const double &x[],double &out[])
  {
   int N=ArraySize(x);
   if(N<=0) return false;
   static CLWaveletHandle h; if(!h.ready) CLWaveletReset(h);
   if(!CLWaveletInit(h)) return false;
   if(h.memA!=INVALID_HANDLE) CLBufferFree(h.memA);
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memA=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   h.memOut=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memA==INVALID_HANDLE || h.memOut==INVALID_HANDLE) return false;
   CLBufferWrite(h.memA,x);
   CLSetKernelArgMem(h.kern_morl,0,h.memA);
   CLSetKernelArg(h.kern_morl,1,N);
   CLSetKernelArgMem(h.kern_morl,2,h.memOut);
   uint offs[1]={0}; uint work[1]={(uint)N};
   if(!CLExecute(h.kern_morl,1,offs,work)) return false;
   ArrayResize(out,N);
   CLBufferRead(h.memOut,out);
   return true;
  }

inline bool WaveletCGau(const double &x[],const int number,Complex64 &out[])
  {
   int N=ArraySize(x);
   if(N<=0) return false;
   static CLWaveletHandle h; if(!h.ready) CLWaveletReset(h);
   if(!CLWaveletInit(h)) return false;
   if(h.memA!=INVALID_HANDLE) CLBufferFree(h.memA);
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memA=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   h.memOut=CLBufferCreate(h.ctx,N*sizeof(double)*2,CL_MEM_WRITE_ONLY);
   if(h.memA==INVALID_HANDLE || h.memOut==INVALID_HANDLE) return false;
   CLBufferWrite(h.memA,x);
   CLSetKernelArgMem(h.kern_cgau,0,h.memA);
   CLSetKernelArg(h.kern_cgau,1,N);
   CLSetKernelArg(h.kern_cgau,2,number);
   CLSetKernelArgMem(h.kern_cgau,3,h.memOut);
   uint offs[1]={0}; uint work[1]={(uint)N};
   if(!CLExecute(h.kern_cgau,1,offs,work)) return false;
   double buf[]; ArrayResize(buf,2*N);
   CLBufferRead(h.memOut,buf);
   _unpack_complex_buf(buf,out);
   return true;
  }

inline bool WaveletShan(const double &x[],const double FB,const double FC,Complex64 &out[])
  {
   int N=ArraySize(x);
   if(N<=0) return false;
   static CLWaveletHandle h; if(!h.ready) CLWaveletReset(h);
   if(!CLWaveletInit(h)) return false;
   if(h.memA!=INVALID_HANDLE) CLBufferFree(h.memA);
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memA=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   h.memOut=CLBufferCreate(h.ctx,N*sizeof(double)*2,CL_MEM_WRITE_ONLY);
   if(h.memA==INVALID_HANDLE || h.memOut==INVALID_HANDLE) return false;
   CLBufferWrite(h.memA,x);
   CLSetKernelArgMem(h.kern_shan,0,h.memA);
   CLSetKernelArg(h.kern_shan,1,N);
   CLSetKernelArg(h.kern_shan,2,FB);
   CLSetKernelArg(h.kern_shan,3,FC);
   CLSetKernelArgMem(h.kern_shan,4,h.memOut);
   uint offs[1]={0}; uint work[1]={(uint)N};
   if(!CLExecute(h.kern_shan,1,offs,work)) return false;
   double buf[]; ArrayResize(buf,2*N);
   CLBufferRead(h.memOut,buf);
   _unpack_complex_buf(buf,out);
   return true;
  }

inline bool WaveletFBSP(const double &x[],const int M,const double FB,const double FC,Complex64 &out[])
  {
   int N=ArraySize(x);
   if(N<=0) return false;
   static CLWaveletHandle h; if(!h.ready) CLWaveletReset(h);
   if(!CLWaveletInit(h)) return false;
   if(h.memA!=INVALID_HANDLE) CLBufferFree(h.memA);
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memA=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   h.memOut=CLBufferCreate(h.ctx,N*sizeof(double)*2,CL_MEM_WRITE_ONLY);
   if(h.memA==INVALID_HANDLE || h.memOut==INVALID_HANDLE) return false;
   CLBufferWrite(h.memA,x);
   CLSetKernelArgMem(h.kern_fbsp,0,h.memA);
   CLSetKernelArg(h.kern_fbsp,1,N);
   CLSetKernelArg(h.kern_fbsp,2,M);
   CLSetKernelArg(h.kern_fbsp,3,FB);
   CLSetKernelArg(h.kern_fbsp,4,FC);
   CLSetKernelArgMem(h.kern_fbsp,5,h.memOut);
   uint offs[1]={0}; uint work[1]={(uint)N};
   if(!CLExecute(h.kern_fbsp,1,offs,work)) return false;
   double buf[]; ArrayResize(buf,2*N);
   CLBufferRead(h.memOut,buf);
   _unpack_complex_buf(buf,out);
   return true;
  }

inline bool WaveletCMor(const double &x[],const double FB,const double FC,Complex64 &out[])
  {
   int N=ArraySize(x);
   if(N<=0) return false;
   static CLWaveletHandle h; if(!h.ready) CLWaveletReset(h);
   if(!CLWaveletInit(h)) return false;
   if(h.memA!=INVALID_HANDLE) CLBufferFree(h.memA);
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memA=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   h.memOut=CLBufferCreate(h.ctx,N*sizeof(double)*2,CL_MEM_WRITE_ONLY);
   if(h.memA==INVALID_HANDLE || h.memOut==INVALID_HANDLE) return false;
   CLBufferWrite(h.memA,x);
   CLSetKernelArgMem(h.kern_cmor,0,h.memA);
   CLSetKernelArg(h.kern_cmor,1,N);
   CLSetKernelArg(h.kern_cmor,2,FB);
   CLSetKernelArg(h.kern_cmor,3,FC);
   CLSetKernelArgMem(h.kern_cmor,4,h.memOut);
   uint offs[1]={0}; uint work[1]={(uint)N};
   if(!CLExecute(h.kern_cmor,1,offs,work)) return false;
   double buf[]; ArrayResize(buf,2*N);
   CLBufferRead(h.memOut,buf);
   _unpack_complex_buf(buf,out);
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
   int total=W*N;
   h.memOut=CLBufferCreate(h.ctx,total*sizeof(double),CL_MEM_WRITE_ONLY);
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

   double buf[]; ArrayResize(buf,total);
   CLBufferRead(h.memOut,buf);
   ArrayResize(out,W,N);
   for(int wi=0;wi<W;wi++)
     {
      int base=wi*N;
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
   int total2=W*N;
   h.memOut=CLBufferCreate(h.ctx,total2*sizeof(double)*2,CL_MEM_WRITE_ONLY);
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

   double buf[]; ArrayResize(buf,total2*2);
   CLBufferRead(h.memOut,buf);
   ArrayResize(out,W,N);
   for(int wi=0;wi<W;wi++)
     {
      int base=wi*N*2;
      for(int i=0;i<N;i++) out[wi][i]=Cx(buf[base+2*i],buf[base+2*i+1]);
     }
   return true;
  }

enum CWTWaveletType
  {
   CWT_WAVELET_RICKER=0,
   CWT_WAVELET_MORLET2=1
  };

inline bool CWT_Ricker_Flat(const double &data[],const double &widths[],double &outFlat[],int &outW,int &outN)
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
   int total=W*N;
   h.memOut=CLBufferCreate(h.ctx,total*sizeof(double),CL_MEM_WRITE_ONLY);
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

   ArrayResize(outFlat,total);
   CLBufferRead(h.memOut,outFlat);
   outW=W; outN=N;
   return true;
  }

inline bool CWT_Morlet2_Flat(const double &data[],const double &widths[],const double w,Complex64 &outFlat[],int &outW,int &outN)
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
   int total=W*N;
   h.memOut=CLBufferCreate(h.ctx,total*sizeof(double)*2,CL_MEM_WRITE_ONLY);
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

   double buf[]; ArrayResize(buf,total*2);
   CLBufferRead(h.memOut,buf);
   ArrayResize(outFlat,total);
   for(int i=0;i<total;i++)
     outFlat[i]=Cx(buf[2*i],buf[2*i+1]);
   outW=W; outN=N;
   return true;
  }

#endif
