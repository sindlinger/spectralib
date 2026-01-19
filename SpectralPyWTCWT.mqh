#ifndef __SPECTRAL_PYWT_CWT_MQH__
#define __SPECTRAL_PYWT_CWT_MQH__

#include "SpectralCommon.mqh"
#include "SpectralPyWTCommon.mqh"
#include "SpectralWavelets.mqh"
#include "SpectralOpenCLFFT.mqh"
#include "SpectralOpenCLCommon.mqh"

// Continuous wavelet struct (PyWavelets compatible defaults)
struct PyWTContinuousWavelet
  {
   string name;
   string family_name;
   string short_name;
   int family_number;
   bool complex_cwt;
   double lower_bound;
   double upper_bound;
   double center_frequency;
   double bandwidth_frequency;
   int fbsp_order;
  };

inline void PyWT_ResetCWT(PyWTContinuousWavelet &w)
  {
   w.name=""; w.family_name=""; w.short_name=""; w.family_number=0;
   w.complex_cwt=false;
   w.lower_bound=-8.0; w.upper_bound=8.0;
   w.center_frequency=0.0; w.bandwidth_frequency=0.0; w.fbsp_order=0;
  }

inline bool PyWT_ParseCWTName(const string name, PyWTContinuousWavelet &w)
  {
   PyWT_ResetCWT(w);
   string nm = name;
   StringToLower(nm);
   w.name=nm;

   // defaults from wavelets.c
   if(StringFind(nm,"gaus") == 0)
     {
      w.short_name="gaus";
      w.family_name="Gaussian";
      w.complex_cwt=false;
      w.lower_bound=-5.0; w.upper_bound=5.0;
      int ord=(int)StringToInteger(StringSubstr(nm,4));
      if(ord<=0 || ord>8) return false;
      w.family_number=ord;
      return true;
     }
   if(nm=="mexh")
     {
      w.short_name="mexh"; w.family_name="Mexican hat wavelet";
      w.complex_cwt=false; w.lower_bound=-8.0; w.upper_bound=8.0; w.family_number=1;
      return true;
     }
   if(nm=="morl")
     {
      w.short_name="morl"; w.family_name="Morlet wavelet";
      w.complex_cwt=false; w.lower_bound=-8.0; w.upper_bound=8.0; w.family_number=1;
      return true;
     }
   if(StringFind(nm,"cgau") == 0)
     {
      w.short_name="cgau"; w.family_name="Complex Gaussian wavelets";
      w.complex_cwt=true; w.lower_bound=-5.0; w.upper_bound=5.0;
      int ord=(int)StringToInteger(StringSubstr(nm,4));
      if(ord<=0 || ord>8) return false;
      w.family_number=ord;
      return true;
     }
   if(StringFind(nm,"shan") == 0)
     {
      w.short_name="shan"; w.family_name="Shannon wavelets";
      w.complex_cwt=true; w.lower_bound=-20.0; w.upper_bound=20.0;
      w.center_frequency=1.0; w.bandwidth_frequency=0.5; w.fbsp_order=0;
      string params = StringSubstr(nm,4);
      if(params!="")
        {
         int dash = StringFind(params,"-");
         if(dash<0) return false;
         double B = StringToDouble(StringSubstr(params,0,dash));
         double C = StringToDouble(StringSubstr(params,dash+1));
         w.bandwidth_frequency=B;
         w.center_frequency=C;
        }
      return true;
     }
   if(StringFind(nm,"fbsp") == 0)
     {
      w.short_name="fbsp"; w.family_name="Frequency B-Spline wavelets";
      w.complex_cwt=true; w.lower_bound=-20.0; w.upper_bound=20.0;
      w.center_frequency=0.5; w.bandwidth_frequency=1.0; w.fbsp_order=2;
      string params = StringSubstr(nm,4);
      if(params!="")
        {
         // M-B-C
         int d1 = StringFind(params,"-");
         if(d1<0) return false;
         int d2 = StringFind(params,"-",d1+1);
         if(d2<0) return false;
         double M = StringToDouble(StringSubstr(params,0,d1));
         double B = StringToDouble(StringSubstr(params,d1+1,d2-d1-1));
         double C = StringToDouble(StringSubstr(params,d2+1));
         if(M<1.0 || MathMod(M,1.0)!=0.0) return false;
         w.fbsp_order=(int)M;
         w.bandwidth_frequency=B;
         w.center_frequency=C;
        }
      return true;
     }
   if(StringFind(nm,"cmor") == 0)
     {
      w.short_name="cmor"; w.family_name="Complex Morlet wavelets";
      w.complex_cwt=true; w.lower_bound=-8.0; w.upper_bound=8.0;
      w.center_frequency=0.5; w.bandwidth_frequency=1.0; w.fbsp_order=0;
      string params = StringSubstr(nm,4);
      if(params!="")
        {
         int dash = StringFind(params,"-");
         if(dash<0) return false;
         double B = StringToDouble(StringSubstr(params,0,dash));
         double C = StringToDouble(StringSubstr(params,dash+1));
         w.bandwidth_frequency=B;
         w.center_frequency=C;
        }
      return true;
     }
   return false;
  }

inline void PyWT_Linspace(const double a,const double b,const int n,double &out[])
  {
   ArrayResize(out,n);
   if(n<=1){ if(n==1) out[0]=a; return; }
   double step=(b-a)/(double)(n-1);
   for(int i=0;i<n;i++) out[i]=a + step*(double)i;
  }

// CWT psi (single) using OpenCL wavelet kernels
inline bool PyWT_CwtPsiSingle(const double &x[],const PyWTContinuousWavelet &w, Complex64 &psi[])
  {
   int N=ArraySize(x);
   if(N<=0) return false;
   if(w.short_name=="gaus")
     {
      double tmp[]; if(!WaveletGaus(x,w.family_number,tmp)) return false;
      ArrayResize(psi,N);
      for(int i=0;i<N;i++) psi[i]=Cx(tmp[i],0.0);
      return true;
     }
   if(w.short_name=="mexh")
     {
      double tmp[]; if(!WaveletMexh(x,tmp)) return false;
      ArrayResize(psi,N);
      for(int i=0;i<N;i++) psi[i]=Cx(tmp[i],0.0);
      return true;
     }
   if(w.short_name=="morl")
     {
      double tmp[]; if(!WaveletMorl(x,tmp)) return false;
      ArrayResize(psi,N);
      for(int i=0;i<N;i++) psi[i]=Cx(tmp[i],0.0);
      return true;
     }
   if(w.short_name=="cgau")
     {
      Complex64 tmpc[]; if(!WaveletCGau(x,w.family_number,tmpc)) return false;
      ArrayResize(psi,N);
      for(int i=0;i<N;i++) psi[i]=tmpc[i];
      return true;
     }
   if(w.short_name=="shan")
     {
      Complex64 tmpc[]; if(!WaveletShan(x,w.bandwidth_frequency,w.center_frequency,tmpc)) return false;
      ArrayResize(psi,N);
      for(int i=0;i<N;i++) psi[i]=tmpc[i];
      return true;
     }
   if(w.short_name=="fbsp")
     {
      Complex64 tmpc[]; if(!WaveletFBSP(x,w.fbsp_order,w.bandwidth_frequency,w.center_frequency,tmpc)) return false;
      ArrayResize(psi,N);
      for(int i=0;i<N;i++) psi[i]=tmpc[i];
      return true;
     }
   if(w.short_name=="cmor")
     {
      Complex64 tmpc[]; if(!WaveletCMor(x,w.bandwidth_frequency,w.center_frequency,tmpc)) return false;
      ArrayResize(psi,N);
      for(int i=0;i<N;i++) psi[i]=tmpc[i];
      return true;
     }
   return false;
  }

inline bool PyWT_IntegrateWavelet(const PyWTContinuousWavelet &w,const int precision,Complex64 &int_psi[],double &x[])
  {
   int n = 1 << precision;
   if(n<=0) return false;
   PyWT_Linspace(w.lower_bound,w.upper_bound,n,x);
   Complex64 psi[];
   if(!PyWT_CwtPsiSingle(x,w,psi)) return false;
   ArrayResize(int_psi,n);
   double dx = (x[n-1]-x[0])/(double)(n-1);
   Complex64 acc=Cx(0.0,0.0);
   for(int i=0;i<n;i++)
     {
      acc.re += psi[i].re * dx;
      acc.im += psi[i].im * dx;
      int_psi[i]=acc;
     }
   return true;
  }

inline int PyWT_NextFastLen(const int n)
  {
   int p=1;
   while(p < n) p <<= 1;
   return p;
  }

// Estimate central frequency numerically (FFT magnitude peak)
inline bool PyWT_CentralFrequency(const PyWTContinuousWavelet &w,const int precision,double &out_cf)
  {
   if(w.center_frequency > 0.0 && (w.short_name=="cmor" || w.short_name=="shan" || w.short_name=="fbsp"))
     {
      out_cf = w.center_frequency;
      return true;
     }
   int n = 1 << precision;
   if(n<=0) return false;
   double x[]; PyWT_Linspace(w.lower_bound,w.upper_bound,n,x);
   Complex64 psi[];
   if(!PyWT_CwtPsiSingle(x,w,psi)) return false;
   if(n<=1) return false;
   double dt = x[1] - x[0];
   Complex64 in[]; ArrayResize(in,n);
   for(int i=0;i<n;i++) in[i]=psi[i];
   CLFFTPlan plan; CLFFTReset(plan);
   if(!CLFFTInit(plan,n)) { CLFFTFree(plan); return false; }
   if(!CLFFTUploadComplexBatch(plan,in,1)) { CLFFTFree(plan); return false; }
   if(!CLFFTExecuteBatchFromMemA_NoRead(plan,1,false)) { CLFFTFree(plan); return false; }
   int idx_max=0;
   double maxv=0.0;
   if(!CLFFTMaxMagIndexFromMem(plan,plan.memFinal,n,idx_max,maxv)) { CLFFTFree(plan); return false; }
   CLFFTFree(plan);
   // freq bins (fft freq)
   double freq;
   if(idx_max <= n/2)
      freq = (double)idx_max / (dt * (double)n);
   else
      freq = - (double)(n - idx_max) / (dt * (double)n);
   out_cf = MathAbs(freq);
   return true;
  }

inline bool PyWT_Scale2Frequency(const PyWTContinuousWavelet &w,const double &scales[],const int precision,double &freqs[])
  {
   int N=ArraySize(scales);
   if(N<=0) return false;
   double cf;
   if(!PyWT_CentralFrequency(w,precision,cf)) return false;
   ArrayResize(freqs,N);
   for(int i=0;i<N;i++)
     freqs[i] = cf / scales[i];
   return true;
  }

// Simple full convolution kernels (real/complex)
struct PyWTCWTConvHandle
  {
   int ctx;
   int prog;
   int kern_conv_real;
   int kern_conv_cplx;
   int kern_conv_cplx2;
   int kern_diff;
   int kern_trim;
   int memIn;
   int memF;
   int memFi;
   int memF2;
   int memOut;
   int memCoef;
   int memTrim;
   int lenIn;
   int lenF;
   int lenOut;
   int lenCoef;
   int lenTrim;
   bool ready;
  };

// FFT-domain multiply helpers (OpenCL)
struct PyWTCWTFftMulHandle
  {
   int ctx;
   int prog;
   int kern_copy;
   int kern_mul;
   int memData;
   int len;
   bool ready;
  };

// Wavelet generation + integration (OpenCL, same context as FFT)
struct PyWTCWTPsiHandle
  {
   int ctx;
   int prog;
   int kern_gaus;
   int kern_mexh;
   int kern_morl;
   int kern_cgau;
   int kern_shan;
   int kern_fbsp;
   int kern_cmor;
   int kern_cumsum;
   int kern_build;
   int memPsi;
   int memInt;
   int len;
   bool ready;
  };

inline void PyWTCWTFftMulReset(PyWTCWTFftMulHandle &h)
  {
   h.ctx=INVALID_HANDLE; h.prog=INVALID_HANDLE; h.kern_copy=INVALID_HANDLE; h.kern_mul=INVALID_HANDLE;
   h.memData=INVALID_HANDLE; h.len=0; h.ready=false;
  }

inline void PyWTCWTFftMulFree(PyWTCWTFftMulHandle &h)
  {
   if(h.memData!=INVALID_HANDLE){ CLBufferFree(h.memData); h.memData=INVALID_HANDLE; }
   if(h.kern_copy!=INVALID_HANDLE){ CLKernelFree(h.kern_copy); h.kern_copy=INVALID_HANDLE; }
   if(h.kern_mul!=INVALID_HANDLE){ CLKernelFree(h.kern_mul); h.kern_mul=INVALID_HANDLE; }
   if(h.prog!=INVALID_HANDLE){ CLProgramFree(h.prog); h.prog=INVALID_HANDLE; }
   h.ctx=INVALID_HANDLE; h.len=0; h.ready=false;
  }

inline bool PyWTCWTFftMulInit(PyWTCWTFftMulHandle &h,const int ctx)
  {
   if(h.ready && h.ctx==ctx) return true;
   PyWTCWTFftMulFree(h);
   PyWTCWTFftMulReset(h);
   h.ctx=ctx;
   string code=
   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
   "__kernel void copy_cplx(__global const double2* src, __global double2* dst, int n){\n"
   "  int i=get_global_id(0); if(i>=n) return; dst[i]=src[i]; }\n"
   "__kernel void cmul_cplx(__global const double2* a, __global const double2* b, __global double2* out, int n){\n"
   "  int i=get_global_id(0); if(i>=n) return; double2 x=a[i]; double2 y=b[i];\n"
   "  out[i]=(double2)(x.x*y.x - x.y*y.y, x.x*y.y + x.y*y.x); }\n";
   h.prog=CLProgramCreate(h.ctx,code);
   if(h.prog==INVALID_HANDLE){ PyWTCWTFftMulFree(h); return false; }
   h.kern_copy=CLKernelCreate(h.prog,"copy_cplx");
   h.kern_mul=CLKernelCreate(h.prog,"cmul_cplx");
   if(h.kern_copy==INVALID_HANDLE || h.kern_mul==INVALID_HANDLE){ PyWTCWTFftMulFree(h); return false; }
   h.ready=true;
   return true;
  }

inline bool PyWTCWTFftEnsureData(PyWTCWTFftMulHandle &h,const int n)
  {
   if(!h.ready) return false;
   if(h.memData!=INVALID_HANDLE && h.len==n) return true;
   if(h.memData!=INVALID_HANDLE) { CLBufferFree(h.memData); h.memData=INVALID_HANDLE; }
   h.memData=CLBufferCreate(h.ctx,n*sizeof(double)*2,CL_MEM_READ_WRITE);
   if(h.memData==INVALID_HANDLE) return false;
   h.len=n;
   return true;
  }

inline bool PyWTCWTFftCopy(PyWTCWTFftMulHandle &h,const int srcMem,const int dstMem,const int n)
  {
   if(!h.ready) return false;
   CLSetKernelArgMem(h.kern_copy,0,srcMem);
   CLSetKernelArgMem(h.kern_copy,1,dstMem);
   CLSetKernelArg(h.kern_copy,2,n);
   uint offs[1]={0}; uint work[1]={(uint)n};
   return CLExecute(h.kern_copy,1,offs,work);
  }

inline bool PyWTCWTFftMul(PyWTCWTFftMulHandle &h,const int aMem,const int bMem,const int outMem,const int n)
  {
   if(!h.ready) return false;
   CLSetKernelArgMem(h.kern_mul,0,aMem);
   CLSetKernelArgMem(h.kern_mul,1,bMem);
   CLSetKernelArgMem(h.kern_mul,2,outMem);
   CLSetKernelArg(h.kern_mul,3,n);
   uint offs[1]={0}; uint work[1]={(uint)n};
   return CLExecute(h.kern_mul,1,offs,work);
  }

inline void PyWTCWTPsiReset(PyWTCWTPsiHandle &h)
  {
   h.ctx=INVALID_HANDLE; h.prog=INVALID_HANDLE;
   h.kern_gaus=INVALID_HANDLE; h.kern_mexh=INVALID_HANDLE; h.kern_morl=INVALID_HANDLE;
   h.kern_cgau=INVALID_HANDLE; h.kern_shan=INVALID_HANDLE; h.kern_fbsp=INVALID_HANDLE; h.kern_cmor=INVALID_HANDLE;
   h.kern_cumsum=INVALID_HANDLE; h.kern_build=INVALID_HANDLE;
   h.memPsi=INVALID_HANDLE; h.memInt=INVALID_HANDLE; h.len=0; h.ready=false;
  }

inline void PyWTCWTPsiFree(PyWTCWTPsiHandle &h)
  {
   if(h.memPsi!=INVALID_HANDLE){ CLBufferFree(h.memPsi); h.memPsi=INVALID_HANDLE; }
   if(h.memInt!=INVALID_HANDLE){ CLBufferFree(h.memInt); h.memInt=INVALID_HANDLE; }
   if(h.kern_gaus!=INVALID_HANDLE){ CLKernelFree(h.kern_gaus); h.kern_gaus=INVALID_HANDLE; }
   if(h.kern_mexh!=INVALID_HANDLE){ CLKernelFree(h.kern_mexh); h.kern_mexh=INVALID_HANDLE; }
   if(h.kern_morl!=INVALID_HANDLE){ CLKernelFree(h.kern_morl); h.kern_morl=INVALID_HANDLE; }
   if(h.kern_cgau!=INVALID_HANDLE){ CLKernelFree(h.kern_cgau); h.kern_cgau=INVALID_HANDLE; }
   if(h.kern_shan!=INVALID_HANDLE){ CLKernelFree(h.kern_shan); h.kern_shan=INVALID_HANDLE; }
   if(h.kern_fbsp!=INVALID_HANDLE){ CLKernelFree(h.kern_fbsp); h.kern_fbsp=INVALID_HANDLE; }
   if(h.kern_cmor!=INVALID_HANDLE){ CLKernelFree(h.kern_cmor); h.kern_cmor=INVALID_HANDLE; }
   if(h.kern_cumsum!=INVALID_HANDLE){ CLKernelFree(h.kern_cumsum); h.kern_cumsum=INVALID_HANDLE; }
   if(h.kern_build!=INVALID_HANDLE){ CLKernelFree(h.kern_build); h.kern_build=INVALID_HANDLE; }
   if(h.prog!=INVALID_HANDLE){ CLProgramFree(h.prog); h.prog=INVALID_HANDLE; }
   h.ctx=INVALID_HANDLE; h.len=0; h.ready=false;
  }

inline bool PyWTCWTPsiInit(PyWTCWTPsiHandle &h,const int ctx)
  {
   if(h.ready && h.ctx==ctx) return true;
   PyWTCWTPsiFree(h);
   PyWTCWTPsiReset(h);
   h.ctx=ctx;
   string code=
   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
   "#define M_PI 3.1415926535897932384626433832795\n"
   "__kernel void gen_gaus(int N, double lower, double upper, int number, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=N) return; double delta=(upper-lower)/(double)(N-1);\n"
   "  double xi=lower + delta*(double)i; double x2=xi*xi; double base=sqrt(M_PI/2.0); double v=0.0;\n"
   "  if(number==1){ v=-2.0*xi*exp(-x2)/sqrt(base); }\n"
   "  else if(number==2){ v=-2.0*(2.0*x2-1.0)*exp(-x2)/sqrt(3.0*base); }\n"
   "  else if(number==3){ v=-4.0*(-2.0*pow(xi,3.0)+3.0*xi)*exp(-x2)/sqrt(15.0*base); }\n"
   "  else if(number==4){ v=4.0*(-12.0*x2+4.0*pow(xi,4.0)+3.0)*exp(-x2)/sqrt(105.0*base); }\n"
   "  else if(number==5){ v=8.0*(-4.0*pow(xi,5.0)+20.0*pow(xi,3.0)-15.0*xi)*exp(-x2)/sqrt(105.0*9.0*base); }\n"
   "  else if(number==6){ v=-8.0*(8.0*pow(xi,6.0)-60.0*pow(xi,4.0)+90.0*x2-15.0)*exp(-x2)/sqrt(105.0*9.0*11.0*base); }\n"
   "  else if(number==7){ v=-16.0*(-8.0*pow(xi,7.0)+84.0*pow(xi,5.0)-210.0*pow(xi,3.0)+105.0*xi)*exp(-x2)/sqrt(105.0*9.0*11.0*13.0*base); }\n"
   "  else if(number==8){ v=16.0*(16.0*pow(xi,8.0)-224.0*pow(xi,6.0)+840.0*pow(xi,4.0)-840.0*x2+105.0)*exp(-x2)/sqrt(105.0*9.0*11.0*13.0*15.0*base); }\n"
   "  out[i]=(double2)(v,0.0); }\n"
   "__kernel void gen_mexh(int N, double lower, double upper, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=N) return; double delta=(upper-lower)/(double)(N-1);\n"
   "  double xi=lower + delta*(double)i; double x2=xi*xi; double v=(1.0-x2)*exp(-x2*0.5)*2.0/(sqrt(3.0)*sqrt(sqrt(M_PI)));\n"
   "  out[i]=(double2)(v,0.0); }\n"
   "__kernel void gen_morl(int N, double lower, double upper, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=N) return; double delta=(upper-lower)/(double)(N-1);\n"
   "  double xi=lower + delta*(double)i; double v=cos(5.0*xi)*exp(-xi*xi*0.5); out[i]=(double2)(v,0.0); }\n"
   "__kernel void gen_cgau(int N, double lower, double upper, int number, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=N) return; double delta=(upper-lower)/(double)(N-1);\n"
   "  double xi=lower + delta*(double)i; double x2=xi*xi; double ce=cos(xi); double se=sin(xi); double ex=exp(-x2);\n"
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
   "__kernel void gen_shan(int N, double lower, double upper, double FB, double FC, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=N) return; double delta=(upper-lower)/(double)(N-1);\n"
   "  double xi=lower + delta*(double)i; double c=cos(2.0*M_PI*FC*xi); double s=sin(2.0*M_PI*FC*xi);\n"
   "  double scale=sqrt(FB); double vr=c*scale; double vi=s*scale; if(xi!=0.0){ double t=sin(xi*FB*M_PI)/(xi*FB*M_PI); vr*=t; vi*=t; }\n"
   "  out[i]=(double2)(vr,vi); }\n"
   "__kernel void gen_fbsp(int N, double lower, double upper, int M, double FB, double FC, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=N) return; double delta=(upper-lower)/(double)(N-1);\n"
   "  double xi=lower + delta*(double)i; double c=cos(2.0*M_PI*FC*xi); double s=sin(2.0*M_PI*FC*xi);\n"
   "  double scale=sqrt(FB); double vr=c*scale; double vi=s*scale; if(xi!=0.0){ double t=sin(M_PI*xi*FB/(double)M)/(M_PI*xi*FB/(double)M); double p=pow(t,(double)M); vr*=p; vi*=p; }\n"
   "  out[i]=(double2)(vr,vi); }\n"
   "__kernel void gen_cmor(int N, double lower, double upper, double FB, double FC, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=N) return; double delta=(upper-lower)/(double)(N-1);\n"
   "  double xi=lower + delta*(double)i; double c=cos(2.0*M_PI*FC*xi); double s=sin(2.0*M_PI*FC*xi);\n"
   "  double ex=exp(-(xi*xi)/FB); double scale=1.0/sqrt(M_PI*FB); out[i]=(double2)(c*ex*scale, s*ex*scale); }\n"
   "__kernel void cumsum_cplx(__global const double2* in, __global double2* out, int N, double dx, int conj){\n"
   "  if(get_global_id(0)!=0) return; double2 acc=(double2)(0.0,0.0);\n"
   "  for(int i=0;i<N;i++){ double2 v=in[i]; if(conj!=0) v.y=-v.y; acc.x += v.x*dx; acc.y += v.y*dx; out[i]=acc; }\n"
   "}\n"
   "__kernel void build_wavelet_padded(__global const double2* intpsi, int intlen, double inv, int idx_count, __global double2* out, int outlen){\n"
   "  int i=get_global_id(0); if(i>=outlen) return; if(i<idx_count){ int k=idx_count-1-i; int j=(int)floor((double)k*inv);\n"
   "    if(j<0) j=0; if(j>=intlen) j=intlen-1; out[i]=intpsi[j]; } else { out[i]=(double2)(0.0,0.0); } }\n";
   h.prog=CLProgramCreate(h.ctx,code);
   if(h.prog==INVALID_HANDLE){ PyWTCWTPsiFree(h); return false; }
   h.kern_gaus=CLKernelCreate(h.prog,"gen_gaus");
   h.kern_mexh=CLKernelCreate(h.prog,"gen_mexh");
   h.kern_morl=CLKernelCreate(h.prog,"gen_morl");
   h.kern_cgau=CLKernelCreate(h.prog,"gen_cgau");
   h.kern_shan=CLKernelCreate(h.prog,"gen_shan");
   h.kern_fbsp=CLKernelCreate(h.prog,"gen_fbsp");
   h.kern_cmor=CLKernelCreate(h.prog,"gen_cmor");
   h.kern_cumsum=CLKernelCreate(h.prog,"cumsum_cplx");
   h.kern_build=CLKernelCreate(h.prog,"build_wavelet_padded");
   if(h.kern_gaus==INVALID_HANDLE || h.kern_mexh==INVALID_HANDLE || h.kern_morl==INVALID_HANDLE ||
      h.kern_cgau==INVALID_HANDLE || h.kern_shan==INVALID_HANDLE || h.kern_fbsp==INVALID_HANDLE || h.kern_cmor==INVALID_HANDLE ||
      h.kern_cumsum==INVALID_HANDLE || h.kern_build==INVALID_HANDLE)
     { PyWTCWTPsiFree(h); return false; }
   h.ready=true;
   return true;
  }

inline bool PyWTCWTPsiEnsure(PyWTCWTPsiHandle &h,const int n)
  {
   if(!h.ready) return false;
   if(h.memPsi!=INVALID_HANDLE && h.memInt!=INVALID_HANDLE && h.len==n) return true;
   if(h.memPsi!=INVALID_HANDLE) { CLBufferFree(h.memPsi); h.memPsi=INVALID_HANDLE; }
   if(h.memInt!=INVALID_HANDLE) { CLBufferFree(h.memInt); h.memInt=INVALID_HANDLE; }
   h.memPsi=CLBufferCreate(h.ctx,n*sizeof(double)*2,CL_MEM_READ_WRITE);
   h.memInt=CLBufferCreate(h.ctx,n*sizeof(double)*2,CL_MEM_READ_WRITE);
   if(h.memPsi==INVALID_HANDLE || h.memInt==INVALID_HANDLE) return false;
   h.len=n;
   return true;
  }

inline bool PyWTCWTPsiGenerate(PyWTCWTPsiHandle &h,const PyWTContinuousWavelet &w,const int n)
  {
   if(!h.ready) return false;
   if(n<=1) return false;
   if(!PyWTCWTPsiEnsure(h,n)) return false;
   double lower=w.lower_bound;
   double upper=w.upper_bound;
   uint offs[1]={0}; uint work[1]={(uint)n};
   if(w.short_name=="gaus")
     {
      CLSetKernelArg(h.kern_gaus,0,n);
      CLSetKernelArg(h.kern_gaus,1,lower);
      CLSetKernelArg(h.kern_gaus,2,upper);
      CLSetKernelArg(h.kern_gaus,3,w.family_number);
      CLSetKernelArgMem(h.kern_gaus,4,h.memPsi);
      if(!CLExecute(h.kern_gaus,1,offs,work)) return false;
     }
   else if(w.short_name=="mexh")
     {
      CLSetKernelArg(h.kern_mexh,0,n);
      CLSetKernelArg(h.kern_mexh,1,lower);
      CLSetKernelArg(h.kern_mexh,2,upper);
      CLSetKernelArgMem(h.kern_mexh,3,h.memPsi);
      if(!CLExecute(h.kern_mexh,1,offs,work)) return false;
     }
   else if(w.short_name=="morl")
     {
      CLSetKernelArg(h.kern_morl,0,n);
      CLSetKernelArg(h.kern_morl,1,lower);
      CLSetKernelArg(h.kern_morl,2,upper);
      CLSetKernelArgMem(h.kern_morl,3,h.memPsi);
      if(!CLExecute(h.kern_morl,1,offs,work)) return false;
     }
   else if(w.short_name=="cgau")
     {
      CLSetKernelArg(h.kern_cgau,0,n);
      CLSetKernelArg(h.kern_cgau,1,lower);
      CLSetKernelArg(h.kern_cgau,2,upper);
      CLSetKernelArg(h.kern_cgau,3,w.family_number);
      CLSetKernelArgMem(h.kern_cgau,4,h.memPsi);
      if(!CLExecute(h.kern_cgau,1,offs,work)) return false;
     }
   else if(w.short_name=="shan")
     {
      CLSetKernelArg(h.kern_shan,0,n);
      CLSetKernelArg(h.kern_shan,1,lower);
      CLSetKernelArg(h.kern_shan,2,upper);
      CLSetKernelArg(h.kern_shan,3,w.bandwidth_frequency);
      CLSetKernelArg(h.kern_shan,4,w.center_frequency);
      CLSetKernelArgMem(h.kern_shan,5,h.memPsi);
      if(!CLExecute(h.kern_shan,1,offs,work)) return false;
     }
   else if(w.short_name=="fbsp")
     {
      CLSetKernelArg(h.kern_fbsp,0,n);
      CLSetKernelArg(h.kern_fbsp,1,lower);
      CLSetKernelArg(h.kern_fbsp,2,upper);
      CLSetKernelArg(h.kern_fbsp,3,w.fbsp_order);
      CLSetKernelArg(h.kern_fbsp,4,w.bandwidth_frequency);
      CLSetKernelArg(h.kern_fbsp,5,w.center_frequency);
      CLSetKernelArgMem(h.kern_fbsp,6,h.memPsi);
      if(!CLExecute(h.kern_fbsp,1,offs,work)) return false;
     }
   else if(w.short_name=="cmor")
     {
      CLSetKernelArg(h.kern_cmor,0,n);
      CLSetKernelArg(h.kern_cmor,1,lower);
      CLSetKernelArg(h.kern_cmor,2,upper);
      CLSetKernelArg(h.kern_cmor,3,w.bandwidth_frequency);
      CLSetKernelArg(h.kern_cmor,4,w.center_frequency);
      CLSetKernelArgMem(h.kern_cmor,5,h.memPsi);
      if(!CLExecute(h.kern_cmor,1,offs,work)) return false;
     }
   else
     {
      return false;
     }

   double dx=(upper-lower)/(double)(n-1);
   CLSetKernelArgMem(h.kern_cumsum,0,h.memPsi);
   CLSetKernelArgMem(h.kern_cumsum,1,h.memInt);
   CLSetKernelArg(h.kern_cumsum,2,n);
   CLSetKernelArg(h.kern_cumsum,3,dx);
   CLSetKernelArg(h.kern_cumsum,4,(int)(w.complex_cwt?1:0));
   uint work1[1]={1};
   if(!CLExecute(h.kern_cumsum,1,offs,work1)) return false;
   return true;
  }

inline bool PyWTCWTPsiBuildPadded(PyWTCWTPsiHandle &h,const int outMem,const int outlen,const double inv,const int idx_count)
  {
   if(!h.ready) return false;
   CLSetKernelArgMem(h.kern_build,0,h.memInt);
   CLSetKernelArg(h.kern_build,1,h.len);
   CLSetKernelArg(h.kern_build,2,inv);
   CLSetKernelArg(h.kern_build,3,idx_count);
   CLSetKernelArgMem(h.kern_build,4,outMem);
   CLSetKernelArg(h.kern_build,5,outlen);
   uint offs[1]={0}; uint work[1]={(uint)outlen};
   return CLExecute(h.kern_build,1,offs,work);
  }

inline void PyWTCWTConvReset(PyWTCWTConvHandle &h)
  {
   h.ctx=INVALID_HANDLE; h.prog=INVALID_HANDLE;
   h.kern_conv_real=INVALID_HANDLE; h.kern_conv_cplx=INVALID_HANDLE; h.kern_conv_cplx2=INVALID_HANDLE;
   h.kern_diff=INVALID_HANDLE; h.kern_trim=INVALID_HANDLE;
   h.memIn=INVALID_HANDLE; h.memF=INVALID_HANDLE; h.memFi=INVALID_HANDLE; h.memF2=INVALID_HANDLE; h.memOut=INVALID_HANDLE;
   h.memCoef=INVALID_HANDLE; h.memTrim=INVALID_HANDLE;
   h.lenIn=0; h.lenF=0; h.lenOut=0; h.lenCoef=0; h.lenTrim=0;
   h.ready=false;
  }

inline void PyWTCWTConvFree(PyWTCWTConvHandle &h)
  {
   if(h.memIn!=INVALID_HANDLE){ CLBufferFree(h.memIn); h.memIn=INVALID_HANDLE; }
   if(h.memF!=INVALID_HANDLE){ CLBufferFree(h.memF); h.memF=INVALID_HANDLE; }
   if(h.memFi!=INVALID_HANDLE){ CLBufferFree(h.memFi); h.memFi=INVALID_HANDLE; }
   if(h.memF2!=INVALID_HANDLE){ CLBufferFree(h.memF2); h.memF2=INVALID_HANDLE; }
   if(h.memOut!=INVALID_HANDLE){ CLBufferFree(h.memOut); h.memOut=INVALID_HANDLE; }
   if(h.memCoef!=INVALID_HANDLE){ CLBufferFree(h.memCoef); h.memCoef=INVALID_HANDLE; }
   if(h.memTrim!=INVALID_HANDLE){ CLBufferFree(h.memTrim); h.memTrim=INVALID_HANDLE; }
   if(h.kern_conv_real!=INVALID_HANDLE){ CLKernelFree(h.kern_conv_real); h.kern_conv_real=INVALID_HANDLE; }
   if(h.kern_conv_cplx!=INVALID_HANDLE){ CLKernelFree(h.kern_conv_cplx); h.kern_conv_cplx=INVALID_HANDLE; }
   if(h.kern_conv_cplx2!=INVALID_HANDLE){ CLKernelFree(h.kern_conv_cplx2); h.kern_conv_cplx2=INVALID_HANDLE; }
   if(h.kern_diff!=INVALID_HANDLE){ CLKernelFree(h.kern_diff); h.kern_diff=INVALID_HANDLE; }
   if(h.kern_trim!=INVALID_HANDLE){ CLKernelFree(h.kern_trim); h.kern_trim=INVALID_HANDLE; }
   if(h.prog!=INVALID_HANDLE){ CLProgramFree(h.prog); h.prog=INVALID_HANDLE; }
   if(h.ctx!=INVALID_HANDLE){ CLContextFree(h.ctx); h.ctx=INVALID_HANDLE; }
   h.lenIn=0; h.lenF=0; h.lenOut=0; h.lenCoef=0; h.lenTrim=0;
   h.ready=false;
  }

inline bool PyWTCWTConvInit(PyWTCWTConvHandle &h,const int ctx=INVALID_HANDLE)
  {
   if(h.ready && (ctx==INVALID_HANDLE || h.ctx==ctx)) return true;
   PyWTCWTConvReset(h);
   h.ctx=(ctx!=INVALID_HANDLE ? ctx : CLCreateContextGPUFloat64("SpectralPyWTCWT"));
   if(h.ctx==INVALID_HANDLE) return false;
   string code=
   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
   "__kernel void conv_full_real(__global const double* x, int N, __global const double* h, int F, __global double* out){\n"
   "  int o=get_global_id(0); int O=N+F-1; if(o>=O) return; double sum=0.0;\n"
   "  for(int j=0;j<F;j++){ int idx=o-j; if(idx>=0 && idx<N){ sum += h[j]*x[idx]; }} out[o]=sum; }\n"
   "__kernel void conv_full_cplx(__global const double* x, int N, __global const double* hr, __global const double* hi, int F, __global double2* out){\n"
   "  int o=get_global_id(0); int O=N+F-1; if(o>=O) return; double sumr=0.0; double sumi=0.0;\n"
   "  for(int j=0;j<F;j++){ int idx=o-j; if(idx>=0 && idx<N){ double v=x[idx]; sumr += hr[j]*v; sumi += hi[j]*v; }} out[o]=(double2)(sumr,sumi); }\n"
   "__kernel void conv_full_cplx2(__global const double* x, int N, __global const double2* h, int F, __global double2* out){\n"
   "  int o=get_global_id(0); int O=N+F-1; if(o>=O) return; double sumr=0.0; double sumi=0.0;\n"
   "  for(int j=0;j<F;j++){ int idx=o-j; if(idx>=0 && idx<N){ double v=x[idx]; double2 hv=h[j]; sumr += hv.x*v; sumi += hv.y*v; }} out[o]=(double2)(sumr,sumi); }\n"
   "__kernel void diff_scale_cplx(__global const double2* in, int n, double scale, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=n-1) return; double2 a=in[i]; double2 b=in[i+1];\n"
   "  out[i]=(double2)((b.x-a.x)*scale, (b.y-a.y)*scale); }\n"
   "__kernel void trim_center_cplx(__global const double2* in, int start, int n, __global double2* out, int outlen){\n"
   "  int i=get_global_id(0); if(i>=outlen) return; int j=start+i; if(j<0) j=0; if(j>=n) j=n-1; out[i]=in[j]; }\n";
   h.prog=CLProgramCreate(h.ctx,code);
   if(h.prog==INVALID_HANDLE){ PyWTCWTConvFree(h); return false; }
   h.kern_conv_real=CLKernelCreate(h.prog,"conv_full_real");
   h.kern_conv_cplx=CLKernelCreate(h.prog,"conv_full_cplx");
   h.kern_conv_cplx2=CLKernelCreate(h.prog,"conv_full_cplx2");
   h.kern_diff=CLKernelCreate(h.prog,"diff_scale_cplx");
   h.kern_trim=CLKernelCreate(h.prog,"trim_center_cplx");
   if(h.kern_conv_real==INVALID_HANDLE || h.kern_conv_cplx==INVALID_HANDLE || h.kern_conv_cplx2==INVALID_HANDLE ||
      h.kern_diff==INVALID_HANDLE || h.kern_trim==INVALID_HANDLE){ PyWTCWTConvFree(h); return false; }
   h.ready=true;
   return true;
  }

inline bool PyWTCWTConvEnsure(PyWTCWTConvHandle &h,const int N,const int F,const int conv_len,const int coef_len,const int outlen)
  {
   if(!h.ready) return false;
   if(N<=0 || F<=0 || conv_len<=0 || coef_len<=0 || outlen<=0) return false;
   if(h.memIn==INVALID_HANDLE || h.lenIn!=N)
     {
      if(h.memIn!=INVALID_HANDLE) CLBufferFree(h.memIn);
      h.memIn=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
      if(h.memIn==INVALID_HANDLE) return false;
      h.lenIn=N;
     }
   if(h.memF2==INVALID_HANDLE || h.lenF!=F)
     {
      if(h.memF2!=INVALID_HANDLE) CLBufferFree(h.memF2);
      h.memF2=CLBufferCreate(h.ctx,F*sizeof(double)*2,CL_MEM_READ_WRITE);
      if(h.memF2==INVALID_HANDLE) return false;
      h.lenF=F;
     }
   if(h.memOut==INVALID_HANDLE || h.lenOut!=conv_len)
     {
      if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
      h.memOut=CLBufferCreate(h.ctx,conv_len*sizeof(double)*2,CL_MEM_READ_WRITE);
      if(h.memOut==INVALID_HANDLE) return false;
      h.lenOut=conv_len;
     }
   if(h.memCoef==INVALID_HANDLE || h.lenCoef!=coef_len)
     {
      if(h.memCoef!=INVALID_HANDLE) CLBufferFree(h.memCoef);
      h.memCoef=CLBufferCreate(h.ctx,coef_len*sizeof(double)*2,CL_MEM_READ_WRITE);
      if(h.memCoef==INVALID_HANDLE) return false;
      h.lenCoef=coef_len;
     }
   if(h.memTrim==INVALID_HANDLE || h.lenTrim!=outlen)
     {
      if(h.memTrim!=INVALID_HANDLE) CLBufferFree(h.memTrim);
      h.memTrim=CLBufferCreate(h.ctx,outlen*sizeof(double)*2,CL_MEM_READ_WRITE);
      if(h.memTrim==INVALID_HANDLE) return false;
      h.lenTrim=outlen;
     }
   return true;
  }

inline bool PyWT_ConvFullReal(const double &data[],const double &filt[],double &out[])
  {
   int N=ArraySize(data), F=ArraySize(filt);
   if(N<=0 || F<=0) return false;
   int O=N+F-1; ArrayResize(out,O);
   static PyWTCWTConvHandle h; if(!h.ready) PyWTCWTConvReset(h);
   if(!PyWTCWTConvInit(h)) return false;
   if(h.memIn!=INVALID_HANDLE) CLBufferFree(h.memIn);
   if(h.memF!=INVALID_HANDLE) CLBufferFree(h.memF);
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memIn=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   h.memF=CLBufferCreate(h.ctx,F*sizeof(double),CL_MEM_READ_ONLY);
   h.memOut=CLBufferCreate(h.ctx,O*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memIn==INVALID_HANDLE || h.memF==INVALID_HANDLE || h.memOut==INVALID_HANDLE) return false;
   CLBufferWrite(h.memIn,data);
   CLBufferWrite(h.memF,filt);
   CLSetKernelArgMem(h.kern_conv_real,0,h.memIn);
   CLSetKernelArg(h.kern_conv_real,1,N);
   CLSetKernelArgMem(h.kern_conv_real,2,h.memF);
   CLSetKernelArg(h.kern_conv_real,3,F);
   CLSetKernelArgMem(h.kern_conv_real,4,h.memOut);
   uint offs[1]={0}; uint work[1]={(uint)O};
   if(!CLExecute(h.kern_conv_real,1,offs,work)) return false;
   CLBufferRead(h.memOut,out);
   return true;
  }

inline bool PyWT_ConvFullComplex(const double &data[],const Complex64 &filt[],Complex64 &out[])
  {
   int N=ArraySize(data), F=ArraySize(filt);
   if(N<=0 || F<=0) return false;
   int O=N+F-1; ArrayResize(out,O);
   double fr[]; double fi[]; ArrayResize(fr,F); ArrayResize(fi,F);
   for(int i=0;i<F;i++){ fr[i]=filt[i].re; fi[i]=filt[i].im; }
   static PyWTCWTConvHandle h; if(!h.ready) PyWTCWTConvReset(h);
   if(!PyWTCWTConvInit(h)) return false;
   if(h.memIn!=INVALID_HANDLE) CLBufferFree(h.memIn);
   if(h.memF!=INVALID_HANDLE) CLBufferFree(h.memF);
   if(h.memFi!=INVALID_HANDLE) CLBufferFree(h.memFi);
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memIn=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   h.memF=CLBufferCreate(h.ctx,F*sizeof(double),CL_MEM_READ_ONLY);
   h.memFi=CLBufferCreate(h.ctx,F*sizeof(double),CL_MEM_READ_ONLY);
   h.memOut=CLBufferCreate(h.ctx,O*sizeof(double)*2,CL_MEM_WRITE_ONLY);
   if(h.memIn==INVALID_HANDLE || h.memF==INVALID_HANDLE || h.memFi==INVALID_HANDLE || h.memOut==INVALID_HANDLE) return false;
   CLBufferWrite(h.memIn,data);
   CLBufferWrite(h.memF,fr);
   CLBufferWrite(h.memFi,fi);
   CLSetKernelArgMem(h.kern_conv_cplx,0,h.memIn);
   CLSetKernelArg(h.kern_conv_cplx,1,N);
   CLSetKernelArgMem(h.kern_conv_cplx,2,h.memF);
   CLSetKernelArgMem(h.kern_conv_cplx,3,h.memFi);
   CLSetKernelArg(h.kern_conv_cplx,4,F);
   CLSetKernelArgMem(h.kern_conv_cplx,5,h.memOut);
   uint offs[1]={0}; uint work[1]={(uint)O};
   if(!CLExecute(h.kern_conv_cplx,1,offs,work)) return false;
   double buf[]; ArrayResize(buf,2*O);
   CLBufferRead(h.memOut,buf);
   for(int i=0;i<O;i++) out[i]=Cx(buf[2*i],buf[2*i+1]);
   return true;
  }

// CWT main (1D data). Output is [n_scales][N] Complex64.
inline bool PyWT_CWT(const double &data[],const double &scales[],const string wavelet_name,
                     const double sampling_period,const string method,const int precision,
                     Complex64 &out[][],double &frequencies[])
  {
   PyWTContinuousWavelet w;
   if(!PyWT_ParseCWTName(wavelet_name,w)) return false;
   int N=ArraySize(data);
   int S=ArraySize(scales);
   if(N<=0 || S<=0) return false;
   // Pre-allocate output as S x N to avoid resizing subarrays of struct arrays
   ArrayResize(out,S,N);

   // methods
   string mth = method;
   StringToLower(mth);
   if(mth!="conv" && mth!="fft") return false;
   bool use_fft = (mth=="fft");

   int n = 1 << precision;
   if(n<=1) return false;
   double range = w.upper_bound - w.lower_bound;
   double step = range / (double)(n-1);

   int size_scale0 = -1;
   CLFFTPlan plan; CLFFTReset(plan);
   static PyWTCWTFftMulHandle mul; if(!mul.ready) PyWTCWTFftMulReset(mul);
   static PyWTCWTPsiHandle psi; if(!psi.ready) PyWTCWTPsiReset(psi);
   static PyWTCWTConvHandle conv; if(!conv.ready) PyWTCWTConvReset(conv);

   if(!use_fft)
     {
      if(!PyWTCWTConvInit(conv)) return false;
      if(!PyWTCWTPsiInit(psi,conv.ctx)) { PyWTCWTConvFree(conv); return false; }
      if(!PyWTCWTPsiGenerate(psi,w,n)) { PyWTCWTConvFree(conv); return false; }
      if(conv.memIn!=INVALID_HANDLE && conv.lenIn!=N){ CLBufferFree(conv.memIn); conv.memIn=INVALID_HANDLE; conv.lenIn=0; }
      if(conv.memIn==INVALID_HANDLE)
        {
         conv.memIn=CLBufferCreate(conv.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
         if(conv.memIn==INVALID_HANDLE) { PyWTCWTConvFree(conv); return false; }
         conv.lenIn=N;
        }
      CLBufferWrite(conv.memIn,data);
     }

   for(int si=0; si<S; si++)
     {
      double scale=scales[si];
      if(scale<=0.0) return false;

      int J = (int)MathFloor(scale * range + 1.0);
      if(J<=0) return false;
      double inv = 1.0/(scale*step);
      int idx_count=0;
      int psi_len = n;
      for(int k=0;k<J;k++){ int j=(int)MathFloor((double)k*inv); if(j < psi_len){ idx_count++; } }
      if(idx_count<=0) return false;

      Complex64 conv_c[];
      if(use_fft)
        {
         int size_scale = PyWT_NextFastLen(N + idx_count - 1);
         if(size_scale != size_scale0)
           {
            size_scale0 = size_scale;
            if(!CLFFTInit(plan,size_scale)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
            if(!CLFFTEnsureBatchBuffers(plan,1)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
            if(!PyWTCWTFftMulInit(mul,plan.ctx)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
            if(!PyWTCWTFftEnsureData(mul,size_scale)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
            if(!PyWTCWTPsiInit(psi,plan.ctx)) { PyWTCWTPsiFree(psi); PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
            if(!PyWTCWTPsiGenerate(psi,w,n)) { PyWTCWTPsiFree(psi); PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
            // FFT of data (padded) -> mul.memData
            Complex64 in[]; ArrayResize(in,size_scale);
            for(int i=0;i<size_scale;i++){ in[i]=Cx((i<N)?data[i]:0.0,0.0); }
            if(!CLFFTUploadComplexBatch(plan,in,1)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
            if(!CLFFTExecuteBatchFromMemA_NoRead(plan,1,false)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
            if(!PyWTCWTFftCopy(mul,plan.memFinal,mul.memData,size_scale)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
           }
         // FFT of wavelet (scaled, reversed, padded) in GPU
         if(!PyWTCWTPsiBuildPadded(psi,plan.memA,size_scale,inv,idx_count)) { PyWTCWTPsiFree(psi); PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
         if(!CLFFTExecuteBatchFromMemA_NoRead(plan,1,false)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
         // multiply in GPU: mul.memData (data FFT) * plan.memFinal (wavelet FFT) -> plan.memA
         if(!PyWTCWTFftMul(mul,mul.memData,plan.memFinal,plan.memA,size_scale)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
         // inverse FFT on product
         if(!CLFFTExecuteBatchFromMemA_NoRead(plan,1,true)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
         int conv_len = N + idx_count - 1;
         ArrayResize(conv_c,conv_len);
         double buf[]; ArrayResize(buf,2*size_scale);
         CLBufferRead(plan.memFinal,buf);
         for(int i=0;i<conv_len;i++) conv_c[i]=Cx(buf[2*i],buf[2*i+1]);
        }
      else
        {
         int conv_len = N + idx_count - 1;
         int coef_len = conv_len - 1;
         if(conv_len<=1 || coef_len<=0) return false;
         if(!PyWTCWTConvEnsure(conv,N,idx_count,conv_len,coef_len,N)) { PyWTCWTConvFree(conv); return false; }
         if(!PyWTCWTPsiBuildPadded(psi,conv.memF2,idx_count,inv,idx_count)) { PyWTCWTConvFree(conv); return false; }
         CLSetKernelArgMem(conv.kern_conv_cplx2,0,conv.memIn);
         CLSetKernelArg(conv.kern_conv_cplx2,1,N);
         CLSetKernelArgMem(conv.kern_conv_cplx2,2,conv.memF2);
         CLSetKernelArg(conv.kern_conv_cplx2,3,idx_count);
         CLSetKernelArgMem(conv.kern_conv_cplx2,4,conv.memOut);
         uint offs1[1]={0}; uint work1[1]={(uint)conv_len};
         if(!CLExecute(conv.kern_conv_cplx2,1,offs1,work1)) { PyWTCWTConvFree(conv); return false; }

         double scale_fac = -MathSqrt(scale);
         CLSetKernelArgMem(conv.kern_diff,0,conv.memOut);
         CLSetKernelArg(conv.kern_diff,1,conv_len);
         CLSetKernelArg(conv.kern_diff,2,scale_fac);
         CLSetKernelArgMem(conv.kern_diff,3,conv.memCoef);
         uint work2[1]={(uint)coef_len};
         if(!CLExecute(conv.kern_diff,1,offs1,work2)) { PyWTCWTConvFree(conv); return false; }

         double d = ((double)coef_len - (double)N) / 2.0;
         int start = (int)MathFloor(d);
         int end = coef_len - (int)MathCeil(d);
         if(d < 0.0) { PyWTCWTConvFree(conv); return false; }
         int L = end - start;
         if(L<=0 || L!=N) { PyWTCWTConvFree(conv); return false; }
         CLSetKernelArgMem(conv.kern_trim,0,conv.memCoef);
         CLSetKernelArg(conv.kern_trim,1,start);
         CLSetKernelArg(conv.kern_trim,2,coef_len);
         CLSetKernelArgMem(conv.kern_trim,3,conv.memTrim);
         CLSetKernelArg(conv.kern_trim,4,L);
         uint work3[1]={(uint)L};
         if(!CLExecute(conv.kern_trim,1,offs1,work3)) { PyWTCWTConvFree(conv); return false; }

         double buf[]; ArrayResize(buf,2*L);
         CLBufferRead(conv.memTrim,buf);
         if(L!=N) { PyWTCWTConvFree(conv); return false; }
         for(int i=0;i<N;i++) out[si][i]=Cx(buf[2*i],buf[2*i+1]);
         continue;
        }

      // coef = -sqrt(scale) * diff(conv)
      int conv_len = ArraySize(conv_c);
      if(conv_len<2) return false;
      int coef_len = conv_len - 1;
      Complex64 coef[]; ArrayResize(coef,coef_len);
      double scale_fac = -MathSqrt(scale);
      for(int i=0;i<coef_len;i++)
        {
         coef[i].re = (conv_c[i+1].re - conv_c[i].re) * scale_fac;
         coef[i].im = (conv_c[i+1].im - conv_c[i].im) * scale_fac;
        }

      // trim to data length
      double d = ((double)coef_len - (double)N) / 2.0;
      int start = (int)MathFloor(d);
      int end = coef_len - (int)MathCeil(d);
      if(d > 0.0)
        {
         int L=end-start;
         if(L!=N) return false;
         for(int i=0;i<N;i++) out[si][i]=coef[start+i];
        }
      else if(d==0.0)
        {
         if(coef_len!=N) return false;
         for(int i=0;i<N;i++) out[si][i]=coef[i];
        }
      else
        {
         // too small scale
         return false;
        }
     }

   PyWTCWTPsiFree(psi);
   PyWTCWTFftMulFree(mul);
   CLFFTFree(plan);

   // frequencies
   if(!PyWT_Scale2Frequency(w,scales,precision,frequencies)) return false;
   for(int i=0;i<ArraySize(frequencies);i++) frequencies[i] /= sampling_period;
   return true;
  }

// Flat output version to avoid 2D struct arrays in callers.
// out_flat size = S*N, index = si*N + i
inline bool PyWT_CWT_Flat(const double &data[],const double &scales[],const string wavelet_name,
                          const double sampling_period,const string method,const int precision,
                          Complex64 &out_flat[],double &frequencies[])
  {
   PyWTContinuousWavelet w;
   if(!PyWT_ParseCWTName(wavelet_name,w)) return false;
   int N=ArraySize(data);
   int S=ArraySize(scales);
   if(N<=0 || S<=0) return false;
   ArrayResize(out_flat, S*N);

   string mth = method;
   StringToLower(mth);
   if(mth!="conv" && mth!="fft") return false;
   bool use_fft = (mth=="fft");

   int n = 1 << precision;
   if(n<=1) return false;
   double range = w.upper_bound - w.lower_bound;
   double step = range / (double)(n-1);

   int size_scale0 = -1;
   CLFFTPlan plan; CLFFTReset(plan);
   static PyWTCWTFftMulHandle mul; if(!mul.ready) PyWTCWTFftMulReset(mul);
   static PyWTCWTPsiHandle psi; if(!psi.ready) PyWTCWTPsiReset(psi);
   static PyWTCWTConvHandle conv; if(!conv.ready) PyWTCWTConvReset(conv);

   if(!use_fft)
     {
      if(!PyWTCWTConvInit(conv)) return false;
      if(!PyWTCWTPsiInit(psi,conv.ctx)) { PyWTCWTConvFree(conv); return false; }
      if(!PyWTCWTPsiGenerate(psi,w,n)) { PyWTCWTConvFree(conv); return false; }
      if(conv.memIn!=INVALID_HANDLE && conv.lenIn!=N){ CLBufferFree(conv.memIn); conv.memIn=INVALID_HANDLE; conv.lenIn=0; }
      if(conv.memIn==INVALID_HANDLE)
        {
         conv.memIn=CLBufferCreate(conv.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
         if(conv.memIn==INVALID_HANDLE) { PyWTCWTConvFree(conv); return false; }
         conv.lenIn=N;
        }
      CLBufferWrite(conv.memIn,data);
     }

   for(int si=0; si<S; si++)
     {
      double scale=scales[si];
      if(scale<=0.0) return false;

      int J = (int)MathFloor(scale * range + 1.0);
      if(J<=0) return false;
      double inv = 1.0/(scale*step);
      int idx_count=0;
      int psi_len = n;
      for(int k=0;k<J;k++){ int j=(int)MathFloor((double)k*inv); if(j < psi_len){ idx_count++; } }
      if(idx_count<=0) return false;

      Complex64 conv_c[];
      if(use_fft)
        {
         int size_scale = PyWT_NextFastLen(N + idx_count - 1);
         if(size_scale != size_scale0)
           {
            size_scale0 = size_scale;
            if(!CLFFTInit(plan,size_scale)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
            if(!CLFFTEnsureBatchBuffers(plan,1)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
            if(!PyWTCWTFftMulInit(mul,plan.ctx)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
            if(!PyWTCWTFftEnsureData(mul,size_scale)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
            if(!PyWTCWTPsiInit(psi,plan.ctx)) { PyWTCWTPsiFree(psi); PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
            if(!PyWTCWTPsiGenerate(psi,w,n)) { PyWTCWTPsiFree(psi); PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
            Complex64 in[]; ArrayResize(in,size_scale);
            for(int i=0;i<size_scale;i++){ in[i]=Cx((i<N)?data[i]:0.0,0.0); }
            if(!CLFFTUploadComplexBatch(plan,in,1)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
            if(!CLFFTExecuteBatchFromMemA_NoRead(plan,1,false)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
            if(!PyWTCWTFftCopy(mul,plan.memFinal,mul.memData,size_scale)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
           }
         if(!PyWTCWTPsiBuildPadded(psi,plan.memA,size_scale,inv,idx_count)) { PyWTCWTPsiFree(psi); PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
         if(!CLFFTExecuteBatchFromMemA_NoRead(plan,1,false)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
         if(!PyWTCWTFftMul(mul,mul.memData,plan.memFinal,plan.memA,size_scale)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
         if(!CLFFTExecuteBatchFromMemA_NoRead(plan,1,true)) { PyWTCWTFftMulFree(mul); CLFFTFree(plan); return false; }
         int conv_len = N + idx_count - 1;
         ArrayResize(conv_c,conv_len);
         double buf[]; ArrayResize(buf,2*size_scale);
         CLBufferRead(plan.memFinal,buf);
         for(int i=0;i<conv_len;i++) conv_c[i]=Cx(buf[2*i],buf[2*i+1]);
        }
      else
        {
         int conv_len = N + idx_count - 1;
         int coef_len = conv_len - 1;
         if(conv_len<=1 || coef_len<=0) return false;
         if(!PyWTCWTConvEnsure(conv,N,idx_count,conv_len,coef_len,N)) { PyWTCWTConvFree(conv); return false; }
         if(!PyWTCWTPsiBuildPadded(psi,conv.memF2,idx_count,inv,idx_count)) { PyWTCWTConvFree(conv); return false; }
         CLSetKernelArgMem(conv.kern_conv_cplx2,0,conv.memIn);
         CLSetKernelArg(conv.kern_conv_cplx2,1,N);
         CLSetKernelArgMem(conv.kern_conv_cplx2,2,conv.memF2);
         CLSetKernelArg(conv.kern_conv_cplx2,3,idx_count);
         CLSetKernelArgMem(conv.kern_conv_cplx2,4,conv.memOut);
         uint offs1[1]={0}; uint work1[1]={(uint)conv_len};
         if(!CLExecute(conv.kern_conv_cplx2,1,offs1,work1)) { PyWTCWTConvFree(conv); return false; }

         double scale_fac = -MathSqrt(scale);
         CLSetKernelArgMem(conv.kern_diff,0,conv.memOut);
         CLSetKernelArg(conv.kern_diff,1,conv_len);
         CLSetKernelArg(conv.kern_diff,2,scale_fac);
         CLSetKernelArgMem(conv.kern_diff,3,conv.memCoef);
         uint work2[1]={(uint)coef_len};
         if(!CLExecute(conv.kern_diff,1,offs1,work2)) { PyWTCWTConvFree(conv); return false; }

         double d = ((double)coef_len - (double)N) / 2.0;
         int start = (int)MathFloor(d);
         int end = coef_len - (int)MathCeil(d);
         if(d < 0.0) { PyWTCWTConvFree(conv); return false; }
         int L = end - start;
         if(L<=0 || L!=N) { PyWTCWTConvFree(conv); return false; }
         CLSetKernelArgMem(conv.kern_trim,0,conv.memCoef);
         CLSetKernelArg(conv.kern_trim,1,start);
         CLSetKernelArg(conv.kern_trim,2,coef_len);
         CLSetKernelArgMem(conv.kern_trim,3,conv.memTrim);
         CLSetKernelArg(conv.kern_trim,4,L);
         uint work3[1]={(uint)L};
         if(!CLExecute(conv.kern_trim,1,offs1,work3)) { PyWTCWTConvFree(conv); return false; }

         double buf[]; ArrayResize(buf,2*L);
         CLBufferRead(conv.memTrim,buf);
         if(L!=N) { PyWTCWTConvFree(conv); return false; }
         int base = si*N;
         for(int i=0;i<N;i++) out_flat[base+i]=Cx(buf[2*i],buf[2*i+1]);
         continue;
        }

      int conv_len = ArraySize(conv_c);
      if(conv_len<2) return false;
      int coef_len = conv_len - 1;
      Complex64 coef[]; ArrayResize(coef,coef_len);
      double scale_fac = -MathSqrt(scale);
      for(int i=0;i<coef_len;i++)
        {
         coef[i].re = (conv_c[i+1].re - conv_c[i].re) * scale_fac;
         coef[i].im = (conv_c[i+1].im - conv_c[i].im) * scale_fac;
        }

      double d = ((double)coef_len - (double)N) / 2.0;
      int start = (int)MathFloor(d);
      int end = coef_len - (int)MathCeil(d);
      if(d > 0.0)
        {
         int L=end-start;
         if(L!=N) return false;
         int base = si*N;
         for(int i=0;i<N;i++) out_flat[base+i]=coef[start+i];
        }
      else if(d==0.0)
        {
         if(coef_len!=N) return false;
         int base = si*N;
         for(int i=0;i<N;i++) out_flat[base+i]=coef[i];
        }
      else
        {
         return false;
        }
     }

   PyWTCWTPsiFree(psi);
   PyWTCWTFftMulFree(mul);
   CLFFTFree(plan);

   if(!PyWT_Scale2Frequency(w,scales,precision,frequencies)) return false;
   for(int i=0;i<ArraySize(frequencies);i++) frequencies[i] /= sampling_period;
   return true;
  }

// CWT batch for 2D input with flat output to avoid 2D struct arrays.
// axis=1 (rows) -> out_rows = S*rows, out_cols = cols, index = (si*rows + r)*out_cols + c
// axis=0 (cols) -> out_rows = S*cols, out_cols = rows, index = (si*cols + c)*out_cols + r
inline bool PyWT_CWT_Axis_Flat(const double &data[][],const double &scales[],const string wavelet_name,
                               const double sampling_period,const string method,const int precision,const int axis,
                               Complex64 &out_flat[],int &out_rows,int &out_cols,double &frequencies[])
  {
   int rows=ArrayRange(data,0);
   int cols=ArrayRange(data,1);
   if(rows<=0 || cols<=0) return false;
   int S=ArraySize(scales);
   if(S<=0) return false;

   if(axis==1 || axis==-1)
     {
      out_rows = S*rows;
      out_cols = cols;
      ArrayResize(out_flat, out_rows*out_cols);
      for(int r=0;r<rows;r++)
        {
         double series[]; ArrayResize(series,cols);
         for(int c=0;c<cols;c++) series[c]=data[r][c];
         Complex64 rowOutFlat[];
         double freqs[];
         if(!PyWT_CWT_Flat(series,scales,wavelet_name,sampling_period,method,precision,rowOutFlat,freqs)) return false;
         if(r==0){ ArrayResize(frequencies,ArraySize(freqs)); for(int i=0;i<ArraySize(freqs);i++) frequencies[i]=freqs[i]; }
         for(int si=0;si<S;si++)
           {
            int row_idx = si*rows + r;
            int base_in = si*cols;
            int base_out = row_idx*out_cols;
            for(int i=0;i<cols;i++) out_flat[base_out+i]=rowOutFlat[base_in+i];
           }
        }
      return true;
     }
   if(axis==0)
     {
      out_rows = S*cols;
      out_cols = rows;
      ArrayResize(out_flat, out_rows*out_cols);
      for(int c=0;c<cols;c++)
        {
         double series[]; ArrayResize(series,rows);
         for(int r=0;r<rows;r++) series[r]=data[r][c];
         Complex64 colOutFlat[];
         double freqs[];
         if(!PyWT_CWT_Flat(series,scales,wavelet_name,sampling_period,method,precision,colOutFlat,freqs)) return false;
         if(c==0){ ArrayResize(frequencies,ArraySize(freqs)); for(int i=0;i<ArraySize(freqs);i++) frequencies[i]=freqs[i]; }
         for(int si=0;si<S;si++)
           {
            int row_idx = si*cols + c;
            int base_in = si*rows;
            int base_out = row_idx*out_cols;
            for(int r=0;r<rows;r++) out_flat[base_out+r]=colOutFlat[base_in+r];
           }
        }
      return true;
     }
   return false;
  }

// Wrapper kept for compatibility; 2D struct arrays are not supported reliably.
inline bool PyWT_CWT_Axis(const double &data[][],const double &scales[],const string wavelet_name,
                          const double sampling_period,const string method,const int precision,const int axis,
                          Complex64 &out[][],double &frequencies[])
  {
   ArrayResize(out,0,0);
   ArrayResize(frequencies,0);
   return false;
  }

#endif // __SPECTRAL_PYWT_CWT_MQH__
