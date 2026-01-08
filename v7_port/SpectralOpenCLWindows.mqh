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

inline double _acosh_d(const double x)
  {
   return MathLog(x + MathSqrt(x*x - 1.0));
  }

inline double _cosh_d(const double x)
  {
   return 0.5*(MathExp(x) + MathExp(-x));
  }

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
   "  else if(type==21){ double norm=params[0]; double mod_pi=2.0*M_PI/N; double temp=mod_pi*(i - N/2.0 + 0.5); double dot=0.0; for(int k=1;k<=ncoeff;k++){ dot += coeffs[k-1]*cos(temp*(double)k);} double val=1.0 + 2.0*dot; if(norm>0.5){ double temp2=mod_pi*(((N-1.0)/2.0) - N/2.0 + 0.5); double dot2=0.0; for(int k=1;k<=ncoeff;k++){ dot2 += coeffs[k-1]*cos(temp2*(double)k);} double scale=1.0/(1.0+2.0*dot2); val*=scale; } w=val; }\n"
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

// Chebyshev window using OpenCL FFT (no CPU FFT fallback)
inline bool CLWindowChebwin(CLFFTPlan &plan,const int M,const double at,const bool sym,double &out[])
  {
   if(M<=1)
     {
      ArrayResize(out,M);
      for(int i=0;i<M;i++) out[i]=1.0;
      return true;
     }
   int Mx=M;
   bool trunc=false;
   if(!sym) { Mx=M+1; trunc=true; }
   double order=(double)(Mx-1);
   double beta=_cosh_d((1.0/order) * _acosh_d(MathPow(10.0, MathAbs(at)/20.0)));
   double Npi = PI/(double)Mx;
   bool odd = ((Mx & 1)!=0);
   Complex64 p[];
   ArrayResize(p,Mx);
   for(int i=0;i<Mx;i++)
     {
      double x=beta*MathCos((double)i*Npi);
      double real=0.0;
      if(x>1.0) real=_cosh_d(order*_acosh_d(x));
      else if(x<-1.0) real=(odd?1.0:-1.0)*_cosh_d(order*_acosh_d(-x));
      else real=MathCos(order*MathAcos(x));
      if(odd)
         p[i]=Cx(real,0.0);
      else
        {
         double ang=(double)i*Npi;
         p[i]=Cx(real*MathCos(ang), real*MathSin(ang));
        }
     }

   Complex64 spec[];
   if(!CLFFTExecute(plan,p,spec,false)) return false;

   double wfull[];
   ArrayResize(wfull,Mx);
   for(int i=0;i<Mx;i++) wfull[i]=spec[i].re;

   double w[];
   if(odd)
     {
      int n=(Mx+1)/2;
      ArrayResize(w,Mx);
      int idx=0;
      for(int i=n-1;i>=1;i--) { w[idx++]=wfull[i]; }
      for(int i=0;i<n;i++) { w[idx++]=wfull[i]; }
     }
   else
     {
      int n= Mx/2 + 1;
      ArrayResize(w,Mx);
      int idx=0;
      for(int i=n-1;i>=1;i--) { w[idx++]=wfull[i]; }
      for(int i=1;i<n;i++) { w[idx++]=wfull[i]; }
     }

   double wmax=0.0;
   for(int i=0;i<Mx;i++) if(w[i]>wmax) wmax=w[i];
   if(wmax==0.0) wmax=1.0;
   for(int i=0;i<Mx;i++) w[i]/=wmax;

   if(trunc)
     {
      ArrayResize(out,M);
      for(int i=0;i<M;i++) out[i]=w[i];
     }
   else
     {
      ArrayResize(out,Mx);
      ArrayCopy(out,w);
     }
   return true;
  }

inline bool CLWindowTaylor(CLWinHandle &h,const int M,const int nbar,const double sll,const bool norm,const bool sym,double &out[])
  {
   if(M<=1)
     {
      ArrayResize(out,M);
      for(int i=0;i<M;i++) out[i]=1.0;
      return true;
     }
   int Mx=M;
   bool trunc=false;
   if(!sym) { Mx=M+1; trunc=true; }

   if(nbar<1)
     {
      ArrayResize(out,Mx);
      for(int i=0;i<Mx;i++) out[i]=1.0;
      if(trunc)
        {
         ArrayResize(out,M);
        }
      return true;
     }

   double B=MathPow(10.0,sll/20.0);
   double A=_acosh_d(B)/PI;
   double s2=(double)(nbar*nbar)/(A*A + (nbar-0.5)*(nbar-0.5));
   int mcount=nbar-1;
   double Fm[];
   ArrayResize(Fm,mcount);
   for(int mi=0;mi<mcount;mi++)
     {
      double m=mi+1;
      double numer_sign=(mi%2==0)?1.0:-1.0;
      double numer=1.0;
      for(int k=0;k<mcount;k++)
        {
         double mk=k+1;
         double term=1.0 - (m*m)/(s2*(A*A + (mk-0.5)*(mk-0.5)));
         numer*=term;
        }
      double denom=1.0;
      for(int k=0;k<mi;k++)
        {
         double mk=k+1;
         denom*= (1.0 - (m*m)/(mk*mk));
        }
      for(int k=mi+1;k<mcount;k++)
        {
         double mk=k+1;
         denom*= (1.0 - (m*m)/(mk*mk));
        }
      Fm[mi]=numer_sign*numer/(2.0*denom);
     }

   double params[];
   ArrayResize(params,1);
   params[0]=norm?1.0:0.0;
   double w[];
   if(!CLWindowGenerate(h,WIN_TAYLOR,Mx,true,params,Fm,w)) return false;
   if(trunc)
     {
      ArrayResize(out,M);
      for(int i=0;i<M;i++) out[i]=w[i];
     }
   else
     {
      ArrayResize(out,Mx);
      ArrayCopy(out,w);
     }
   return true;
  }

inline bool CLGetWindowParams(const string win,const int Nx,const bool fftbins,const double &params[],double &out[])
  {
   string name=StringToLower(win);
   static CLWinHandle h;
   static CLFFTPlan plan;
   if(!h.ready) CLWinReset(h);
   if(!plan.ready) CLFFTReset(plan);
   double p[]; double c[];
   ArrayResize(p,0); ArrayResize(c,0);
   int type=WIN_HANN;

   if(name=="boxcar" || name=="box" || name=="ones" || name=="rect" || name=="rectangular") type=WIN_BOXCAR;
   else if(name=="triang" || name=="triangle" || name=="tri") type=WIN_TRIANG;
   else if(name=="parzen" || name=="parz" || name=="par") type=WIN_PARZEN;
   else if(name=="bohman" || name=="bman" || name=="bmn") type=WIN_BOHMAN;
   else if(name=="blackman" || name=="black" || name=="blk") type=WIN_BLACKMAN;
   else if(name=="blackmanharris" || name=="blackharr" || name=="bkh") type=WIN_BLACKMANHARRIS;
   else if(name=="nuttall" || name=="nutl" || name=="nut") type=WIN_NUTTALL;
   else if(name=="flattop" || name=="flat" || name=="flt") type=WIN_FLATTOP;
   else if(name=="bartlett" || name=="bart" || name=="brt") type=WIN_BARTLETT;
   else if(name=="hann" || name=="hanning" || name=="han") type=WIN_HANN;
   else if(name=="hamming" || name=="hamm" || name=="ham") type=WIN_HAMMING;
   else if(name=="barthann" || name=="brthan" || name=="bth") type=WIN_BARTHANN;
   else if(name=="cosine" || name=="halfcosine") type=WIN_COSINE;
   else if(name=="tukey" || name=="tuk")
     {
      ArrayResize(p,1); p[0]=(ArraySize(params)>0?params[0]:0.5);
      type=WIN_TUKEY;
     }
   else if(name=="kaiser" || name=="ksr")
     {
      ArrayResize(p,1); p[0]=(ArraySize(params)>0?params[0]:0.0);
      type=WIN_KAISER;
     }
   else if(name=="gaussian" || name=="gauss" || name=="gss")
     {
      ArrayResize(p,1); p[0]=(ArraySize(params)>0?params[0]:1.0);
      type=WIN_GAUSSIAN;
     }
   else if(name=="general_gaussian" || name=="general gaussian" || name=="general gauss" || name=="general_gauss" || name=="ggs")
     {
      ArrayResize(p,2);
      p[0]=(ArraySize(params)>0?params[0]:1.0);
      p[1]=(ArraySize(params)>1?params[1]:1.0);
      type=WIN_GENERAL_GAUSSIAN;
     }
   else if(name=="general_cosine" || name=="general cosine")
     {
      int n=ArraySize(params);
      ArrayResize(c,n);
      for(int i=0;i<n;i++) c[i]=params[i];
      type=WIN_GENERAL_COSINE;
     }
   else if(name=="general_hamming")
     {
      ArrayResize(p,1); p[0]=(ArraySize(params)>0?params[0]:0.54);
      type=WIN_GENERAL_HAMMING;
     }
   else if(name=="exponential" || name=="poisson")
     {
      ArrayResize(p,2);
      p[0]=(ArraySize(params)>0?params[0]:1.0);
      p[1]=(ArraySize(params)>1?params[1]:-1.0);
      type=WIN_EXPONENTIAL;
     }
   else if(name=="chebwin" || name=="cheb")
     {
      double at=(ArraySize(params)>0?params[0]:100.0);
      return CLWindowChebwin(plan,Nx,at,!fftbins,out);
     }
   else if(name=="taylor")
     {
      int nbar=(ArraySize(params)>0?(int)params[0]:4);
      double sll=(ArraySize(params)>1?params[1]:30.0);
      bool norm=true; if(ArraySize(params)>2) norm=(params[2]!=0.0);
      return CLWindowTaylor(h,Nx,nbar,sll,norm,!fftbins,out);
     }

   return CLWindowGenerate(h,type,Nx,!fftbins,p,c,out);
  }

inline bool CLGetWindow(const string win,const int Nx,const bool fftbins,double &out[])
  {
   double params[];
   ArrayResize(params,0);
   return CLGetWindowParams(win,Nx,fftbins,params,out);
  }

#endif
