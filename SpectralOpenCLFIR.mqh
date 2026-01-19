#ifndef __SPECTRAL_OPENCL_FIR_MQH__
#define __SPECTRAL_OPENCL_FIR_MQH__

#include "SpectralCommon.mqh"
#include "SpectralOpenCLFFT.mqh"
#include "SpectralOpenCLWindows.mqh"
#include "SpectralOpenCLCommon.mqh"

// OpenCL FIR design kernels (firwin / firls / minimum_phase helpers)

struct CLFIRPlan
  {
   int ctx;
   int prog;
   int kern_firwin;
   int kern_firwin_build;
   int kern_firls_q;
   int kern_firls_b;
   int kern_firls_Q;
   int kern_chol_solve;
   int kern_vec_mul;
   int kern_vec_add;
   int kern_vec_scale;
   int kern_sum;
   int kern_min;
   int kern_max;
   int kern_vec_log;
   int kern_vec_exp;
   int kern_vec_sqrt;
   int kern_vec_abs;
   int kern_vec_cos;
   int kern_vec_sinc;
   int kern_firwin_cos;
   int kern_firls_coeffs;
   int kern_cplx_exp;
   int kern_cplx_mul_real;
   int kern_cplx_mul_cplx;
   int kern_cplx_from_real;
   int kern_cplx_real;
   int kern_interp1d;
   int kern_make_shift;
   int memBands;
   int memDesired;
   int memWeight;
   int memWin;
   int memH;
   int memHC;
   int memQ;
   int memq;
   int memb;
   int memx;
   int memFlag;
   bool ready;
  };

inline void CLFIRReset(CLFIRPlan &p)
  {
   p.ctx=INVALID_HANDLE; p.prog=INVALID_HANDLE;
   p.kern_firwin=INVALID_HANDLE; p.kern_firwin_build=INVALID_HANDLE;
   p.kern_firls_q=INVALID_HANDLE; p.kern_firls_b=INVALID_HANDLE;
   p.kern_firls_Q=INVALID_HANDLE; p.kern_chol_solve=INVALID_HANDLE;
   p.kern_vec_mul=INVALID_HANDLE; p.kern_vec_add=INVALID_HANDLE;
   p.kern_vec_scale=INVALID_HANDLE; p.kern_sum=INVALID_HANDLE; p.kern_min=INVALID_HANDLE; p.kern_max=INVALID_HANDLE;
   p.kern_vec_log=INVALID_HANDLE; p.kern_vec_exp=INVALID_HANDLE;
   p.kern_vec_sqrt=INVALID_HANDLE; p.kern_vec_abs=INVALID_HANDLE;
   p.kern_vec_cos=INVALID_HANDLE; p.kern_vec_sinc=INVALID_HANDLE;
   p.kern_firwin_cos=INVALID_HANDLE; p.kern_firls_coeffs=INVALID_HANDLE;
   p.kern_cplx_exp=INVALID_HANDLE; p.kern_cplx_mul_real=INVALID_HANDLE;
   p.kern_cplx_mul_cplx=INVALID_HANDLE; p.kern_cplx_from_real=INVALID_HANDLE;
   p.kern_cplx_real=INVALID_HANDLE;
   p.kern_interp1d=INVALID_HANDLE; p.kern_make_shift=INVALID_HANDLE;
   p.memBands=INVALID_HANDLE; p.memDesired=INVALID_HANDLE; p.memWeight=INVALID_HANDLE;
   p.memWin=INVALID_HANDLE; p.memH=INVALID_HANDLE; p.memHC=INVALID_HANDLE;
   p.memQ=INVALID_HANDLE; p.memq=INVALID_HANDLE; p.memb=INVALID_HANDLE;
   p.memx=INVALID_HANDLE; p.memFlag=INVALID_HANDLE;
   p.ready=false;
  }

inline void CLFIRFree(CLFIRPlan &p)
  {
   if(p.memBands!=INVALID_HANDLE) { CLBufferFree(p.memBands); p.memBands=INVALID_HANDLE; }
   if(p.memDesired!=INVALID_HANDLE) { CLBufferFree(p.memDesired); p.memDesired=INVALID_HANDLE; }
   if(p.memWeight!=INVALID_HANDLE) { CLBufferFree(p.memWeight); p.memWeight=INVALID_HANDLE; }
   if(p.memWin!=INVALID_HANDLE) { CLBufferFree(p.memWin); p.memWin=INVALID_HANDLE; }
   if(p.memH!=INVALID_HANDLE) { CLBufferFree(p.memH); p.memH=INVALID_HANDLE; }
   if(p.memHC!=INVALID_HANDLE) { CLBufferFree(p.memHC); p.memHC=INVALID_HANDLE; }
   if(p.memQ!=INVALID_HANDLE) { CLBufferFree(p.memQ); p.memQ=INVALID_HANDLE; }
   if(p.memq!=INVALID_HANDLE) { CLBufferFree(p.memq); p.memq=INVALID_HANDLE; }
   if(p.memb!=INVALID_HANDLE) { CLBufferFree(p.memb); p.memb=INVALID_HANDLE; }
   if(p.memx!=INVALID_HANDLE) { CLBufferFree(p.memx); p.memx=INVALID_HANDLE; }
   if(p.memFlag!=INVALID_HANDLE) { CLBufferFree(p.memFlag); p.memFlag=INVALID_HANDLE; }
   if(p.kern_firwin!=INVALID_HANDLE) { CLKernelFree(p.kern_firwin); p.kern_firwin=INVALID_HANDLE; }
   if(p.kern_firwin_build!=INVALID_HANDLE) { CLKernelFree(p.kern_firwin_build); p.kern_firwin_build=INVALID_HANDLE; }
   if(p.kern_firls_q!=INVALID_HANDLE) { CLKernelFree(p.kern_firls_q); p.kern_firls_q=INVALID_HANDLE; }
   if(p.kern_firls_b!=INVALID_HANDLE) { CLKernelFree(p.kern_firls_b); p.kern_firls_b=INVALID_HANDLE; }
   if(p.kern_firls_Q!=INVALID_HANDLE) { CLKernelFree(p.kern_firls_Q); p.kern_firls_Q=INVALID_HANDLE; }
   if(p.kern_chol_solve!=INVALID_HANDLE) { CLKernelFree(p.kern_chol_solve); p.kern_chol_solve=INVALID_HANDLE; }
   if(p.kern_vec_mul!=INVALID_HANDLE) { CLKernelFree(p.kern_vec_mul); p.kern_vec_mul=INVALID_HANDLE; }
   if(p.kern_vec_add!=INVALID_HANDLE) { CLKernelFree(p.kern_vec_add); p.kern_vec_add=INVALID_HANDLE; }
   if(p.kern_vec_scale!=INVALID_HANDLE) { CLKernelFree(p.kern_vec_scale); p.kern_vec_scale=INVALID_HANDLE; }
   if(p.kern_sum!=INVALID_HANDLE) { CLKernelFree(p.kern_sum); p.kern_sum=INVALID_HANDLE; }
   if(p.kern_min!=INVALID_HANDLE) { CLKernelFree(p.kern_min); p.kern_min=INVALID_HANDLE; }
   if(p.kern_max!=INVALID_HANDLE) { CLKernelFree(p.kern_max); p.kern_max=INVALID_HANDLE; }
   if(p.kern_vec_log!=INVALID_HANDLE) { CLKernelFree(p.kern_vec_log); p.kern_vec_log=INVALID_HANDLE; }
   if(p.kern_vec_exp!=INVALID_HANDLE) { CLKernelFree(p.kern_vec_exp); p.kern_vec_exp=INVALID_HANDLE; }
   if(p.kern_vec_sqrt!=INVALID_HANDLE) { CLKernelFree(p.kern_vec_sqrt); p.kern_vec_sqrt=INVALID_HANDLE; }
   if(p.kern_vec_abs!=INVALID_HANDLE) { CLKernelFree(p.kern_vec_abs); p.kern_vec_abs=INVALID_HANDLE; }
   if(p.kern_vec_cos!=INVALID_HANDLE) { CLKernelFree(p.kern_vec_cos); p.kern_vec_cos=INVALID_HANDLE; }
   if(p.kern_vec_sinc!=INVALID_HANDLE) { CLKernelFree(p.kern_vec_sinc); p.kern_vec_sinc=INVALID_HANDLE; }
   if(p.kern_firwin_cos!=INVALID_HANDLE) { CLKernelFree(p.kern_firwin_cos); p.kern_firwin_cos=INVALID_HANDLE; }
   if(p.kern_firls_coeffs!=INVALID_HANDLE) { CLKernelFree(p.kern_firls_coeffs); p.kern_firls_coeffs=INVALID_HANDLE; }
   if(p.kern_cplx_exp!=INVALID_HANDLE) { CLKernelFree(p.kern_cplx_exp); p.kern_cplx_exp=INVALID_HANDLE; }
   if(p.kern_cplx_mul_real!=INVALID_HANDLE) { CLKernelFree(p.kern_cplx_mul_real); p.kern_cplx_mul_real=INVALID_HANDLE; }
   if(p.kern_cplx_mul_cplx!=INVALID_HANDLE) { CLKernelFree(p.kern_cplx_mul_cplx); p.kern_cplx_mul_cplx=INVALID_HANDLE; }
   if(p.kern_cplx_from_real!=INVALID_HANDLE) { CLKernelFree(p.kern_cplx_from_real); p.kern_cplx_from_real=INVALID_HANDLE; }
   if(p.kern_cplx_real!=INVALID_HANDLE) { CLKernelFree(p.kern_cplx_real); p.kern_cplx_real=INVALID_HANDLE; }
   if(p.kern_interp1d!=INVALID_HANDLE) { CLKernelFree(p.kern_interp1d); p.kern_interp1d=INVALID_HANDLE; }
   if(p.kern_make_shift!=INVALID_HANDLE) { CLKernelFree(p.kern_make_shift); p.kern_make_shift=INVALID_HANDLE; }
   if(p.prog!=INVALID_HANDLE) { CLProgramFree(p.prog); p.prog=INVALID_HANDLE; }
   if(p.ctx!=INVALID_HANDLE)  { CLContextFree(p.ctx); p.ctx=INVALID_HANDLE; }
   p.ready=false;
  }

inline bool CLFIRInit(CLFIRPlan &p)
  {
   if(p.ready) return true;
   CLFIRReset(p);
   p.ctx=CLCreateContextGPUFloat64("SpectralOpenCLFIR");
   if(p.ctx==INVALID_HANDLE) return false;

   string code=
   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"#ifndef M_PI\n"
"#define M_PI 3.1415926535897932384626433832795\n"
"#endif\n"
   "inline double sinc_d(double x){ if(fabs(x)<1e-20) return 1.0; return sin(M_PI*x)/(M_PI*x); }\n"
   "__kernel void firwin_core(__global const double* win, int numtaps, __global const double* bands, int steps, int scale, __global double* h, __global double* hc){\n"
   "  int i=get_global_id(0); if(i>=numtaps) return;\n"
   "  double alpha=0.5*((double)numtaps-1.0);\n"
   "  double m=(double)i - alpha; if(fabs(m)<1e-20) m=1.0e-20;\n"
   "  double temp=0.0;\n"
   "  for(int s=0;s<steps;s++){\n"
   "    double left=bands[2*s+0]; double right=bands[2*s+1];\n"
   "    if(left==0.0) left=1.0e-20; if(right==0.0) right=1.0e-20;\n"
   "    temp += right * (sin(right*m*M_PI)/(right*m*M_PI));\n"
   "    temp -= left  * (sin(left*m*M_PI)/(left*m*M_PI));\n"
   "  }\n"
   "  temp *= win[i];\n"
   "  h[i]=temp;\n"
   "  if(scale!=0){\n"
   "    double left=bands[0]; double right=bands[1];\n"
   "    double sf=0.0; if(left==0.0) sf=0.0; else if(right==1.0) sf=1.0; else sf=0.5*(left+right);\n"
   "    double c=cos(M_PI*m*sf);\n"
   "    hc[i]=temp*c;\n"
   "  }\n"
   "}\n"
   "__kernel void firwin_build(__global const double* win, int numtaps, __global const double* bands, int steps, __global double* h){\n"
   "  int i=get_global_id(0); if(i>=numtaps) return;\n"
   "  double alpha=0.5*((double)numtaps-1.0);\n"
   "  double m=(double)i - alpha;\n"
   "  double temp=0.0;\n"
   "  for(int s=0;s<steps;s++){\n"
   "    double left=bands[2*s+0]; double right=bands[2*s+1];\n"
   "    temp += right * sinc_d(right*m);\n"
   "    temp -= left  * sinc_d(left*m);\n"
   "  }\n"
   "  h[i]=temp * win[i];\n"
   "}\n"
   "__kernel void firls_q(int numtaps, int nbands, __global const double* bands, __global const double* weight, __global double* q){\n"
   "  int n=get_global_id(0); if(n>=numtaps) return;\n"
   "  double sum=0.0;\n"
   "  for(int s=0;s<nbands;s++){\n"
   "    double b1=bands[2*s+0]; double b2=bands[2*s+1];\n"
   "    double w=weight[s];\n"
   "    double term = (b2*sinc_d(b2*(double)n) - b1*sinc_d(b1*(double)n));\n"
   "    sum += w*term;\n"
   "  }\n"
   "  q[n]=sum;\n"
   "}\n"
   "__kernel void firls_b(int M, int nbands, __global const double* bands, __global const double* desired, __global const double* weight, __global double* b){\n"
   "  int n=get_global_id(0); if(n> M) return;\n"
   "  double sum=0.0;\n"
   "  double nn=(double)n;\n"
   "  for(int s=0;s<nbands;s++){\n"
   "    double f1=bands[2*s+0]; double f2=bands[2*s+1];\n"
   "    double d1=desired[2*s+0]; double d2=desired[2*s+1];\n"
   "    double m=(d2-d1)/(f2-f1); double c=d1 - f1*m;\n"
   "    double term2=f2*(m*f2+c)*sinc_d(f2*nn);\n"
   "    double term1=f1*(m*f1+c)*sinc_d(f1*nn);\n"
   "    double extra2=0.0, extra1=0.0;\n"
   "    if(n==0){ extra2 = -m*f2*f2*0.5; extra1 = -m*f1*f1*0.5; }\n"
   "    else { double denom=M_PI*nn; double denom2=denom*denom; extra2 = m*cos(M_PI*nn*f2)/denom2; extra1 = m*cos(M_PI*nn*f1)/denom2; }\n"
   "    double bandterm=(term2+extra2) - (term1+extra1);\n"
   "    sum += weight[s]*bandterm;\n"
   "  }\n"
   "  b[n]=sum;\n"
   "}\n"
   "__kernel void firls_buildQ(int N, __global const double* q, __global double* Q){\n"
   "  int gid=get_global_id(0); if(gid>=N*N) return;\n"
   "  int i=gid / N; int j=gid - i*N;\n"
   "  int idx1 = (i>j)? (i-j) : (j-i);\n"
   "  int idx2 = i + j;\n"
   "  Q[gid] = q[idx1] + q[idx2];\n"
   "}\n"
   "__kernel void chol_solve(__global double* A, __global double* b, __global double* x, int N, __global int* flag){\n"
   "  int gid=get_global_id(0); if(gid>0) return;\n"
   "  for(int i=0;i<N;i++){\n"
   "    for(int j=0;j<=i;j++){\n"
   "      double sum=A[i*N+j];\n"
   "      for(int k=0;k<j;k++) sum -= A[i*N+k]*A[j*N+k];\n"
   "      if(i==j){ if(sum<=0.0){ flag[0]=1; return; } A[i*N+i]=sqrt(sum); }\n"
   "      else { A[i*N+j]=sum/A[j*N+j]; }\n"
   "    }\n"
   "  }\n"
   "  for(int i=0;i<N;i++){\n"
   "    double sum=b[i];\n"
   "    for(int k=0;k<i;k++) sum -= A[i*N+k]*b[k];\n"
   "    b[i]=sum/A[i*N+i];\n"
   "  }\n"
   "  for(int i=N-1;i>=0;i--){\n"
   "    double sum=b[i];\n"
   "    for(int k=i+1;k<N;k++) sum -= A[k*N+i]*x[k];\n"
   "    x[i]=sum/A[i*N+i];\n"
   "  }\n"
   "}\n"
   "__kernel void vec_mul(__global const double* a, __global const double* b, int n, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=n) return; out[i]=a[i]*b[i]; }\n"
   "__kernel void vec_add(__global const double* a, double s, int n, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=n) return; out[i]=a[i]+s; }\n"
   "__kernel void vec_scale(__global double* a, int n, double s){\n"
   "  int i=get_global_id(0); if(i>=n) return; a[i]*=s; }\n"
   "__kernel void reduce_sum(__global const double* a, int n, __global double* out){\n"
   "  int gid=get_global_id(0); if(gid>0) return; double s=0.0; for(int i=0;i<n;i++) s+=a[i]; out[0]=s; }\n"
   "__kernel void reduce_min(__global const double* a, int n, __global double* out){\n"
   "  int gid=get_global_id(0); if(gid>0) return; double m=a[0]; for(int i=1;i<n;i++) if(a[i]<m) m=a[i]; out[0]=m; }\n"
   "__kernel void reduce_max(__global const double* a, int n, __global double* out){\n"
   "  int gid=get_global_id(0); if(gid>0) return; double m=a[0]; for(int i=1;i<n;i++) if(a[i]>m) m=a[i]; out[0]=m; }\n"
   "__kernel void vec_log(__global const double* a, int n, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=n) return; out[i]=log(a[i]); }\n"
   "__kernel void vec_exp(__global const double* a, int n, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=n) return; out[i]=exp(a[i]); }\n"
   "__kernel void vec_sqrt(__global const double* a, int n, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=n) return; out[i]=sqrt(a[i]); }\n"
   "__kernel void vec_abs_cplx(__global const double2* a, int n, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=n) return; double2 v=a[i]; out[i]=sqrt(v.x*v.x+v.y*v.y); }\n"
   "__kernel void vec_cos(__global const double* a, int n, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=n) return; out[i]=cos(a[i]); }\n"
   "__kernel void vec_sinc(__global const double* a, int n, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=n) return; out[i]=sinc_d(a[i]); }\n"
   "__kernel void firwin_cos(int numtaps, double sf, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=numtaps) return; double alpha=0.5*((double)numtaps-1.0); double m=(double)i - alpha; out[i]=cos(M_PI*m*sf); }\n"
   "__kernel void firls_coeffs(__global const double* a, int numtaps, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=numtaps) return; int M=(numtaps-1)/2;\n"
   "  if(i<M) out[i]=a[M-i]; else if(i==M) out[i]=2.0*a[0]; else out[i]=a[i-M];\n"
   "}\n"
   "__kernel void cplx_from_real(__global const double* a, int n, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=n) return; out[i]=(double2)(a[i],0.0); }\n"
   "__kernel void cplx_real(__global const double2* a, int n, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=n) return; out[i]=a[i].x; }\n"
   "__kernel void cplx_mul_real(__global const double2* a, __global const double* b, int n, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=n) return; out[i]=(double2)(a[i].x*b[i], a[i].y*b[i]); }\n"
   "__kernel void cplx_mul_cplx(__global const double2* a, __global const double2* b, int n, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=n) return; double2 x=a[i]; double2 y=b[i]; out[i]=(double2)(x.x*y.x - x.y*y.y, x.x*y.y + x.y*y.x); }\n"
   "__kernel void cplx_exp(__global const double2* a, int n, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=n) return; double2 v=a[i]; double ex=exp(v.x); out[i]=(double2)(ex*cos(v.y), ex*sin(v.y)); }\n"
   "__kernel void interp1d(__global const double* x, int nx, __global const double* xp, __global const double* fp, int np, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=nx) return; double xi=x[i];\n"
   "  if(xi<=xp[0]) { out[i]=fp[0]; return; }\n"
   "  if(xi>=xp[np-1]) { out[i]=fp[np-1]; return; }\n"
   "  int k=0; while(k<np-2 && xi>xp[k+1]) k++; double x0=xp[k]; double x1=xp[k+1];\n"
   "  double y0=fp[k]; double y1=fp[k+1]; double t=(xi-x0)/(x1-x0); out[i]=y0 + t*(y1-y0); }\n"
   "__kernel void make_shift(__global const double* x, int nx, double alpha, double nyq, int ftype, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=nx) return; double ang = -alpha * M_PI * x[i] / nyq;\n"
   "  double c=cos(ang); double s=sin(ang); if(ftype>2){ double t=c; c=-s; s=t; } out[i]=(double2)(c,s); }\n";

   string build_log="";
   p.prog=CLProgramCreate(p.ctx,code,build_log);
   if(p.prog==INVALID_HANDLE)
     {
      PrintFormat("SpectralOpenCLFIR: CLProgramCreate failed (err=%d)", GetLastError());
      if(build_log!="") Print("SpectralOpenCLFIR build log:\n", build_log);
      CLFIRFree(p);
      return false;
     }
   p.kern_firwin=CLKernelCreate(p.prog,"firwin_core");
   p.kern_firwin_build=CLKernelCreate(p.prog,"firwin_build");
   p.kern_firls_q=CLKernelCreate(p.prog,"firls_q");
   p.kern_firls_b=CLKernelCreate(p.prog,"firls_b");
   p.kern_firls_Q=CLKernelCreate(p.prog,"firls_buildQ");
   p.kern_chol_solve=CLKernelCreate(p.prog,"chol_solve");
   p.kern_vec_mul=CLKernelCreate(p.prog,"vec_mul");
   p.kern_vec_add=CLKernelCreate(p.prog,"vec_add");
   p.kern_vec_scale=CLKernelCreate(p.prog,"vec_scale");
   p.kern_sum=CLKernelCreate(p.prog,"reduce_sum");
   p.kern_min=CLKernelCreate(p.prog,"reduce_min");
   p.kern_max=CLKernelCreate(p.prog,"reduce_max");
   p.kern_vec_log=CLKernelCreate(p.prog,"vec_log");
   p.kern_vec_exp=CLKernelCreate(p.prog,"vec_exp");
   p.kern_vec_sqrt=CLKernelCreate(p.prog,"vec_sqrt");
   p.kern_vec_abs=CLKernelCreate(p.prog,"vec_abs_cplx");
   p.kern_vec_cos=CLKernelCreate(p.prog,"vec_cos");
   p.kern_vec_sinc=CLKernelCreate(p.prog,"vec_sinc");
   p.kern_firwin_cos=CLKernelCreate(p.prog,"firwin_cos");
   p.kern_firls_coeffs=CLKernelCreate(p.prog,"firls_coeffs");
   p.kern_cplx_exp=CLKernelCreate(p.prog,"cplx_exp");
   p.kern_cplx_mul_real=CLKernelCreate(p.prog,"cplx_mul_real");
   p.kern_cplx_mul_cplx=CLKernelCreate(p.prog,"cplx_mul_cplx");
   p.kern_cplx_from_real=CLKernelCreate(p.prog,"cplx_from_real");
   p.kern_cplx_real=CLKernelCreate(p.prog,"cplx_real");
   p.kern_interp1d=CLKernelCreate(p.prog,"interp1d");
   p.kern_make_shift=CLKernelCreate(p.prog,"make_shift");

   if(p.kern_firwin==INVALID_HANDLE || p.kern_firwin_build==INVALID_HANDLE ||
      p.kern_firls_q==INVALID_HANDLE || p.kern_firls_b==INVALID_HANDLE || p.kern_firls_Q==INVALID_HANDLE ||
      p.kern_chol_solve==INVALID_HANDLE || p.kern_vec_mul==INVALID_HANDLE || p.kern_vec_add==INVALID_HANDLE ||
      p.kern_vec_scale==INVALID_HANDLE || p.kern_sum==INVALID_HANDLE || p.kern_min==INVALID_HANDLE || p.kern_max==INVALID_HANDLE ||
      p.kern_vec_log==INVALID_HANDLE || p.kern_vec_exp==INVALID_HANDLE || p.kern_vec_sqrt==INVALID_HANDLE ||
      p.kern_vec_abs==INVALID_HANDLE || p.kern_vec_cos==INVALID_HANDLE || p.kern_vec_sinc==INVALID_HANDLE ||
      p.kern_firwin_cos==INVALID_HANDLE || p.kern_firls_coeffs==INVALID_HANDLE ||
      p.kern_cplx_exp==INVALID_HANDLE || p.kern_cplx_mul_real==INVALID_HANDLE || p.kern_cplx_mul_cplx==INVALID_HANDLE ||
      p.kern_cplx_from_real==INVALID_HANDLE || p.kern_cplx_real==INVALID_HANDLE ||
      p.kern_interp1d==INVALID_HANDLE || p.kern_make_shift==INVALID_HANDLE)
     { CLFIRFree(p); return false; }

   p.ready=true;
   return true;
  }

#endif
