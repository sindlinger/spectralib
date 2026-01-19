#ifndef __SAVGOL_MQH__
#define __SAVGOL_MQH__

#include "SpectralCommon.mqh"
#include "SpectralArrayTools.mqh"
#include "SpectralLinalg.mqh"
#include "SpectralOpenCL.mqh"
#include "SpectralOpenCLCommon.mqh"

inline double float_factorial(const int n)
  {
   if(n<0) return 0.0;
   if(n>=171) return DBL_MAX;
   double f=1.0;
   for(int i=2;i<=n;i++) f*=i;
   return f;
  }

// Evaluate polynomial with coefficients p (highest power first) at x array
inline void _polyval(const double &p[],const double &x[],double &y[])
  {
   int np=ArraySize(p);
   int nx=ArraySize(x);
   ArrayResize(y,nx);
   for(int i=0;i<nx;i++)
     {
      double v=0.0;
      for(int k=0;k<np;k++) v=v*x[i]+p[k];
      y[i]=v;
     }
  }

// Derivative of polynomial coefficients p (highest power first)
inline void _polyder(const double &p[],const int m,double &out[])
  {
   if(m<=0)
     { ArrayResize(out,ArraySize(p)); ArrayCopy(out,p); return; }
   int n=ArraySize(p);
   if(n<=m) { ArrayResize(out,1); out[0]=0.0; return; }
   double cur[];
   ArrayResize(cur,n);
   ArrayCopy(cur,p);
   int curN=n;
   for(int d=0;d<m;d++)
     {
      for(int i=0;i<curN-1;i++)
        {
         int power=(curN-1-i);
         cur[i]=cur[i]*power;
        }
      curN--;
      ArrayResize(cur,curN);
     }
   ArrayResize(out,curN);
   ArrayCopy(out,cur);
  }

// Least squares solution for A*c = y (A: m x n)
inline bool _lstsq(const double &A[],int m,int n,const double &y[],double &c[])
  {
   // Use normal equations: (A^T A) c = A^T y
   if(m<=0 || n<=0) return false;
   if(ArraySize(A) < m*n) return false;
   if(ArraySize(y) < m) return false;
   double ATA[];
   ArrayResize(ATA,n*n);
   ArrayInitialize(ATA,0.0);
   for(int r=0;r<n;r++)
     {
      for(int c2=0;c2<n;c2++)
        {
         double sum=0.0;
         for(int i=0;i<m;i++) sum+=A[i*n+r]*A[i*n+c2];
         ATA[r*n+c2]=sum;
        }
     }
   double ATy[];
   ArrayResize(ATy,n);
   for(int r=0;r<n;r++)
     {
      double sum=0.0;
      for(int i=0;i<m;i++) sum+=A[i*n+r]*y[i];
      ATy[r]=sum;
     }
   double ATAinv[];
   if(!MatInvert(ATA,ATAinv,n)) return false;
   ArrayResize(c,n);
   for(int r=0;r<n;r++)
     {
      double sum=0.0;
      for(int k=0;k<n;k++) sum+=ATAinv[r*n+k]*ATy[k];
      c[r]=sum;
     }
   return true;
  }

// Savitzky-Golay coefficients
inline bool savgol_coeffs(int window_length,int polyorder,int deriv,double delta,int pos,const bool use_conv,double &coeffs[])
  {
   if(polyorder>=window_length) return false;
   if(deriv<0) return false;
   int halflen=window_length/2;
   bool even=(window_length%2)==0;
   double posf;
   if(pos<0)
     {
      posf=even ? (halflen-0.5) : halflen;
     }
   else posf=pos;
   if(posf<0 || posf>=window_length) return false;

   int m=polyorder+1;
   // x values
   double x[];
   ArrayResize(x,window_length);
   for(int i=0;i<window_length;i++) x[i]= (double)i - posf;
   if(use_conv)
     {
      // reverse for convolution
      for(int i=0;i<window_length/2;i++)
        { double t=x[i]; x[i]=x[window_length-1-i]; x[window_length-1-i]=t; }
     }
   // build A (window_length x m)
   double A[];
   ArrayResize(A,window_length*m);
   for(int i=0;i<window_length;i++)
     {
      double xp=1.0;
      for(int j=0;j<m;j++)
        { A[i*m+j]=xp; xp*=x[i]; }
     }
   // d vector (derivative selector)
   double dvec[];
   ArrayResize(dvec,m);
   ArrayInitialize(dvec,0.0);
   if(deriv>polyorder)
     {
      ArrayResize(coeffs,window_length);
      ArrayInitialize(coeffs,0.0);
      return true;
     }
   dvec[deriv]=float_factorial(deriv)/MathPow(delta,deriv);

   // Solve (A^T A) c = d
   double ATA[];
   ArrayResize(ATA,m*m);
   ArrayInitialize(ATA,0.0);
   for(int r=0;r<m;r++)
     {
      for(int c2=0;c2<m;c2++)
        {
         double sum=0.0;
         for(int i=0;i<window_length;i++) sum+=A[i*m+r]*A[i*m+c2];
         ATA[r*m+c2]=sum;
        }
     }
   double ATAinv[];
   if(!MatInvert(ATA,ATAinv,m)) return false;
   double cvec[];
   ArrayResize(cvec,m);
   for(int r=0;r<m;r++)
     {
      double sum=0.0;
      for(int k=0;k<m;k++) sum+=ATAinv[r*m+k]*dvec[k];
      cvec[r]=sum;
     }

   // coeffs = A * cvec
   ArrayResize(coeffs,window_length);
   for(int i=0;i<window_length;i++)
     {
      double sum=0.0;
      for(int j=0;j<m;j++) sum+=A[i*m+j]*cvec[j];
      coeffs[i]=sum;
     }
   return true;
  }

// Overload matching cupy/scipy signature: use = "conv" or "dot"
inline bool savgol_coeffs(int window_length,int polyorder,int deriv,double delta,int pos,const string use,double &coeffs[])
  {
   string u=use;
   StringToLower(u);
   if(u!="conv" && u!="dot") return false;
   bool use_conv=(u=="conv");
   return savgol_coeffs(window_length,polyorder,deriv,delta,pos,use_conv,coeffs);
  }

// 1D convolve with boundary modes
enum ConvolveMode
  {
   MODE_MIRROR=0,
   MODE_CONSTANT=1,
   MODE_NEAREST=2,
   MODE_WRAP=3
  };

inline double _get_bound(const double &x[],int idx,const int mode,const double cval)
  {
   int N=ArraySize(x);
   if(idx>=0 && idx<N) return x[idx];
   if(mode==MODE_CONSTANT) return cval;
   if(mode==MODE_NEAREST)
     {
      if(idx<0) return x[0];
      return x[N-1];
     }
   if(mode==MODE_WRAP)
     {
      int j=idx%N;
      if(j<0) j+=N;
      return x[j];
     }
   // mirror
   int j=idx;
   while(j<0 || j>=N)
     {
      if(j<0) j=-j;
      if(j>=N) j=2*N-2-j;
     }
   return x[j];
  }

inline void convolve1d(const double &x[],const double &h[],const int mode,const double cval,double &y[])
  {
   int N=ArraySize(x);
   int M=ArraySize(h);
   ArrayResize(y,N);
   int center=M/2;
   for(int i=0;i<N;i++)
     {
      double sum=0.0;
      for(int k=0;k<M;k++)
        {
         int idx=i + k - center;
         double v=_get_bound(x,idx,mode,cval);
         sum+=v*h[k];
        }
      y[i]=sum;
     }
  }

// Fit polynomial at edges and replace in y (1D only)
inline void _fit_edge_1d(const double &x[],int window_start,int window_stop,
                         int interp_start,int interp_stop,
                         int polyorder,int deriv,double delta,double &y[])
  {
   // slice x_edge
   int wlen=window_stop-window_start;
   double x_edge[];
   ArrayResize(x_edge,wlen);
   for(int i=0;i<wlen;i++) x_edge[i]=x[window_start+i];

   // polyfit x_edge vs index
   // Build Vandermonde
   double A[];
   ArrayResize(A,wlen*(polyorder+1));
   for(int i=0;i<wlen;i++)
     {
      double xp=1.0;
      for(int j=0;j<polyorder+1;j++)
        { A[i*(polyorder+1)+j]=xp; xp*=(double)i; }
     }
   double cvec[];
   if(!_lstsq(A,wlen,polyorder+1,x_edge,cvec)) return;

   // derivative if needed
   double cder[];
   _polyder(cvec,deriv,cder);

   // evaluate polynomial
   int len=interp_stop-interp_start;
   double xi[];
   ArrayResize(xi,len);
   for(int i=0;i<len;i++) xi[i]=(double)(interp_start-window_start+i);
   double vals[];
   _polyval(cder,xi,vals);
   double scale=MathPow(delta,deriv);
   if(scale==0.0) scale=1.0;
   for(int i=0;i<len;i++) y[interp_start+i]=vals[i]/scale;
  }

inline void savgol_filter_1d(const double &x[],int window_length,int polyorder,
                             int deriv,double delta,const string mode,double cval,
                             double &y[])
  {
   if(window_length<1) { ArrayResize(y,0); return; }
   if(polyorder>=window_length) return;
   string mcheck=mode;
   StringToLower(mcheck);
   if(mcheck!="mirror" && mcheck!="constant" && mcheck!="nearest" && mcheck!="interp" && mcheck!="wrap")
     { ArrayResize(y,0); return; }
   int N=ArraySize(x);
   // coeffs
   double coeffs[];
   int pos=-1;
   if(!savgol_coeffs(window_length,polyorder,deriv,delta,pos,true,coeffs)) return;

   if(mcheck=="interp")
     {
      if(window_length>N) return;
      convolve1d(x,coeffs,MODE_CONSTANT,0.0,y);
      int halflen=window_length/2;
      _fit_edge_1d(x,0,window_length,0,halflen,polyorder,deriv,delta,y);
      _fit_edge_1d(x,N-window_length,N,N-halflen,N,polyorder,deriv,delta,y);
     }
   else
     {
      int m=MODE_CONSTANT;
      if(mcheck=="mirror") m=MODE_MIRROR;
      else if(mcheck=="nearest") m=MODE_NEAREST;
      else if(mcheck=="wrap") m=MODE_WRAP;
      else if(mcheck=="constant") m=MODE_CONSTANT;
      convolve1d(x,coeffs,m,cval,y);
     }
  }

// Public cupy/scipy-like API (1D)
inline bool savgol_filter(const double &x[],int window_length,int polyorder,int deriv,double delta,
                          int axis,const string mode,double cval,double &y[])
  {
   if(axis!=0 && axis!=-1) return false;
   savgol_filter_1d(x,window_length,polyorder,deriv,delta,mode,cval,y);
   return (ArraySize(y)>0);
  }

// Public cupy/scipy-like API (2D, axis 0 or 1)
inline bool savgol_filter(const double &x[][],int window_length,int polyorder,int deriv,double delta,
                          int axis,const string mode,double cval,double &y[][])
  {
   int rows=ArrayRange(x,0);
   int cols=ArrayRange(x,1);
   if(rows<=0 || cols<=0) return false;
   if(axis==1 || axis==-1)
     {
      ArrayResize(y,rows,cols);
      for(int r=0;r<rows;r++)
        {
         double row[];
         ArrayResize(row,cols);
         for(int c=0;c<cols;c++) row[c]=x[r][c];
         double out[];
         savgol_filter_1d(row,window_length,polyorder,deriv,delta,mode,cval,out);
         if(ArraySize(out)!=cols) return false;
         for(int c=0;c<cols;c++) y[r][c]=out[c];
        }
      return true;
     }
   if(axis==0)
     {
      ArrayResize(y,rows,cols);
      for(int c=0;c<cols;c++)
        {
         double col[];
         ArrayResize(col,rows);
         for(int r=0;r<rows;r++) col[r]=x[r][c];
         double out[];
         savgol_filter_1d(col,window_length,polyorder,deriv,delta,mode,cval,out);
         if(ArraySize(out)!=rows) return false;
         for(int r=0;r<rows;r++) y[r][c]=out[r];
        }
      return true;
     }
   return false;
  }

// --- OpenCL GPU Savitzky-Golay (float64) ---
struct CLSavgolHandle
  {
   int ctx;
   int prog;
   int kern;
   int memX;
   int memH;
   int memY;
   int lenX;
   int lenH;
   bool ready;
  };

inline void CLSavgolReset(CLSavgolHandle &h)
  {
   h.ctx=INVALID_HANDLE;
   h.prog=INVALID_HANDLE;
   h.kern=INVALID_HANDLE;
   h.memX=INVALID_HANDLE;
   h.memH=INVALID_HANDLE;
   h.memY=INVALID_HANDLE;
   h.lenX=0;
   h.lenH=0;
   h.ready=false;
  }

inline void CLSavgolFree(CLSavgolHandle &h)
  {
   if(h.memX!=INVALID_HANDLE) { CLBufferFree(h.memX); h.memX=INVALID_HANDLE; }
   if(h.memH!=INVALID_HANDLE) { CLBufferFree(h.memH); h.memH=INVALID_HANDLE; }
   if(h.memY!=INVALID_HANDLE) { CLBufferFree(h.memY); h.memY=INVALID_HANDLE; }
   if(h.kern!=INVALID_HANDLE) { CLKernelFree(h.kern); h.kern=INVALID_HANDLE; }
   if(h.prog!=INVALID_HANDLE) { CLProgramFree(h.prog); h.prog=INVALID_HANDLE; }
   if(h.ctx!=INVALID_HANDLE) { CLContextFree(h.ctx); h.ctx=INVALID_HANDLE; }
   h.ready=false;
   h.lenX=0;
   h.lenH=0;
  }

inline bool CLSavgolInit(CLSavgolHandle &h)
  {
   if(h.ready) return true;
   CLSavgolReset(h);
   h.ctx=CLCreateContextGPUFloat64("SavitzkyGolay");
   if(h.ctx==INVALID_HANDLE) return false;

   string code=
   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
   "inline double bound_val(__global const double* x, int N, int idx, int mode, double cval){\n"
   "  if(N<=1){ return (N==1)? x[0] : cval; }\n"
   "  if(idx>=0 && idx<N) return x[idx];\n"
   "  if(mode==1) return cval;\n"
   "  if(mode==2) return (idx<0)? x[0] : x[N-1];\n"
   "  if(mode==3){ int j=idx%N; if(j<0) j+=N; return x[j]; }\n"
   "  int j=idx; while(j<0 || j>=N){ if(j<0) j=-j; if(j>=N) j=2*N-2-j; }\n"
   "  return x[j]; }\n"
   "__kernel void savgol_conv1d(__global const double* x, int N, __global const double* h, int M, int mode, double cval, __global double* y){\n"
   "  int i=get_global_id(0); if(i>=N) return; int center=M/2; double sum=0.0;\n"
   "  for(int k=0;k<M;k++){ int idx=i + k - center; double v=bound_val(x,N,idx,mode,cval); sum += v*h[k]; }\n"
   "  y[i]=sum; }\n";

   h.prog=CLProgramCreate(h.ctx,code);
   if(h.prog==INVALID_HANDLE) { CLSavgolFree(h); return false; }
   h.kern=CLKernelCreate(h.prog,"savgol_conv1d");
   if(h.kern==INVALID_HANDLE) { CLSavgolFree(h); return false; }
   h.ready=true;
   return true;
  }

inline bool savgol_filter_1d_gpu(const double &x[],int window_length,int polyorder,
                                 int deriv,double delta,const string mode,double cval,
                                 double &y[])
  {
   if(window_length<1) { ArrayResize(y,0); return false; }
   if(polyorder>=window_length) return false;
   int N=ArraySize(x);
   if(N<=0) { ArrayResize(y,0); return false; }

   double coeffs[];
   int pos=-1;
   if(!savgol_coeffs(window_length,polyorder,deriv,delta,pos,true,coeffs)) return false;

   string m=mode;
   StringToLower(m);
   if(m!="mirror" && m!="constant" && m!="nearest" && m!="interp" && m!="wrap")
     { ArrayResize(y,0); return false; }
   bool interp=(m=="interp");
   int mode_i=MODE_CONSTANT;
   double cval_use=cval;
   if(interp) { mode_i=MODE_CONSTANT; cval_use=0.0; }
   else if(m=="mirror") mode_i=MODE_MIRROR;
   else if(m=="nearest") mode_i=MODE_NEAREST;
   else if(m=="wrap") mode_i=MODE_WRAP;
   else mode_i=MODE_CONSTANT;

   static CLSavgolHandle h; if(!h.ready) CLSavgolReset(h);
   if(!CLSavgolInit(h)) return false;

   if(h.memX==INVALID_HANDLE || h.lenX!=N)
     {
      if(h.memX!=INVALID_HANDLE) CLBufferFree(h.memX);
      h.memX=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
      if(h.memX==INVALID_HANDLE) return false;
      h.lenX=N;
     }
   if(h.memH==INVALID_HANDLE || h.lenH!=window_length)
     {
      if(h.memH!=INVALID_HANDLE) CLBufferFree(h.memH);
      h.memH=CLBufferCreate(h.ctx,window_length*sizeof(double),CL_MEM_READ_ONLY);
      if(h.memH==INVALID_HANDLE) return false;
      h.lenH=window_length;
     }
   if(h.memY==INVALID_HANDLE || h.lenX!=N)
     {
      if(h.memY!=INVALID_HANDLE) CLBufferFree(h.memY);
      h.memY=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_WRITE);
      if(h.memY==INVALID_HANDLE) return false;
     }

   CLBufferWrite(h.memX,x);
   CLBufferWrite(h.memH,coeffs);
   CLSetKernelArgMem(h.kern,0,h.memX);
   CLSetKernelArg(h.kern,1,N);
   CLSetKernelArgMem(h.kern,2,h.memH);
   CLSetKernelArg(h.kern,3,window_length);
   CLSetKernelArg(h.kern,4,mode_i);
   CLSetKernelArg(h.kern,5,cval_use);
   CLSetKernelArgMem(h.kern,6,h.memY);
   uint offs[1]={0}; uint work[1]={(uint)N};
   if(!CLExecute(h.kern,1,offs,work)) return false;

   ArrayResize(y,N);
   CLBufferRead(h.memY,y);

   if(interp)
     {
      if(window_length>N) return false;
      int halflen=window_length/2;
      _fit_edge_1d(x,0,window_length,0,halflen,polyorder,deriv,delta,y);
      _fit_edge_1d(x,N-window_length,N,N-halflen,N,polyorder,deriv,delta,y);
     }
   return true;
  }

#endif
