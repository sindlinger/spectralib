#ifndef __SAVGOL_MQH__
#define __SAVGOL_MQH__

#include "SpectralCommon.mqh"
#include "SpectralArrayTools.mqh"
#include "SpectralLinalg.mqh"

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
      double next[];
      ArrayResize(next,curN-1);
      for(int i=0;i<curN-1;i++)
        {
         int power=(curN-1-i);
         next[i]=cur[i]*power;
        }
      cur=next;
      curN--;
     }
   ArrayResize(out,curN);
   ArrayCopy(out,cur);
  }

// Least squares solution for A*c = y (A: m x n)
inline bool _lstsq(const double &A[],int m,int n,const double &y[],double &c[])
  {
   // Use normal equations: (A^T A) c = A^T y
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
   // y vector
   double y[];
   ArrayResize(y,m);
   for(int j=0;j<m;j++) y[j]=0.0;
   if(deriv>polyorder)
     {
      ArrayResize(coeffs,window_length);
      ArrayInitialize(coeffs,0.0);
      return true;
     }
   y[deriv]=float_factorial(deriv)/MathPow(delta,deriv);

   // solve A*c = y (least squares)
   // c length = m; we need filter coeffs length window_length: coeffs = A * c
   double cvec[];
   if(!_lstsq(A,window_length,m,y,cvec)) return false;
   ArrayResize(coeffs,window_length);
   for(int i=0;i<window_length;i++)
     {
      double sum=0.0;
      for(int j=0;j<m;j++) sum+=A[i*m+j]*cvec[j];
      coeffs[i]=sum;
     }
   return true;
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
   int N=ArraySize(x);
   // coeffs
   double coeffs[];
   int pos=-1;
   if(!savgol_coeffs(window_length,polyorder,deriv,delta,pos,true,coeffs)) return;

   if(mode=="interp")
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
      if(mode=="mirror") m=MODE_MIRROR;
      else if(mode=="nearest") m=MODE_NEAREST;
      else if(mode=="wrap") m=MODE_WRAP;
      else if(mode=="constant") m=MODE_CONSTANT;
      convolve1d(x,coeffs,m,cval,y);
     }
  }

#endif
