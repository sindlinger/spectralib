#ifndef __SPECTRAL_ARRAYTOOLS_MQH__
#define __SPECTRAL_ARRAYTOOLS_MQH__

#include "SpectralCommon.mqh"

// NOTE: MQL5 does not support true N-D array views. The helpers below
// implement 1D extensions/slicing equivalent to SciPy/cusignal behavior.

inline void odd_ext(const double &x[],int n,double &out[])
  {
   int N=ArraySize(x);
   if(n<1)
     {
      ArrayResize(out,N);
      ArrayCopy(out,x,0,0,N);
      return;
     }
   if(n>N-1) n=N-1;
   int outN=N+2*n;
   ArrayResize(out,outN);
   // left: 2*x0 - x[n:0:-1]
   for(int i=0;i<n;i++)
     {
      double v=x[n-i];
      out[i]=2.0*x[0]-v;
     }
   // middle
   for(int i=0;i<N;i++) out[n+i]=x[i];
   // right: 2*x[-1] - x[-2:-n-2:-1]
   for(int i=0;i<n;i++)
     {
      double v=x[N-2-i];
      out[n+N+i]=2.0*x[N-1]-v;
     }
  }

inline void even_ext(const double &x[],int n,double &out[])
  {
   int N=ArraySize(x);
   if(n<1)
     {
      ArrayResize(out,N);
      ArrayCopy(out,x,0,0,N);
      return;
     }
   if(n>N-1) n=N-1;
   int outN=N+2*n;
   ArrayResize(out,outN);
   // left: x[n:0:-1]
   for(int i=0;i<n;i++)
      out[i]=x[n-i];
   // middle
   for(int i=0;i<N;i++) out[n+i]=x[i];
   // right: x[-2:-n-2:-1]
   for(int i=0;i<n;i++)
      out[n+N+i]=x[N-2-i];
  }

inline void const_ext(const double &x[],int n,double &out[])
  {
   int N=ArraySize(x);
   if(n<1)
     {
      ArrayResize(out,N);
      ArrayCopy(out,x,0,0,N);
      return;
     }
   int outN=N+2*n;
   ArrayResize(out,outN);
   for(int i=0;i<n;i++) out[i]=x[0];
   for(int i=0;i<N;i++) out[n+i]=x[i];
   for(int i=0;i<n;i++) out[n+N+i]=x[N-1];
  }

inline void zero_ext(const double &x[],int n,double &out[])
  {
   int N=ArraySize(x);
   if(n<1)
     {
      ArrayResize(out,N);
      ArrayCopy(out,x,0,0,N);
      return;
     }
   int outN=N+2*n;
   ArrayResize(out,outN);
   for(int i=0;i<n;i++) out[i]=0.0;
   for(int i=0;i<N;i++) out[n+i]=x[i];
   for(int i=0;i<n;i++) out[n+N+i]=0.0;
  }

// axis_slice equivalent for 1D arrays:
// returns x[start:stop:step] into out
inline void axis_slice_1d(const double &x[],int start,int stop,int step,double &out[])
  {
   int N=ArraySize(x);
   if(step==0) step=1;
   if(start<0) start+=N;
   if(stop<0) stop+=N;
   if(start<0) start=0;
   if(stop>N) stop=N;
   int count=0;
   for(int i=start;(step>0? i<stop : i>stop);i+=step) count++;
   ArrayResize(out,count);
   int k=0;
   for(int i=start;(step>0? i<stop : i>stop);i+=step)
     { out[k++]=x[i]; }
  }

#endif
