#ifndef __SPECTRAL_LINALG_MQH__
#define __SPECTRAL_LINALG_MQH__

#include "SpectralCommon.mqh"

inline bool MatInvert(const double &A[],double &Ainv[],int n)
  {
   if(n<=0) return false;
   double tmp[];
   ArrayResize(tmp,n*n*2);
   for(int r=0;r<n;r++)
     {
      for(int c=0;c<n;c++)
        {
         tmp[r*(2*n)+c]=A[r*n+c];
         tmp[r*(2*n)+n+c]=(r==c)?1.0:0.0;
        }
     }
   for(int i=0;i<n;i++)
     {
      double pivot=tmp[i*(2*n)+i];
      int piv=i;
      for(int r=i+1;r<n;r++)
        {
         double v=MathAbs(tmp[r*(2*n)+i]);
         if(v>MathAbs(pivot))
           { pivot=tmp[r*(2*n)+i]; piv=r; }
        }
      if(MathAbs(pivot)<1e-15) return false;
      if(piv!=i)
        {
         for(int c=0;c<2*n;c++)
           {
            double t=tmp[i*(2*n)+c];
            tmp[i*(2*n)+c]=tmp[piv*(2*n)+c];
            tmp[piv*(2*n)+c]=t;
           }
        }
      double invp=1.0/tmp[i*(2*n)+i];
      for(int c=0;c<2*n;c++) tmp[i*(2*n)+c]*=invp;
      for(int r=0;r<n;r++)
        {
         if(r==i) continue;
         double f=tmp[r*(2*n)+i];
         if(MathAbs(f)<1e-15) continue;
         for(int c=0;c<2*n;c++)
            tmp[r*(2*n)+c]-=f*tmp[i*(2*n)+c];
        }
     }
   ArrayResize(Ainv,n*n);
   for(int r=0;r<n;r++)
      for(int c=0;c<n;c++)
         Ainv[r*n+c]=tmp[r*(2*n)+n+c];
   return true;
  }

#endif
