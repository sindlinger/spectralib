#ifndef __SPECTRAL_SIGNALTOOLS_MQH__
#define __SPECTRAL_SIGNALTOOLS_MQH__

#include "SpectralCommon.mqh"

enum DetrendType
  {
   DETREND_NONE=0,
   DETREND_CONSTANT=1,
   DETREND_LINEAR=2
  };

// Detrend each row (segment) of a 2D array [segments][nperseg]
inline void detrend_segments(double &seg[][],const int detrend_type)
  {
   if(detrend_type==DETREND_NONE) return;
   int nseg=ArrayRange(seg,0);
   if(nseg<=0) return;
   int nper=ArrayRange(seg,1);
   if(nper<=0) return;

   if(detrend_type==DETREND_CONSTANT)
     {
      for(int s=0;s<nseg;s++)
        {
         double sum=0.0;
         for(int i=0;i<nper;i++) sum+=seg[s][i];
         double mean=sum/nper;
         for(int i=0;i<nper;i++) seg[s][i]-=mean;
        }
      return;
     }

   if(detrend_type==DETREND_LINEAR)
     {
      // linear regression against sample index
      double N=(double)nper;
      double sumX=(N-1.0)*N*0.5;           // sum i
      double sumXX=(N-1.0)*N*(2.0*N-1.0)/6.0; // sum i^2
      double denom=N*sumXX - sumX*sumX;
      if(MathAbs(denom)<1e-12) return;
      for(int s=0;s<nseg;s++)
        {
         double sumY=0.0;
         double sumXY=0.0;
         for(int i=0;i<nper;i++)
           {
            double y=seg[s][i];
            sumY+=y;
            sumXY+=y*(double)i;
           }
         double slope=(N*sumXY - sumX*sumY)/denom;
         double intercept=(sumY - slope*sumX)/N;
         for(int i=0;i<nper;i++)
            seg[s][i]-=(intercept + slope*(double)i);
        }
     }
  }

#endif
