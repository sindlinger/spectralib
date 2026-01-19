#ifndef __SPECTRAL_WINDOWS_MQH__
#define __SPECTRAL_WINDOWS_MQH__

#include "SpectralCommon.mqh"
#include "SpectralFFT.mqh"

inline bool _len_guards(int M)
  {
   if(M<0) return true;
   return (M<=1);
  }

inline void _extend(int M,const bool sym,int &Mout,bool &needs_trunc)
  {
   if(!sym)
     { Mout=M+1; needs_trunc=true; }
   else
     { Mout=M; needs_trunc=false; }
  }

inline void _truncate(const double &w[],const bool needed,double &out[])
  {
   if(!needed)
     {
      ArrayResize(out,ArraySize(w));
      ArrayCopy(out,w);
      return;
     }
   int N=ArraySize(w);
   if(N<=1) { ArrayResize(out,N); ArrayCopy(out,w); return; }
   ArrayResize(out,N-1);
   for(int i=0;i<N-1;i++) out[i]=w[i];
  }

inline void general_cosine(int M,const double &a[],const bool sym,double &out[])
  {
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double w[];
   ArrayResize(w,Mx);
   double delta=2.0*PI/(double)(Mx-1);
   for(int i=0;i<Mx;i++)
     {
      double fac=-PI + delta*i;
      double temp=0.0;
      int n=ArraySize(a);
      for(int k=0;k<n;k++) temp+=a[k]*MathCos((double)k*fac);
      w[i]=temp;
     }
   _truncate(w,trunc,out);
  }

inline void boxcar(int M,const bool sym,double &out[])
  {
   if(M<=0) { ArrayResize(out,0); return; }
   ArrayResize(out,M);
   for(int i=0;i<M;i++) out[i]=1.0;
  }

inline void triang(int M,const bool sym,double &out[])
  {
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double w[];
   ArrayResize(w,Mx);
   double N=(double)Mx;
   for(int n=0;n<Mx;n++)
     {
      double val=1.0 - MathAbs((n - (N-1.0)/2.0)/((N+1.0)/2.0));
      w[n]=val;
     }
   _truncate(w,trunc,out);
  }

inline void parzen(int M,const bool sym,double &out[])
  {
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double w[];
   ArrayResize(w,Mx);
   double N=(double)Mx;
   double half=(N-1.0)/2.0;
   for(int n=0;n<Mx;n++)
     {
      double x=MathAbs((n-half)/(half+1.0));
      double v;
      if(x<=0.5)
         v=1.0 - 6.0*x*x + 6.0*x*x*x;
      else if(x<=1.0)
         v=2.0*MathPow(1.0-x,3.0);
      else v=0.0;
      w[n]=v;
     }
   _truncate(w,trunc,out);
  }

inline void bohman(int M,const bool sym,double &out[])
  {
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double w[];
   ArrayResize(w,Mx);
   double N=(double)Mx;
   double half=(N-1.0)/2.0;
   for(int n=0;n<Mx;n++)
     {
      double x=MathAbs((n-half)/half);
      double v=(1.0-x)*MathCos(PI*x) + (1.0/PI)*MathSin(PI*x);
      w[n]=v;
     }
   _truncate(w,trunc,out);
  }

inline void blackman(int M,const bool sym,double &out[])
  {
   double a[]={0.42,0.5,0.08};
   // w = a0 - a1 cos + a2 cos2
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double w[];
   ArrayResize(w,Mx);
   double N=(double)Mx;
   for(int n=0;n<Mx;n++)
     {
      double ang=2.0*PI*n/(N-1.0);
      w[n]=a[0]-a[1]*MathCos(ang)+a[2]*MathCos(2.0*ang);
     }
   _truncate(w,trunc,out);
  }

inline void nuttall(int M,const bool sym,double &out[])
  {
   // 4-term Nuttall
   double a0=0.355768;
   double a1=0.487396;
   double a2=0.144232;
   double a3=0.012604;
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double w[];
   ArrayResize(w,Mx);
   double N=(double)Mx;
   for(int n=0;n<Mx;n++)
     {
      double ang=2.0*PI*n/(N-1.0);
      w[n]=a0 - a1*MathCos(ang) + a2*MathCos(2.0*ang) - a3*MathCos(3.0*ang);
     }
   _truncate(w,trunc,out);
  }

inline void blackmanharris(int M,const bool sym,double &out[])
  {
   double a0=0.35875;
   double a1=0.48829;
   double a2=0.14128;
   double a3=0.01168;
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double w[];
   ArrayResize(w,Mx);
   double N=(double)Mx;
   for(int n=0;n<Mx;n++)
     {
      double ang=2.0*PI*n/(N-1.0);
      w[n]=a0 - a1*MathCos(ang) + a2*MathCos(2.0*ang) - a3*MathCos(3.0*ang);
     }
   _truncate(w,trunc,out);
  }

inline void flattop(int M,const bool sym,double &out[])
  {
   double a0=1.0;
   double a1=1.93;
   double a2=1.29;
   double a3=0.388;
   double a4=0.0322;
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double w[];
   ArrayResize(w,Mx);
   double N=(double)Mx;
   for(int n=0;n<Mx;n++)
     {
      double ang=2.0*PI*n/(N-1.0);
      w[n]=a0 - a1*MathCos(ang) + a2*MathCos(2.0*ang) - a3*MathCos(3.0*ang) + a4*MathCos(4.0*ang);
     }
   _truncate(w,trunc,out);
  }

inline void bartlett(int M,const bool sym,double &out[])
  {
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double w[];
   ArrayResize(w,Mx);
   double N=(double)Mx;
   double half=(N-1.0)/2.0;
   for(int n=0;n<Mx;n++)
      w[n]=1.0 - MathAbs((n-half)/half);
   _truncate(w,trunc,out);
  }

inline void hann(int M,const bool sym,double &out[])
  {
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double w[];
   ArrayResize(w,Mx);
   double N=(double)Mx;
   for(int n=0;n<Mx;n++)
     {
      double ang=2.0*PI*n/(N-1.0);
      w[n]=0.5 - 0.5*MathCos(ang);
     }
   _truncate(w,trunc,out);
  }

inline void tukey(int M,double alpha,const bool sym,double &out[])
  {
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double w[];
   ArrayResize(w,Mx);
   if(alpha<=0.0)
     {
      for(int i=0;i<Mx;i++) w[i]=1.0;
     }
   else if(alpha>=1.0)
     {
      // Hann
      double N=(double)Mx;
      for(int n=0;n<Mx;n++)
        { double ang=2.0*PI*n/(N-1.0); w[n]=0.5-0.5*MathCos(ang); }
     }
   else
     {
      double N=(double)Mx;
      double edge=alpha*(N-1.0)/2.0;
      for(int n=0;n<Mx;n++)
        {
         if(n<edge)
           {
            double ang=PI*(2.0*n/alpha/(N-1.0)-1.0);
            w[n]=0.5*(1.0+MathCos(ang));
           }
         else if(n<= (N-1.0)*(1.0-alpha/2.0))
            w[n]=1.0;
         else
           {
            double ang=PI*(2.0*n/alpha/(N-1.0)-2.0/alpha+1.0);
            w[n]=0.5*(1.0+MathCos(ang));
           }
        }
     }
   _truncate(w,trunc,out);
  }

inline void barthann(int M,const bool sym,double &out[])
  {
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double w[];
   ArrayResize(w,Mx);
   double N=(double)Mx;
   double half=(N-1.0)/2.0;
   for(int n=0;n<Mx;n++)
     {
      double x=MathAbs((n-half)/half);
      w[n]=0.62 - 0.48*x + 0.38*MathCos(PI*x);
     }
   _truncate(w,trunc,out);
  }

inline void general_hamming(int M,double alpha,const bool sym,double &out[])
  {
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double w[];
   ArrayResize(w,Mx);
   double N=(double)Mx;
   for(int n=0;n<Mx;n++)
     {
      double ang=2.0*PI*n/(N-1.0);
      w[n]=alpha - (1.0-alpha)*MathCos(ang);
     }
   _truncate(w,trunc,out);
  }

inline void hamming(int M,const bool sym,double &out[])
  { general_hamming(M,0.54,sym,out); }

inline double _bessel_i0(double x)
  {
   // Approximation of I0 from Numerical Recipes
   double ax=MathAbs(x);
   double y;
   if(ax<3.75)
     {
      y=x/3.75; y*=y;
      return 1.0 + y*(3.5156229 + y*(3.0899424 + y*(1.2067492
             + y*(0.2659732 + y*(0.360768e-1 + y*0.45813e-2)))));
     }
   y=3.75/ax;
   return (MathExp(ax)/MathSqrt(ax))*(0.39894228 + y*(0.1328592e-1
          + y*(0.225319e-2 + y*(-0.157565e-2 + y*(0.916281e-2
          + y*(-0.2057706e-1 + y*(0.2635537e-1 + y*(-0.1647633e-1
          + y*0.392377e-2))))))));
  }

inline void kaiser(int M,double beta,const bool sym,double &out[])
  {
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double w[];
   ArrayResize(w,Mx);
   double denom=_bessel_i0(beta);
   double N=(double)Mx;
   for(int n=0;n<Mx;n++)
     {
      double r=2.0*n/(N-1.0)-1.0;
      double val=_bessel_i0(beta*MathSqrt(1.0-r*r))/denom;
      w[n]=val;
     }
   _truncate(w,trunc,out);
  }

inline void gaussian(int M,double std,const bool sym,double &out[])
  {
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double w[];
   ArrayResize(w,Mx);
   double N=(double)Mx;
   double mid=(N-1.0)/2.0;
   for(int n=0;n<Mx;n++)
     {
      double x=(n-mid)/std;
      w[n]=MathExp(-0.5*x*x);
     }
   _truncate(w,trunc,out);
  }

inline void general_gaussian(int M,double p,double sig,const bool sym,double &out[])
  {
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double w[];
   ArrayResize(w,Mx);
   double N=(double)Mx;
   double mid=(N-1.0)/2.0;
   for(int n=0;n<Mx;n++)
     {
      double x=MathAbs((n-mid)/sig);
      w[n]=MathExp(-0.5*MathPow(x,2.0*p));
     }
   _truncate(w,trunc,out);
  }

inline double _acosh(double x)
  { return MathLog(x + MathSqrt(x*x-1.0)); }

inline void chebwin(int M,double at,const bool sym,double &out[])
  {
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double order=(double)Mx-1.0;
   double beta=MathCosh((1.0/order)*_acosh(MathPow(10.0,MathAbs(at)/20.0)));

   // build frequency response p
   Complex64 p[];
   ArrayResize(p,Mx);
   double N=PI*(1.0/(double)Mx);
   bool odd=(Mx & 1)==1;
   for(int i=0;i<Mx;i++)
     {
      double x=beta*MathCos(i*N);
      double realv;
      if(x>1.0) realv=MathCosh(order*_acosh(x));
      else if(x<-1.0) realv=(odd?1.0:-1.0)*MathCosh(order*_acosh(-x));
      else realv=MathCos(order*MathArccos(x));
      if(odd)
         p[i]=Cx(realv,0.0);
      else
        {
         double ang=N*i;
         p[i]=Cx(realv*MathCos(ang),realv*MathSin(ang));
        }
     }

   // FFT of p
   Complex64 P[];
   DFT(p,P,false);
   // real part
   double wtmp[];
   ArrayResize(wtmp,Mx);
   for(int i=0;i<Mx;i++) wtmp[i]=P[i].re;

   // arrange symmetric
   double w[];
   if((Mx & 1)==1)
     {
      int n=(Mx+1)/2;
      ArrayResize(w,Mx);
      // w = concat(w[n-1:0:-1], w[:n])
      int idx=0;
      for(int i=n-1;i>0;i--) w[idx++]=wtmp[i];
      for(int i=0;i<n;i++) w[idx++]=wtmp[i];
     }
   else
     {
      int n=Mx/2+1;
      ArrayResize(w,Mx);
      int idx=0;
      for(int i=n-1;i>0;i--) w[idx++]=wtmp[i];
      for(int i=1;i<n;i++) w[idx++]=wtmp[i];
     }
   // normalize
   double maxv=0.0;
   for(int i=0;i<ArraySize(w);i++) if(MathAbs(w[i])>maxv) maxv=MathAbs(w[i]);
   if(maxv>0.0) for(int i=0;i<ArraySize(w);i++) w[i]/=maxv;
   _truncate(w,trunc,out);
  }

inline void cosine(int M,const bool sym,double &out[])
  {
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double w[];
   ArrayResize(w,Mx);
   for(int i=0;i<Mx;i++)
      w[i]=MathSin(PI/((double)Mx)*(i+0.5));
   _truncate(w,trunc,out);
  }

inline void exponential(int M,double center,double tau,const bool sym,double &out[])
  {
   if(sym && center>=0.0) { /* center must be default */ }
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);
   double w[];
   ArrayResize(w,Mx);
   if(center<0.0) center=(Mx-1.0)/2.0;
   for(int i=0;i<Mx;i++)
      w[i]=MathExp(-MathAbs(i-center)/tau);
   _truncate(w,trunc,out);
  }

inline void taylor(int M,int nbar,double sll,const bool norm,const bool sym,double &out[])
  {
   if(_len_guards(M)) { ArrayResize(out,M); for(int i=0;i<M;i++) out[i]=1.0; return; }
   int Mx; bool trunc;
   _extend(M,sym,Mx,trunc);

   double B=MathPow(10.0,sll/20.0);
   double A=_acosh(B)/PI;
   double s2=(double)nbar*(double)nbar/(A*A + (nbar-0.5)*(nbar-0.5));
   int mcount=nbar-1;
   if(mcount<1) { ArrayResize(out,Mx); for(int i=0;i<Mx;i++) out[i]=1.0; _truncate(out,trunc,out); return; }

   double Fm[];
   ArrayResize(Fm,mcount);
   for(int mi=0;mi<mcount;mi++)
     {
      double m=mi+1;
      double numer_sign=(mi%2==0)?1.0:-1.0;
      // product terms
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

   // compute window
   double w[];
   ArrayResize(w,Mx);
   double mod_pi=2.0*PI/(double)Mx;
   for(int i=0;i<Mx;i++)
     {
      double temp=mod_pi*(i - (double)Mx/2.0 + 0.5);
      double dot=0.0;
      for(int k=1;k<nbar;k++) dot+=Fm[k-1]*MathCos(temp*k);
      double val=1.0 + 2.0*dot;
      if(norm)
        {
         double temp2=mod_pi*(((Mx-1.0)/2.0) - (double)Mx/2.0 + 0.5);
         double dot2=0.0;
         for(int k=1;k<nbar;k++) dot2+=Fm[k-1]*MathCos(temp2*k);
         double scale=1.0/(1.0+2.0*dot2);
         val*=scale;
        }
      w[i]=val;
     }
   _truncate(w,trunc,out);
  }

inline void get_window_params(const string win,const int Nx,const bool fftbins,const double &params[],double &out[])
  {
   string name=win;
   StringToLower(name);
   if(name=="boxcar" || name=="box" || name=="ones" || name=="rect" || name=="rectangular")
     { boxcar(Nx,!fftbins,out); return; }
   if(name=="triang" || name=="triangle" || name=="tri")
     { triang(Nx,!fftbins,out); return; }
   if(name=="parzen" || name=="parz" || name=="par")
     { parzen(Nx,!fftbins,out); return; }
   if(name=="bohman" || name=="bman" || name=="bmn")
     { bohman(Nx,!fftbins,out); return; }
   if(name=="blackman" || name=="black" || name=="blk")
     { blackman(Nx,!fftbins,out); return; }
   if(name=="blackmanharris" || name=="blackharr" || name=="bkh")
     { blackmanharris(Nx,!fftbins,out); return; }
   if(name=="nuttall" || name=="nutl" || name=="nut")
     { nuttall(Nx,!fftbins,out); return; }
   if(name=="flattop" || name=="flat" || name=="flt")
     { flattop(Nx,!fftbins,out); return; }
   if(name=="bartlett" || name=="bart" || name=="brt")
     { bartlett(Nx,!fftbins,out); return; }
   if(name=="hann" || name=="hanning" || name=="han")
     { hann(Nx,!fftbins,out); return; }
   if(name=="hamming" || name=="hamm" || name=="ham")
     { hamming(Nx,!fftbins,out); return; }
   if(name=="barthann" || name=="brthan" || name=="bth")
     { barthann(Nx,!fftbins,out); return; }
   if(name=="cosine" || name=="halfcosine")
     { cosine(Nx,!fftbins,out); return; }
   // parameterized windows: use params array (matches tuple usage in Python)
   if(name=="tukey" || name=="tuk")
     {
      double alpha=(ArraySize(params)>0?params[0]:0.5);
      tukey(Nx,alpha,!fftbins,out); return;
     }
   if(name=="kaiser" || name=="ksr")
     {
      double beta=(ArraySize(params)>0?params[0]:0.0);
      kaiser(Nx,beta,!fftbins,out); return;
     }
   if(name=="gaussian" || name=="gauss" || name=="gss")
     {
      double std=(ArraySize(params)>0?params[0]:1.0);
      gaussian(Nx,std,!fftbins,out); return;
     }
   if(name=="general_gaussian" || name=="general gaussian" || name=="general gauss" || name=="general_gauss" || name=="ggs")
     {
      double p=(ArraySize(params)>0?params[0]:1.0);
      double sig=(ArraySize(params)>1?params[1]:1.0);
      general_gaussian(Nx,p,sig,!fftbins,out); return;
     }
   if(name=="chebwin" || name=="cheb")
     {
      double at=(ArraySize(params)>0?params[0]:100.0);
      chebwin(Nx,at,!fftbins,out); return;
     }
   if(name=="exponential" || name=="poisson")
     {
      double tau=(ArraySize(params)>0?params[0]:1.0);
      double center=(ArraySize(params)>1?params[1]:-1.0);
      exponential(Nx,center,tau,!fftbins,out); return;
     }
   if(name=="taylor")
     {
      int nbar=(ArraySize(params)>0?(int)params[0]:4);
      double sll=(ArraySize(params)>1?params[1]:30.0);
      bool norm=true;
      if(ArraySize(params)>2) norm=(params[2]!=0.0);
      taylor(Nx,nbar,sll,norm,!fftbins,out); return;
     }

   // default fallback to hann
   hann(Nx,!fftbins,out);
  }

inline void get_window(const string win,const int Nx,const bool fftbins,double &out[])
  {
   double params[];
   ArrayResize(params,0);
   get_window_params(win,Nx,fftbins,params,out);
  }


#endif
