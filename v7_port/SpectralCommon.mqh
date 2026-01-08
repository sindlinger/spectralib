#ifndef __SPECTRAL_COMMON_MQH__
#define __SPECTRAL_COMMON_MQH__

// Common math utilities and complex helpers for the spectral port (float64).

struct Complex64
  {
   double re;
   double im;
  };

inline Complex64 Cx(double re,double im)
  {
   Complex64 c; c.re=re; c.im=im; return c;
  }

inline Complex64 CxAdd(const Complex64 &a,const Complex64 &b)
  { return Cx(a.re+b.re,a.im+b.im); }

inline Complex64 CxSub(const Complex64 &a,const Complex64 &b)
  { return Cx(a.re-b.re,a.im-b.im); }

inline Complex64 CxMul(const Complex64 &a,const Complex64 &b)
  { return Cx(a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re); }

inline Complex64 CxScale(const Complex64 &a,double s)
  { return Cx(a.re*s,a.im*s); }

inline Complex64 CxConj(const Complex64 &a)
  { return Cx(a.re,-a.im); }

inline double CxAbs(const Complex64 &a)
  { return MathSqrt(a.re*a.re + a.im*a.im); }

inline double CxArg(const Complex64 &a)
  { return MathArctan2(a.im,a.re); }

inline bool IsPowerOfTwo(const int n)
  {
   if(n<=0) return false;
   return (n & (n-1))==0;
  }

inline int NextPow2(int n)
  {
   int p=1;
   while(p<n && p>0) p<<=1;
   return p;
  }

inline double ClampD(const double v,const double lo,const double hi)
  {
   if(v<lo) return lo;
   if(v>hi) return hi;
   return v;
  }

inline void Arange(const int n,double &out[])
  {
   ArrayResize(out,n);
   for(int i=0;i<n;i++) out[i]=(double)i;
  }

inline void Linspace(const double start,const double stop,const int n,double &out[])
  {
   ArrayResize(out,n);
   if(n<=1) { if(n==1) out[0]=start; return; }
   double step=(stop-start)/((double)(n-1));
   for(int i=0;i<n;i++) out[i]=start+step*i;
  }

#endif
