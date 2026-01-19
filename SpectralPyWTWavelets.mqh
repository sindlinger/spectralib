#ifndef __SPECTRAL_PYWT_WAVELETS_MQH__

#define __SPECTRAL_PYWT_WAVELETS_MQH__

#include "SpectralCommon.mqh"
#include "SpectralOpenCLCommon.mqh"
#include "SpectralPyWTCommon.mqh"
#include "SpectralPyWTCoeffs.mqh"


struct PyWTDiscreteWavelet
  {
   int support_width;
   int symmetry;
   bool orthogonal;
   bool biorthogonal;
   bool compact_support;
   string family_name;
   string short_name;
   double dec_hi[];
   double dec_lo[];
   double rec_hi[];
   double rec_lo[];
   int dec_len;
   int rec_len;
   int vanishing_moments_psi;
   int vanishing_moments_phi;
  };


inline void PyWT_ResetWavelet(PyWTDiscreteWavelet &w)
  {
   w.support_width=0; w.symmetry=PYWT_UNKNOWN; w.orthogonal=false; w.biorthogonal=false; w.compact_support=false;
   w.family_name=""; w.short_name="";
   ArrayResize(w.dec_hi,0); ArrayResize(w.dec_lo,0); ArrayResize(w.rec_hi,0); ArrayResize(w.rec_lo,0);
   w.dec_len=0; w.rec_len=0; w.vanishing_moments_psi=0; w.vanishing_moments_phi=0;
  }


inline void PyWT_CopyArray(const double &src[],double &dst[])
  {
   int n=ArraySize(src);
   ArrayResize(dst,n);
   for(int i=0;i<n;i++) dst[i]=src[i];
  }


inline void PyWT_ReverseArray(double &arr[])
  {
   int n=ArraySize(arr);
   for(int i=0,j=n-1;i<j;i++,j--) { double tmp=arr[i]; arr[i]=arr[j]; arr[j]=tmp; }
  }

// GPU builders for wavelet filters (float64)
struct PyWTWaveletHandle
  {
   int ctx;
   int prog;
   int kern_build_orth;
   int kern_build_bior;
   int kern_swap_reverse;
   int memIn1;
   int memIn2;
   int memIn3;
   int memIn4;
   int memOut1;
   int memOut2;
   int memOut3;
   int memOut4;
   int lenIn1;
   int lenIn2;
   int lenIn3;
   int lenIn4;
   int lenOut;
   bool ready;
  };

inline void PyWTWaveletReset(PyWTWaveletHandle &h)
  {
   h.ctx=INVALID_HANDLE; h.prog=INVALID_HANDLE;
   h.kern_build_orth=INVALID_HANDLE; h.kern_build_bior=INVALID_HANDLE; h.kern_swap_reverse=INVALID_HANDLE;
   h.memIn1=INVALID_HANDLE; h.memIn2=INVALID_HANDLE; h.memIn3=INVALID_HANDLE; h.memIn4=INVALID_HANDLE;
   h.memOut1=INVALID_HANDLE; h.memOut2=INVALID_HANDLE; h.memOut3=INVALID_HANDLE; h.memOut4=INVALID_HANDLE;
   h.lenIn1=0; h.lenIn2=0; h.lenIn3=0; h.lenIn4=0; h.lenOut=0;
   h.ready=false;
  }

inline void PyWTWaveletFree(PyWTWaveletHandle &h)
  {
   if(h.memIn1!=INVALID_HANDLE){ CLBufferFree(h.memIn1); h.memIn1=INVALID_HANDLE; }
   if(h.memIn2!=INVALID_HANDLE){ CLBufferFree(h.memIn2); h.memIn2=INVALID_HANDLE; }
   if(h.memIn3!=INVALID_HANDLE){ CLBufferFree(h.memIn3); h.memIn3=INVALID_HANDLE; }
   if(h.memIn4!=INVALID_HANDLE){ CLBufferFree(h.memIn4); h.memIn4=INVALID_HANDLE; }
   if(h.memOut1!=INVALID_HANDLE){ CLBufferFree(h.memOut1); h.memOut1=INVALID_HANDLE; }
   if(h.memOut2!=INVALID_HANDLE){ CLBufferFree(h.memOut2); h.memOut2=INVALID_HANDLE; }
   if(h.memOut3!=INVALID_HANDLE){ CLBufferFree(h.memOut3); h.memOut3=INVALID_HANDLE; }
   if(h.memOut4!=INVALID_HANDLE){ CLBufferFree(h.memOut4); h.memOut4=INVALID_HANDLE; }
   if(h.kern_build_orth!=INVALID_HANDLE){ CLKernelFree(h.kern_build_orth); h.kern_build_orth=INVALID_HANDLE; }
   if(h.kern_build_bior!=INVALID_HANDLE){ CLKernelFree(h.kern_build_bior); h.kern_build_bior=INVALID_HANDLE; }
   if(h.kern_swap_reverse!=INVALID_HANDLE){ CLKernelFree(h.kern_swap_reverse); h.kern_swap_reverse=INVALID_HANDLE; }
   if(h.prog!=INVALID_HANDLE){ CLProgramFree(h.prog); h.prog=INVALID_HANDLE; }
   if(h.ctx!=INVALID_HANDLE){ CLContextFree(h.ctx); h.ctx=INVALID_HANDLE; }
   h.lenIn1=0; h.lenIn2=0; h.lenIn3=0; h.lenIn4=0; h.lenOut=0;
   h.ready=false;
  }

inline bool PyWTWaveletInit(PyWTWaveletHandle &h)
  {
   if(h.ready) return true;
   PyWTWaveletReset(h);
   h.ctx=CLCreateContextGPUFloat64("SpectralPyWTWavelets");
   if(h.ctx==INVALID_HANDLE) return false;
   string code=
   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
   "__kernel void build_orth(__global const double* lo, int n, double scale, __global double* rec_lo, __global double* dec_lo, __global double* rec_hi, __global double* dec_hi){\n"
   "  int i=get_global_id(0); if(i>=n) return; double v=lo[i]*scale; int j=n-1-i; double vr=lo[j]*scale;\n"
   "  rec_lo[i]=v; dec_lo[i]=vr; double s=(i&1)?-1.0:1.0; rec_hi[i]=s*vr; double s2=(j&1)?-1.0:1.0; dec_hi[i]=s2*v; }\n"
   "__kernel void build_bior(__global const double* rec_src, __global const double* dec_src, int rec_len, int dec_len, int shift,\n"
   "  __global double* rec_lo, __global double* dec_lo, __global double* rec_hi, __global double* dec_hi){\n"
   "  int i=get_global_id(0); if(i>=rec_len) return; double r=rec_src[i+shift]; double d=dec_src[dec_len-1-i];\n"
   "  rec_lo[i]=r; dec_lo[i]=d; double s=(i&1)?-1.0:1.0; rec_hi[i]=s*d; int j=dec_len-1-i; double s2=(j&1)?-1.0:1.0; dec_hi[i]=s2*r; }\n"
   "__kernel void swap_reverse(__global const double* rec_lo, __global const double* rec_hi, __global const double* dec_lo, __global const double* dec_hi, int n,\n"
   "  __global double* out_rec_lo, __global double* out_rec_hi, __global double* out_dec_lo, __global double* out_dec_hi){\n"
   "  int i=get_global_id(0); if(i>=n) return; int j=n-1-i; out_rec_lo[i]=dec_lo[j]; out_rec_hi[i]=dec_hi[j]; out_dec_lo[i]=rec_lo[j]; out_dec_hi[i]=rec_hi[j]; }\n";
   h.prog=CLProgramCreate(h.ctx,code);
   if(h.prog==INVALID_HANDLE){ PyWTWaveletFree(h); return false; }
   h.kern_build_orth=CLKernelCreate(h.prog,"build_orth");
   h.kern_build_bior=CLKernelCreate(h.prog,"build_bior");
   h.kern_swap_reverse=CLKernelCreate(h.prog,"swap_reverse");
   if(h.kern_build_orth==INVALID_HANDLE || h.kern_build_bior==INVALID_HANDLE || h.kern_swap_reverse==INVALID_HANDLE)
     { PyWTWaveletFree(h); return false; }
   h.ready=true;
   return true;
  }

inline bool PyWTWaveletEnsure(PyWTWaveletHandle &h,const int lenIn1,const int lenIn2,const int lenIn3,const int lenIn4,const int lenOut)
  {
   if(!h.ready) return false;
   if(lenIn1<=0 || lenOut<=0) return false;
   if(h.memIn1==INVALID_HANDLE || h.lenIn1!=lenIn1)
     {
      if(h.memIn1!=INVALID_HANDLE) CLBufferFree(h.memIn1);
      h.memIn1=CLBufferCreate(h.ctx,lenIn1*sizeof(double),CL_MEM_READ_ONLY);
      if(h.memIn1==INVALID_HANDLE) return false;
      h.lenIn1=lenIn1;
     }
   if(lenIn2>0)
     {
      if(h.memIn2==INVALID_HANDLE || h.lenIn2!=lenIn2)
        {
         if(h.memIn2!=INVALID_HANDLE) CLBufferFree(h.memIn2);
         h.memIn2=CLBufferCreate(h.ctx,lenIn2*sizeof(double),CL_MEM_READ_ONLY);
         if(h.memIn2==INVALID_HANDLE) return false;
         h.lenIn2=lenIn2;
        }
     }
   if(lenIn3>0)
     {
      if(h.memIn3==INVALID_HANDLE || h.lenIn3!=lenIn3)
        {
         if(h.memIn3!=INVALID_HANDLE) CLBufferFree(h.memIn3);
         h.memIn3=CLBufferCreate(h.ctx,lenIn3*sizeof(double),CL_MEM_READ_ONLY);
         if(h.memIn3==INVALID_HANDLE) return false;
         h.lenIn3=lenIn3;
        }
     }
   if(lenIn4>0)
     {
      if(h.memIn4==INVALID_HANDLE || h.lenIn4!=lenIn4)
        {
         if(h.memIn4!=INVALID_HANDLE) CLBufferFree(h.memIn4);
         h.memIn4=CLBufferCreate(h.ctx,lenIn4*sizeof(double),CL_MEM_READ_ONLY);
         if(h.memIn4==INVALID_HANDLE) return false;
         h.lenIn4=lenIn4;
        }
     }
   if(h.memOut1==INVALID_HANDLE || h.lenOut!=lenOut)
     {
      if(h.memOut1!=INVALID_HANDLE) CLBufferFree(h.memOut1);
      if(h.memOut2!=INVALID_HANDLE) CLBufferFree(h.memOut2);
      if(h.memOut3!=INVALID_HANDLE) CLBufferFree(h.memOut3);
      if(h.memOut4!=INVALID_HANDLE) CLBufferFree(h.memOut4);
      h.memOut1=CLBufferCreate(h.ctx,lenOut*sizeof(double),CL_MEM_READ_WRITE);
      h.memOut2=CLBufferCreate(h.ctx,lenOut*sizeof(double),CL_MEM_READ_WRITE);
      h.memOut3=CLBufferCreate(h.ctx,lenOut*sizeof(double),CL_MEM_READ_WRITE);
      h.memOut4=CLBufferCreate(h.ctx,lenOut*sizeof(double),CL_MEM_READ_WRITE);
      if(h.memOut1==INVALID_HANDLE || h.memOut2==INVALID_HANDLE || h.memOut3==INVALID_HANDLE || h.memOut4==INVALID_HANDLE) return false;
      h.lenOut=lenOut;
     }
   return true;
  }

inline bool PyWT_BuildOrthGPU(const double &lo[],const double scale,PyWTDiscreteWavelet &w)
  {
   int n=ArraySize(lo);
   if(n<=0) return false;
   static PyWTWaveletHandle h; if(!h.ready) PyWTWaveletReset(h);
   if(!PyWTWaveletInit(h)) return false;
   if(!PyWTWaveletEnsure(h,n,0,0,0,n)) return false;
   CLBufferWrite(h.memIn1,lo);
   CLSetKernelArgMem(h.kern_build_orth,0,h.memIn1);
   CLSetKernelArg(h.kern_build_orth,1,n);
   CLSetKernelArg(h.kern_build_orth,2,scale);
   CLSetKernelArgMem(h.kern_build_orth,3,h.memOut1);
   CLSetKernelArgMem(h.kern_build_orth,4,h.memOut2);
   CLSetKernelArgMem(h.kern_build_orth,5,h.memOut3);
   CLSetKernelArgMem(h.kern_build_orth,6,h.memOut4);
   uint offs[1]={0}; uint work[1]={(uint)n};
   if(!CLExecute(h.kern_build_orth,1,offs,work)) return false;
   ArrayResize(w.rec_lo,n); ArrayResize(w.dec_lo,n); ArrayResize(w.rec_hi,n); ArrayResize(w.dec_hi,n);
   CLBufferRead(h.memOut1,w.rec_lo);
   CLBufferRead(h.memOut2,w.dec_lo);
   CLBufferRead(h.memOut3,w.rec_hi);
   CLBufferRead(h.memOut4,w.dec_hi);
   return true;
  }

inline bool PyWT_BuildBiorGPU(const double &rec_src[],const double &dec_src[],const int shift,PyWTDiscreteWavelet &w)
  {
   int nrec=ArraySize(rec_src);
   int ndec=ArraySize(dec_src);
   if(nrec<=0 || ndec<=0) return false;
   if(w.rec_len<=0 || w.dec_len<=0) return false;
   if(nrec < w.rec_len + shift) return false;
   static PyWTWaveletHandle h; if(!h.ready) PyWTWaveletReset(h);
   if(!PyWTWaveletInit(h)) return false;
   if(!PyWTWaveletEnsure(h,nrec,ndec,0,0,w.rec_len)) return false;
   CLBufferWrite(h.memIn1,rec_src);
   CLBufferWrite(h.memIn2,dec_src);
   CLSetKernelArgMem(h.kern_build_bior,0,h.memIn1);
   CLSetKernelArgMem(h.kern_build_bior,1,h.memIn2);
   CLSetKernelArg(h.kern_build_bior,2,w.rec_len);
   CLSetKernelArg(h.kern_build_bior,3,w.dec_len);
   CLSetKernelArg(h.kern_build_bior,4,shift);
   CLSetKernelArgMem(h.kern_build_bior,5,h.memOut1);
   CLSetKernelArgMem(h.kern_build_bior,6,h.memOut2);
   CLSetKernelArgMem(h.kern_build_bior,7,h.memOut3);
   CLSetKernelArgMem(h.kern_build_bior,8,h.memOut4);
   uint offs[1]={0}; uint work[1]={(uint)w.rec_len};
   if(!CLExecute(h.kern_build_bior,1,offs,work)) return false;
   ArrayResize(w.rec_lo,w.rec_len); ArrayResize(w.dec_lo,w.dec_len); ArrayResize(w.rec_hi,w.rec_len); ArrayResize(w.dec_hi,w.dec_len);
   CLBufferRead(h.memOut1,w.rec_lo);
   CLBufferRead(h.memOut2,w.dec_lo);
   CLBufferRead(h.memOut3,w.rec_hi);
   CLBufferRead(h.memOut4,w.dec_hi);
   return true;
  }

inline bool PyWT_SwapReverseGPU(PyWTDiscreteWavelet &w)
  {
   int n=w.rec_len;
   if(n<=0) return false;
   static PyWTWaveletHandle h; if(!h.ready) PyWTWaveletReset(h);
   if(!PyWTWaveletInit(h)) return false;
   if(!PyWTWaveletEnsure(h,n,n,n,n,n)) return false;
   CLBufferWrite(h.memIn1,w.rec_lo);
   CLBufferWrite(h.memIn2,w.rec_hi);
   CLBufferWrite(h.memIn3,w.dec_lo);
   CLBufferWrite(h.memIn4,w.dec_hi);
   CLSetKernelArgMem(h.kern_swap_reverse,0,h.memIn1);
   CLSetKernelArgMem(h.kern_swap_reverse,1,h.memIn2);
   CLSetKernelArgMem(h.kern_swap_reverse,2,h.memIn3);
   CLSetKernelArgMem(h.kern_swap_reverse,3,h.memIn4);
   CLSetKernelArg(h.kern_swap_reverse,4,n);
   CLSetKernelArgMem(h.kern_swap_reverse,5,h.memOut1);
   CLSetKernelArgMem(h.kern_swap_reverse,6,h.memOut2);
   CLSetKernelArgMem(h.kern_swap_reverse,7,h.memOut3);
   CLSetKernelArgMem(h.kern_swap_reverse,8,h.memOut4);
   uint offs[1]={0}; uint work[1]={(uint)n};
   if(!CLExecute(h.kern_swap_reverse,1,offs,work)) return false;
   CLBufferRead(h.memOut1,w.rec_lo);
   CLBufferRead(h.memOut2,w.rec_hi);
   CLBufferRead(h.memOut3,w.dec_lo);
   CLBufferRead(h.memOut4,w.dec_hi);
   return true;
  }


inline bool PyWT_BlankDiscreteWavelet(const int filters_length,PyWTDiscreteWavelet &w)
  {
   if(filters_length<=0) return false;
   w.dec_len=filters_length; w.rec_len=filters_length;
   ArrayResize(w.dec_hi,filters_length); ArrayResize(w.dec_lo,filters_length);
   ArrayResize(w.rec_hi,filters_length); ArrayResize(w.rec_lo,filters_length);
   for(int i=0;i<filters_length;i++){ w.dec_hi[i]=0.0; w.dec_lo[i]=0.0; w.rec_hi[i]=0.0; w.rec_lo[i]=0.0; }
   return true;
  }


inline bool PyWT_GetDb(const int order,double &out[]){
   switch(order){
      case 1: PyWT_CopyArray(db1, out); return true;
      case 2: PyWT_CopyArray(db2, out); return true;
      case 3: PyWT_CopyArray(db3, out); return true;
      case 4: PyWT_CopyArray(db4, out); return true;
      case 5: PyWT_CopyArray(db5, out); return true;
      case 6: PyWT_CopyArray(db6, out); return true;
      case 7: PyWT_CopyArray(db7, out); return true;
      case 8: PyWT_CopyArray(db8, out); return true;
      case 9: PyWT_CopyArray(db9, out); return true;
      case 10: PyWT_CopyArray(db10, out); return true;
      case 11: PyWT_CopyArray(db11, out); return true;
      case 12: PyWT_CopyArray(db12, out); return true;
      case 13: PyWT_CopyArray(db13, out); return true;
      case 14: PyWT_CopyArray(db14, out); return true;
      case 15: PyWT_CopyArray(db15, out); return true;
      case 16: PyWT_CopyArray(db16, out); return true;
      case 17: PyWT_CopyArray(db17, out); return true;
      case 18: PyWT_CopyArray(db18, out); return true;
      case 19: PyWT_CopyArray(db19, out); return true;
      case 20: PyWT_CopyArray(db20, out); return true;
      case 21: PyWT_CopyArray(db21, out); return true;
      case 22: PyWT_CopyArray(db22, out); return true;
      case 23: PyWT_CopyArray(db23, out); return true;
      case 24: PyWT_CopyArray(db24, out); return true;
      case 25: PyWT_CopyArray(db25, out); return true;
      case 26: PyWT_CopyArray(db26, out); return true;
      case 27: PyWT_CopyArray(db27, out); return true;
      case 28: PyWT_CopyArray(db28, out); return true;
      case 29: PyWT_CopyArray(db29, out); return true;
      case 30: PyWT_CopyArray(db30, out); return true;
      case 31: PyWT_CopyArray(db31, out); return true;
      case 32: PyWT_CopyArray(db32, out); return true;
      case 33: PyWT_CopyArray(db33, out); return true;
      case 34: PyWT_CopyArray(db34, out); return true;
      case 35: PyWT_CopyArray(db35, out); return true;
      case 36: PyWT_CopyArray(db36, out); return true;
      case 37: PyWT_CopyArray(db37, out); return true;
      case 38: PyWT_CopyArray(db38, out); return true;
      default: return false;
   }
}

inline bool PyWT_GetSym(const int order,double &out[]){
   switch(order){
      case 2: PyWT_CopyArray(sym2, out); return true;
      case 3: PyWT_CopyArray(sym3, out); return true;
      case 4: PyWT_CopyArray(sym4, out); return true;
      case 5: PyWT_CopyArray(sym5, out); return true;
      case 6: PyWT_CopyArray(sym6, out); return true;
      case 7: PyWT_CopyArray(sym7, out); return true;
      case 8: PyWT_CopyArray(sym8, out); return true;
      case 9: PyWT_CopyArray(sym9, out); return true;
      case 10: PyWT_CopyArray(sym10, out); return true;
      case 11: PyWT_CopyArray(sym11, out); return true;
      case 12: PyWT_CopyArray(sym12, out); return true;
      case 13: PyWT_CopyArray(sym13, out); return true;
      case 14: PyWT_CopyArray(sym14, out); return true;
      case 15: PyWT_CopyArray(sym15, out); return true;
      case 16: PyWT_CopyArray(sym16, out); return true;
      case 17: PyWT_CopyArray(sym17, out); return true;
      case 18: PyWT_CopyArray(sym18, out); return true;
      case 19: PyWT_CopyArray(sym19, out); return true;
      case 20: PyWT_CopyArray(sym20, out); return true;
      default: return false;
   }
}

inline bool PyWT_GetCoif(const int order,double &out[]){
   switch(order){
      case 1: PyWT_CopyArray(coif1, out); return true;
      case 2: PyWT_CopyArray(coif2, out); return true;
      case 3: PyWT_CopyArray(coif3, out); return true;
      case 4: PyWT_CopyArray(coif4, out); return true;
      case 5: PyWT_CopyArray(coif5, out); return true;
      case 6: PyWT_CopyArray(coif6, out); return true;
      case 7: PyWT_CopyArray(coif7, out); return true;
      case 8: PyWT_CopyArray(coif8, out); return true;
      case 9: PyWT_CopyArray(coif9, out); return true;
      case 10: PyWT_CopyArray(coif10, out); return true;
      case 11: PyWT_CopyArray(coif11, out); return true;
      case 12: PyWT_CopyArray(coif12, out); return true;
      case 13: PyWT_CopyArray(coif13, out); return true;
      case 14: PyWT_CopyArray(coif14, out); return true;
      case 15: PyWT_CopyArray(coif15, out); return true;
      case 16: PyWT_CopyArray(coif16, out); return true;
      case 17: PyWT_CopyArray(coif17, out); return true;
      default: return false;
   }
}

inline bool PyWT_GetDmey(double &out[]){ PyWT_CopyArray(dmey, out); return true; }

inline bool PyWT_GetBiorArray(const int N,const int M,double &out[]){
   switch(N){
      case 1:
         switch(M){
            case 0: PyWT_CopyArray(bior1_0, out); return true;
            case 1: PyWT_CopyArray(bior1_1, out); return true;
            case 3: PyWT_CopyArray(bior1_3, out); return true;
            case 5: PyWT_CopyArray(bior1_5, out); return true;
            default: return false;
         }
         break;
      case 2:
         switch(M){
            case 0: PyWT_CopyArray(bior2_0, out); return true;
            case 2: PyWT_CopyArray(bior2_2, out); return true;
            case 4: PyWT_CopyArray(bior2_4, out); return true;
            case 6: PyWT_CopyArray(bior2_6, out); return true;
            case 8: PyWT_CopyArray(bior2_8, out); return true;
            default: return false;
         }
         break;
      case 3:
         switch(M){
            case 0: PyWT_CopyArray(bior3_0, out); return true;
            case 1: PyWT_CopyArray(bior3_1, out); return true;
            case 3: PyWT_CopyArray(bior3_3, out); return true;
            case 5: PyWT_CopyArray(bior3_5, out); return true;
            case 7: PyWT_CopyArray(bior3_7, out); return true;
            case 9: PyWT_CopyArray(bior3_9, out); return true;
            default: return false;
         }
         break;
      case 4:
         switch(M){
            case 0: PyWT_CopyArray(bior4_0, out); return true;
            case 4: PyWT_CopyArray(bior4_4, out); return true;
            default: return false;
         }
         break;
      case 5:
         switch(M){
            case 0: PyWT_CopyArray(bior5_0, out); return true;
            case 5: PyWT_CopyArray(bior5_5, out); return true;
            default: return false;
         }
         break;
      case 6:
         switch(M){
            case 0: PyWT_CopyArray(bior6_0, out); return true;
            case 8: PyWT_CopyArray(bior6_8, out); return true;
            default: return false;
         }
         break;
      default: return false;
   }
   return false;
}


inline bool PyWT_DiscreteWavelet(const int name,const int order,PyWTDiscreteWavelet &w)
  {
   PyWT_ResetWavelet(w);
   const double sqrt2 = 1.4142135623730950488016887242096980785696718753769;
   if(name==PYWT_HAAR){
      if(!PyWT_DiscreteWavelet(PYWT_DB,1,w)) return false;
      w.family_name="Haar"; w.short_name="haar";
      return true;
   }

   if(name==PYWT_RBIO){
      if(!PyWT_DiscreteWavelet(PYWT_BIOR,order,w)) return false;
      // swap rec/dec and reverse filters (GPU)
      if(!PyWT_SwapReverseGPU(w)) return false;
      int tlen = w.rec_len; w.rec_len = w.dec_len; w.dec_len = tlen;
      w.family_name="Reverse biorthogonal"; w.short_name="rbio";
      return true;
   }

   switch(name){
      case PYWT_DB:{
         double lo[]; if(!PyWT_GetDb(order,lo)) return false;
         if(!PyWT_BlankDiscreteWavelet(2*order,w)) return false;
         w.vanishing_moments_psi=order; w.vanishing_moments_phi=0;
         w.support_width=2*order-1; w.orthogonal=true; w.biorthogonal=true; w.symmetry=PYWT_ASYMMETRIC; w.compact_support=true;
         w.family_name="Daubechies"; w.short_name="db";
         if(!PyWT_BuildOrthGPU(lo,1.0,w)) return false;
         return true;
      }
      case PYWT_SYM:{
         double lo[]; if(!PyWT_GetSym(order,lo)) return false;
         if(!PyWT_BlankDiscreteWavelet(2*order,w)) return false;
         w.vanishing_moments_psi=order; w.vanishing_moments_phi=0;
         w.support_width=2*order-1; w.orthogonal=true; w.biorthogonal=true; w.symmetry=PYWT_NEAR_SYMMETRIC; w.compact_support=true;
         w.family_name="Symlets"; w.short_name="sym";
         if(!PyWT_BuildOrthGPU(lo,1.0,w)) return false;
         return true;
      }
      case PYWT_COIF:{
         double lo[]; if(!PyWT_GetCoif(order,lo)) return false;
         if(!PyWT_BlankDiscreteWavelet(6*order,w)) return false;
         w.vanishing_moments_psi=2*order; w.vanishing_moments_phi=2*order-1;
         w.support_width=6*order-1; w.orthogonal=true; w.biorthogonal=true; w.symmetry=PYWT_NEAR_SYMMETRIC; w.compact_support=true;
         w.family_name="Coiflets"; w.short_name="coif";
         if(!PyWT_BuildOrthGPU(lo,sqrt2,w)) return false;
         return true;
      }
      case PYWT_BIOR:{
         int N=order/10; int M=order%10; int M_max=0;
         switch(N){
            case 1: if((M%2)!=1 || M>5) return false; M_max=5; break;
            case 2: if((M%2)!=0 || M<2 || M>8) return false; M_max=8; break;
            case 3: if((M%2)!=1) return false; M_max=9; break;
            case 4: if(M!=4) return false; M_max=4; break;
            case 5: if(M!=5) return false; M_max=5; break;
            case 6: if(M!=8) return false; M_max=8; break;
            default: return false;
         }
         int flen = (N==1) ? 2*M : (2*M + 2);
         if(!PyWT_BlankDiscreteWavelet(flen,w)) return false;
         w.vanishing_moments_psi=order/10; w.vanishing_moments_phi=order%10;
         w.support_width=-1; w.orthogonal=false; w.biorthogonal=true; w.symmetry=PYWT_SYMMETRIC; w.compact_support=true;
         w.family_name="Biorthogonal"; w.short_name="bior";
         double rec_src[]; double dec_src[];
         if(!PyWT_GetBiorArray(N,0,rec_src)) return false;
         if(!PyWT_GetBiorArray(N, M, dec_src)) return false;
         int n = M_max - M;
         if(!PyWT_BuildBiorGPU(rec_src,dec_src,n,w)) return false;
         return true;
      }
      case PYWT_DMEY:{
         double lo[]; if(!PyWT_GetDmey(lo)) return false;
         if(!PyWT_BlankDiscreteWavelet(ArraySize(lo),w)) return false;
         w.vanishing_moments_psi=-1; w.vanishing_moments_phi=-1;
         w.support_width=-1; w.orthogonal=true; w.biorthogonal=true; w.symmetry=PYWT_SYMMETRIC; w.compact_support=true;
         w.family_name="Discrete Meyer (FIR Approximation)"; w.short_name="dmey";
         if(!PyWT_BuildOrthGPU(lo,1.0,w)) return false;
         return true;
      }
      default: return false;
   }
  }


#endif // __SPECTRAL_PYWT_WAVELETS_MQH__
