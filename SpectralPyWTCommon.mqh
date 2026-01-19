#ifndef __SPECTRAL_PYWT_COMMON_MQH__
#define __SPECTRAL_PYWT_COMMON_MQH__

// Core enums and buffer-length helpers from PyWavelets common.{h,c}

enum PyWTSymmetry
  {
   PYWT_UNKNOWN = -1,
   PYWT_ASYMMETRIC = 0,
   PYWT_NEAR_SYMMETRIC = 1,
   PYWT_SYMMETRIC = 2,
   PYWT_ANTI_SYMMETRIC = 3
  };

enum PyWTWaveletName
  {
   PYWT_HAAR = 0,
   PYWT_RBIO = 1,
   PYWT_DB = 2,
   PYWT_SYM = 3,
   PYWT_COIF = 4,
   PYWT_BIOR = 5,
   PYWT_DMEY = 6,
   PYWT_GAUS = 7,
   PYWT_MEXH = 8,
   PYWT_MORL = 9,
   PYWT_CGAU = 10,
   PYWT_SHAN = 11,
   PYWT_FBSP = 12,
   PYWT_CMOR = 13
  };

enum PyWTMode
  {
   PYWT_MODE_INVALID = -1,
   PYWT_MODE_ZEROPAD = 0,
   PYWT_MODE_SYMMETRIC,
   PYWT_MODE_CONSTANT_EDGE,
   PYWT_MODE_SMOOTH,
   PYWT_MODE_PERIODIC,
   PYWT_MODE_PERIODIZATION,
   PYWT_MODE_REFLECT,
   PYWT_MODE_ANTISYMMETRIC,
   PYWT_MODE_ANTIREFLECT,
   PYWT_MODE_MAX
  };

inline int PyWT_DwtBufferLength(const int input_len,const int filter_len,const int mode)
  {
   if(input_len < 1 || filter_len < 1) return 0;
   if(mode==PYWT_MODE_PERIODIZATION)
      return (input_len/2) + (((input_len%2)!=0)?1:0);
   return (input_len + filter_len - 1)/2;
  }

inline int PyWT_ReconstructionBufferLength(const int coeffs_len,const int filter_len)
  {
   if(coeffs_len < 1 || filter_len < 1) return 0;
   return 2*coeffs_len + filter_len - 2;
  }

inline int PyWT_IdwtBufferLength(const int coeffs_len,const int filter_len,const int mode)
  {
   if(mode==PYWT_MODE_PERIODIZATION) return 2*coeffs_len;
   return 2*coeffs_len - filter_len + 2;
  }

inline int PyWT_SwtBufferLength(const int input_len)
  {
   return input_len;
  }

inline int PyWT_DwtMaxLevel(const int input_len,const int filter_len)
  {
   if(filter_len <= 1 || input_len < (filter_len-1)) return 0;
   int n = input_len/(filter_len-1);
   int lvl = 0;
   while(n > 1)
     {
      n >>= 1;
      lvl++;
     }
   return lvl;
  }

inline int PyWT_SwtMaxLevel(const int input_len)
  {
   int j=0;
   int n=input_len;
   while(n > 0)
     {
      if((n & 1)!=0) return j;
      n >>= 1;
      j++;
     }
   return j;
  }

#endif // __SPECTRAL_PYWT_COMMON_MQH__
