#ifndef __SPECTRAL_PYWT_CONV_MQH__
#define __SPECTRAL_PYWT_CONV_MQH__

#include "SpectralPyWTCommon.mqh"
#include "SpectralOpenCLCommon.mqh"

struct PyWTConvHandle
  {
   int ctx;
   int prog;
   int kern_down;
   int kern_down_per;
   int kern_up_full;
   int kern_up_valid;
   int kern_up_valid_per;
   int memIn;
   int memF;
   int memOut;
   bool ready;
  };

inline void PyWTConvReset(PyWTConvHandle &h)
  {
   h.ctx=INVALID_HANDLE; h.prog=INVALID_HANDLE;
   h.kern_down=INVALID_HANDLE; h.kern_down_per=INVALID_HANDLE;
   h.kern_up_full=INVALID_HANDLE; h.kern_up_valid=INVALID_HANDLE; h.kern_up_valid_per=INVALID_HANDLE;
   h.memIn=INVALID_HANDLE; h.memF=INVALID_HANDLE; h.memOut=INVALID_HANDLE;
   h.ready=false;
  }

inline void PyWTConvFree(PyWTConvHandle &h)
  {
   if(h.memIn!=INVALID_HANDLE) { CLBufferFree(h.memIn); h.memIn=INVALID_HANDLE; }
   if(h.memF!=INVALID_HANDLE) { CLBufferFree(h.memF); h.memF=INVALID_HANDLE; }
   if(h.memOut!=INVALID_HANDLE) { CLBufferFree(h.memOut); h.memOut=INVALID_HANDLE; }
   if(h.kern_down!=INVALID_HANDLE) { CLKernelFree(h.kern_down); h.kern_down=INVALID_HANDLE; }
   if(h.kern_down_per!=INVALID_HANDLE) { CLKernelFree(h.kern_down_per); h.kern_down_per=INVALID_HANDLE; }
   if(h.kern_up_full!=INVALID_HANDLE) { CLKernelFree(h.kern_up_full); h.kern_up_full=INVALID_HANDLE; }
   if(h.kern_up_valid!=INVALID_HANDLE) { CLKernelFree(h.kern_up_valid); h.kern_up_valid=INVALID_HANDLE; }
   if(h.kern_up_valid_per!=INVALID_HANDLE) { CLKernelFree(h.kern_up_valid_per); h.kern_up_valid_per=INVALID_HANDLE; }
   if(h.prog!=INVALID_HANDLE) { CLProgramFree(h.prog); h.prog=INVALID_HANDLE; }
   if(h.ctx!=INVALID_HANDLE) { CLContextFree(h.ctx); h.ctx=INVALID_HANDLE; }
   h.ready=false;
  }

inline bool PyWTConvInit(PyWTConvHandle &h)
  {
   if(h.ready) return true;
   PyWTConvReset(h);
   h.ctx=CLCreateContextGPUFloat64("SpectralPyWTConvolution");
   if(h.ctx==INVALID_HANDLE) return false;
   string code=
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"#define MODE_ZEROPAD 0\n"
"#define MODE_SYMMETRIC 1\n"
"#define MODE_CONSTANT_EDGE 2\n"
"#define MODE_SMOOTH 3\n"
"#define MODE_PERIODIC 4\n"
"#define MODE_PERIODIZATION 5\n"
"#define MODE_REFLECT 6\n"
"#define MODE_ANTISYMMETRIC 7\n"
"#define MODE_ANTIREFLECT 8\n"
"\n"
"int downsampling_convolution_periodization(const __global double* input, int N,\n"
"                                            const __global double* filter, int F,\n"
"                                            __global double* output, int step, int fstep)\n"
"{\n"
"    int i = F/2, o = 0;\n"
"    int padding = (step - (N % step)) % step;\n"
"\n"
"    for (; i < F && i < N; i += step, ++o) {\n"
"        double sum = 0.0;\n"
"        int j;\n"
"        int k_start = 0;\n"
"        for (j = 0; j <= i; j += fstep)\n"
"            sum += filter[j] * input[i-j];\n"
"        if (fstep > 1)\n"
"            k_start = j - (i + 1);\n"
"        while (j < F){\n"
"            int k;\n"
"            for (k = k_start; k < padding && j < F; k += fstep, j += fstep)\n"
"                sum += filter[j] * input[N-1];\n"
"            for (k = k_start; k < N && j < F; k += fstep, j += fstep)\n"
"                sum += filter[j] * input[N-1-k];\n"
"        }\n"
"        output[o] = sum;\n"
"    }\n"
"\n"
"    for(; i < N; i+=step, ++o){\n"
"        double sum = 0.0;\n"
"        int j;\n"
"        for(j = 0; j < F; j += fstep)\n"
"            sum += input[i-j]*filter[j];\n"
"        output[o] = sum;\n"
"    }\n"
"\n"
"    for (; i < F && i < N + F/2; i += step, ++o) {\n"
"        double sum = 0.0;\n"
"        int j = 0;\n"
"        int k_start = 0;\n"
"        while (i-j >= N){\n"
"            int k;\n"
"            // for simplicity, not using fstep here\n"
"            for (k = 0; k < padding && i-j >= N; ++k, ++j)\n"
"                sum += filter[i-N-j] * input[N-1];\n"
"            for (k = 0; k < N && i-j >= N; ++k, ++j)\n"
"                sum += filter[i-N-j] * input[k];\n"
"        }\n"
"        if (fstep > 1)\n"
"            j += (fstep - j % fstep) % fstep;  // move to next non-zero entry\n"
"        for (; j <= i; j += fstep)\n"
"            sum += filter[j] * input[i-j];\n"
"        if (fstep > 1)\n"
"            k_start = j - (i + 1);\n"
"        while (j < F){\n"
"            int k;\n"
"            for (k = k_start; k < padding && j < F; k += fstep, j += fstep)\n"
"                sum += filter[j] * input[N-1];\n"
"            for (k = k_start; k < N && j < F; k += fstep, j += fstep)\n"
"                sum += filter[j] * input[N-1-k];\n"
"        }\n"
"        output[o] = sum;\n"
"    }\n"
"\n"
"    for(; i < N + F/2; i += step, ++o){\n"
"        double sum = 0.0;\n"
"        int j = 0;\n"
"        while (i-j >= N){\n"
"            // for simplicity, not using fstep here\n"
"            int k;\n"
"            for (k = 0; k < padding && i-j >= N; ++k, ++j)\n"
"                sum += filter[i-N-j] * input[N-1];\n"
"            for (k = 0; k < N && i-j >= N; ++k, ++j)\n"
"                sum += filter[i-N-j] * input[k];\n"
"        }\n"
"        if (fstep > 1)\n"
"            j += (fstep - j % fstep) % fstep;  // move to next non-zero entry\n"
"        for (; j < F; j += fstep)\n"
"            sum += filter[j] * input[i-j];\n"
"        output[o] = sum;\n"
"    }\n"
"    return 0;\n"
"}\n"
"\n"
"int downsampling_convolution(const __global double* input, int N,\n"
"                              const __global double* filter, int F,\n"
"                              __global double* output,\n"
"                              int step, int mode)\n"
"{\n"
"    int i = step - 1, o = 0;\n"
"\n"
"    if(mode == MODE_PERIODIZATION)\n"
"        return downsampling_convolution_periodization(input, N, filter, F, output, step, 1);\n"
"\n"
"    if (mode == MODE_SMOOTH && N < 2)\n"
"        mode = MODE_CONSTANT_EDGE;\n"
"\n"
"    // left boundary overhang\n"
"    for(; i < F && i < N; i+=step, ++o){\n"
"        double sum = 0.0;\n"
"        int j;\n"
"        for(j = 0; j <= i; ++j)\n"
"            sum += filter[j]*input[i-j];\n"
"\n"
"        switch(mode) {\n"
"        case MODE_SYMMETRIC:\n"
"            while (j < F){\n"
"                int k;\n"
"                for(k = 0; k < N && j < F; ++j, ++k)\n"
"                    sum += filter[j]*input[k];\n"
"                for(k = 0; k < N && j < F; ++k, ++j)\n"
"                    sum += filter[j]*input[N-1-k];\n"
"            }\n"
"            break;\n"
"        case MODE_ANTISYMMETRIC:\n"
"            // half-sample anti-symmetric\n"
"            while (j < F){\n"
"                int k;\n"
"                for(k = 0; k < N && j < F; ++j, ++k)\n"
"                    sum -= filter[j]*input[k];\n"
"                for(k = 0; k < N && j < F; ++k, ++j)\n"
"                    sum += filter[j]*input[N-1-k];\n"
"            }\n"
"            break;\n"
"        case MODE_REFLECT:\n"
"            while (j < F){\n"
"                int k;\n"
"                for(k = 1; k < N && j < F; ++j, ++k)\n"
"                    sum += filter[j]*input[k];\n"
"                for(k = 1; k < N && j < F; ++k, ++j)\n"
"                    sum += filter[j]*input[N-1-k];\n"
"            }\n"
"            break;\n"
"        case MODE_ANTIREFLECT:{\n"
"            // whole-sample anti-symmetric\n"
"            int k;\n"
"            double le = input[0];    // current left edge value\n"
"            double tmp = 0.0;\n"
"            while (j < F) {\n"
"                for(k = 1; k < N && j < F; ++j, ++k){\n"
"                    tmp = le - (input[k] - input[0]);\n"
"                    sum += filter[j]*tmp;\n"
"                }\n"
"                le = tmp;\n"
"                for(k = 1; k < N && j < F; ++j, ++k){\n"
"                    tmp = le + (input[N-1-k] - input[N-1]);\n"
"                    sum += filter[j]*tmp;\n"
"                }\n"
"                le = tmp;\n"
"            }\n"
"            break;\n"
"            }\n"
"        case MODE_CONSTANT_EDGE:\n"
"            for(; j < F; ++j)\n"
"                sum += filter[j]*input[0];\n"
"            break;\n"
"        case MODE_SMOOTH:{\n"
"            int k;\n"
"            for(k = 1; j < F; ++j, ++k)\n"
"                sum += filter[j]*(input[0] + k * (input[0] - input[1]));\n"
"            break;\n"
"        }\n"
"        case MODE_PERIODIC:\n"
"            while (j < F){\n"
"                int k;\n"
"                for(k = 0; k < N && j < F; ++k, ++j)\n"
"                    sum += filter[j]*input[N-1-k];\n"
"            }\n"
"            break;\n"
"        case MODE_ZEROPAD:\n"
"        default:\n"
"            break;\n"
"        }\n"
"        output[o] = sum;\n"
"    }\n"
"\n"
"    // center (if input equal or wider than filter: N >= F)\n"
"    for(; i < N; i+=step, ++o){\n"
"        double sum = 0.0;\n"
"        int j;\n"
"        for(j = 0; j < F; ++j)\n"
"            sum += input[i-j]*filter[j];\n"
"        output[o] = sum;\n"
"    }\n"
"\n"
"    // center (if filter is wider than input: F > N)\n"
"    for(; i < F; i+=step, ++o){\n"
"        double sum = 0.0;\n"
"        int j = 0;\n"
"\n"
"        switch(mode) {\n"
"        case MODE_SYMMETRIC:\n"
"            // Included from original: TODO: j < F-_offset\n"
"            /* Iterate over filter in reverse to process elements away from\n"
"             * data. This gives a known first input element to process (N-1)\n"
"             */\n"
"            while (i - j >= N){\n"
"                int k;\n"
"                for(k = 0; k < N && i-j >= N; ++j, ++k)\n"
"                    sum += filter[i-N-j]*input[N-1-k];\n"
"                for(k = 0; k < N && i-j >= N; ++j, ++k)\n"
"                    sum += filter[i-N-j]*input[k];\n"
"            }\n"
"            break;\n"
"        case MODE_ANTISYMMETRIC:\n"
"            // half-sample anti-symmetric\n"
"            while (i - j >= N){\n"
"                int k;\n"
"                for(k = 0; k < N && i-j >= N; ++j, ++k)\n"
"                    sum -= filter[i-N-j]*input[N-1-k];\n"
"                for(k = 0; k < N && i-j >= N; ++j, ++k)\n"
"                    sum += filter[i-N-j]*input[k];\n"
"            }\n"
"            break;\n"
"        case MODE_REFLECT:\n"
"            while (i - j >= N){\n"
"                int k;\n"
"                for(k = 1; k < N && i-j >= N; ++j, ++k)\n"
"                    sum += filter[i-N-j]*input[N-1-k];\n"
"                for(k = 1; k < N && i-j >= N; ++j, ++k)\n"
"                    sum += filter[i-N-j]*input[k];\n"
"            }\n"
"            break;\n"
"        case MODE_ANTIREFLECT:{\n"
"            // whole-sample anti-symmetric\n"
"            int k;\n"
"            double re = input[N-1];    // current right edge value\n"
"            double tmp = 0.0;\n"
"            while (i - j >= N) {\n"
"                for(k = 1; k < N && i-j >= N; ++j, ++k){\n"
"                    tmp = re - (input[N-1-k] - input[N-1]);\n"
"                    sum += filter[i-N-j]*tmp;\n"
"                }\n"
"                re = tmp;\n"
"                for(k = 1; k < N && i-j >= N; ++j, ++k){\n"
"                    tmp = re + (input[k] - input[0]);\n"
"                    sum += filter[i-N-j]*tmp;\n"
"                }\n"
"                re = tmp;\n"
"            }\n"
"            break;\n"
"        }\n"
"        case MODE_CONSTANT_EDGE:\n"
"            for(; i-j >= N; ++j)\n"
"                sum += filter[j]*input[N-1];\n"
"            break;\n"
"        case MODE_SMOOTH:{\n"
"            int k;\n"
"            for(k = i - N + 1; i-j >= N; ++j, --k)\n"
"                sum += filter[j]*(input[N-1] + k * (input[N-1] - input[N-2]));\n"
"            break;\n"
"        }\n"
"        case MODE_PERIODIC:\n"
"            while (i-j >= N){\n"
"                int k;\n"
"                for (k = 0; k < N && i-j >= N; ++j, ++k)\n"
"                    sum += filter[i-N-j]*input[k];\n"
"            }\n"
"            break;\n"
"        case MODE_ZEROPAD:\n"
"        default:\n"
"            j = i - N + 1;\n"
"            break;\n"
"        }\n"
"\n"
"        for(; j <= i; ++j)\n"
"            sum += filter[j]*input[i-j];\n"
"\n"
"        switch(mode) {\n"
"        case MODE_SYMMETRIC:\n"
"            while (j < F){\n"
"                int k;\n"
"                for(k = 0; k < N && j < F; ++j, ++k)\n"
"                    sum += filter[j]*input[k];\n"
"                for(k = 0; k < N && j < F; ++k, ++j)\n"
"                    sum += filter[j]*input[N-1-k];\n"
"            }\n"
"            break;\n"
"        case MODE_ANTISYMMETRIC:\n"
"            // half-sample anti-symmetric\n"
"            while (j < F){\n"
"                int k;\n"
"                for(k = 0; k < N && j < F; ++j, ++k)\n"
"                    sum -= filter[j]*input[k];\n"
"                for(k = 0; k < N && j < F; ++k, ++j)\n"
"                    sum += filter[j]*input[N-1-k];\n"
"            }\n"
"            break;\n"
"        case MODE_REFLECT:\n"
"            while (j < F){\n"
"                int k;\n"
"                for(k = 1; k < N && j < F; ++j, ++k)\n"
"                    sum += filter[j]*input[k];\n"
"                for(k = 1; k < N && j < F; ++k, ++j)\n"
"                    sum += filter[j]*input[N-1-k];\n"
"            }\n"
"            break;\n"
"        case MODE_ANTIREFLECT:{\n"
"            // whole-sample anti-symmetric\n"
"            int k;\n"
"            double le = input[0];    // current left edge value\n"
"            double tmp = 0.0;\n"
"            while (j < F) {\n"
"                for(k = 1; k < N && j < F; ++j, ++k){\n"
"                    tmp = le - (input[k] - input[0]);\n"
"                    sum += filter[j]*tmp;\n"
"                }\n"
"                le = tmp;\n"
"                for(k = 1; k < N && j < F; ++j, ++k){\n"
"                    tmp = le + (input[N-1-k] - input[N-1]);\n"
"                    sum += filter[j]*tmp;\n"
"                }\n"
"                le = tmp;\n"
"            }\n"
"            break;\n"
"            }\n"
"        case MODE_CONSTANT_EDGE:\n"
"            for(; j < F; ++j)\n"
"                sum += filter[j]*input[0];\n"
"            break;\n"
"        case MODE_SMOOTH:{\n"
"            int k;\n"
"            for(k = 1; j < F; ++j, ++k)\n"
"                sum += filter[j]*(input[0] + k * (input[0] - input[1]));\n"
"            break;\n"
"        }\n"
"        case MODE_PERIODIC:\n"
"            while (j < F){\n"
"                int k;\n"
"                for(k = 0; k < N && j < F; ++k, ++j)\n"
"                    sum += filter[j]*input[N-1-k];\n"
"            }\n"
"            break;\n"
"        case MODE_ZEROPAD:\n"
"        default:\n"
"            break;\n"
"        }\n"
"        output[o] = sum;\n"
"    }\n"
"\n"
"    // right boundary overhang\n"
"    for(; i < N+F-1; i += step, ++o){\n"
"        double sum = 0.0;\n"
"        int j = 0;\n"
"        switch(mode) {\n"
"        case MODE_SYMMETRIC:\n"
"            // Included from original: TODO: j < F-_offset\n"
"            while (i - j >= N){\n"
"                int k;\n"
"                for(k = 0; k < N && i-j >= N; ++j, ++k)\n"
"                    sum += filter[i-N-j]*input[N-1-k];\n"
"                for(k = 0; k < N && i-j >= N; ++j, ++k)\n"
"                    sum += filter[i-N-j]*input[k];\n"
"            }\n"
"            break;\n"
"        case MODE_ANTISYMMETRIC:\n"
"            // half-sample anti-symmetric\n"
"            while (i - j >= N){\n"
"                int k;\n"
"                for(k = 0; k < N && i-j >= N; ++j, ++k)\n"
"                    sum -= filter[i-N-j]*input[N-1-k];\n"
"                for(k = 0; k < N && i-j >= N; ++j, ++k)\n"
"                    sum += filter[i-N-j]*input[k];\n"
"            }\n"
"            break;\n"
"        case MODE_REFLECT:\n"
"            while (i - j >= N){\n"
"                int k;\n"
"                for(k = 1; k < N && i-j >= N; ++j, ++k)\n"
"                    sum += filter[i-N-j]*input[N-1-k];\n"
"                for(k = 1; k < N && i-j >= N; ++j, ++k)\n"
"                    sum += filter[i-N-j]*input[k];\n"
"            }\n"
"            break;\n"
"        case MODE_ANTIREFLECT:{\n"
"            // whole-sample anti-symmetric\n"
"            int k;\n"
"            double re = input[N-1];    //current right edge value\n"
"            double tmp = 0.0;\n"
"            while (i - j >= N) {\n"
"                //first reflection\n"
"                for(k = 1; k < N && i-j >= N; ++j, ++k){\n"
"                    tmp = re - (input[N-1-k] - input[N-1]);\n"
"                    sum += filter[i-N-j]*tmp;\n"
"                }\n"
"                re = tmp;\n"
"                //second reflection\n"
"                for(k = 1; k < N && i-j >= N; ++j, ++k){\n"
"                    tmp = re + (input[k] - input[0]);\n"
"                    sum += filter[i-N-j]*tmp;\n"
"                }\n"
"                re = tmp;\n"
"            }\n"
"            break;\n"
"        }\n"
"        case MODE_CONSTANT_EDGE:\n"
"            for(; i-j >= N; ++j)\n"
"                sum += filter[j]*input[N-1];\n"
"            break;\n"
"        case MODE_SMOOTH:{\n"
"            int k;\n"
"            for(k = i - N + 1; i-j >= N; ++j, --k)\n"
"                sum += filter[j]*(input[N-1] + k * (input[N-1] - input[N-2]));\n"
"            break;\n"
"        }\n"
"        case MODE_PERIODIC:\n"
"            while (i-j >= N){\n"
"                int k;\n"
"                for (k = 0; k < N && i-j >= N; ++j, ++k)\n"
"                    sum += filter[i-N-j]*input[k];\n"
"            }\n"
"            break;\n"
"        case MODE_ZEROPAD:\n"
"        default:\n"
"            j = i - N + 1;\n"
"            break;\n"
"        }\n"
"\n"
"        for(; j <= i; ++j)\n"
"            sum += filter[j]*input[i-j];\n"
"\n"
"        switch(mode) {\n"
"        case MODE_SYMMETRIC:\n"
"            while (j < F){\n"
"                int k;\n"
"                for(k = 0; k < N && j < F; ++j, ++k)\n"
"                    sum += filter[j]*input[k];\n"
"                for(k = 0; k < N && j < F; ++k, ++j)\n"
"                    sum += filter[j]*input[N-1-k];\n"
"            }\n"
"            break;\n"
"        case MODE_ANTISYMMETRIC:\n"
"            // half-sample anti-symmetric\n"
"            while (j < F){\n"
"                int k;\n"
"                for(k = 0; k < N && j < F; ++j, ++k)\n"
"                    sum -= filter[j]*input[k];\n"
"                for(k = 0; k < N && j < F; ++k, ++j)\n"
"                    sum += filter[j]*input[N-1-k];\n"
"            }\n"
"            break;\n"
"        case MODE_REFLECT:\n"
"            while (j < F){\n"
"                int k;\n"
"                for(k = 1; k < N && j < F; ++j, ++k)\n"
"                    sum += filter[j]*input[k];\n"
"                for(k = 1; k < N && j < F; ++k, ++j)\n"
"                    sum += filter[j]*input[N-1-k];\n"
"            }\n"
"            break;\n"
"        case MODE_ANTIREFLECT:{\n"
"            // whole-sample anti-symmetric\n"
"            int k;\n"
"            double le = input[0];    // current left edge value\n"
"            double tmp = 0.0;\n"
"            while (j < F) {\n"
"                for(k = 1; k < N && j < F; ++j, ++k){\n"
"                    tmp = le - (input[k] - input[0]);\n"
"                    sum += filter[j]*tmp;\n"
"                }\n"
"                le = tmp;\n"
"                for(k = 1; k < N && j < F; ++j, ++k){\n"
"                    tmp = le + (input[N-1-k] - input[N-1]);\n"
"                    sum += filter[j]*tmp;\n"
"                }\n"
"                le = tmp;\n"
"            }\n"
"            break;\n"
"            }\n"
"        case MODE_CONSTANT_EDGE:\n"
"            for(; j < F; ++j)\n"
"                sum += filter[j]*input[0];\n"
"            break;\n"
"        case MODE_SMOOTH:{\n"
"            int k;\n"
"            for(k = 1; j < F; ++j, ++k)\n"
"                sum += filter[j]*(input[0] + k * (input[0] - input[1]));\n"
"            break;\n"
"        }\n"
"        case MODE_PERIODIC:\n"
"            while (j < F){\n"
"                int k;\n"
"                for(k = 0; k < N && j < F; ++k, ++j)\n"
"                    sum += filter[j]*input[N-1-k];\n"
"            }\n"
"            break;\n"
"        case MODE_ZEROPAD:\n"
"        default:\n"
"            break;\n"
"        }\n"
"        output[o] = sum;\n"
"    }\n"
"\n"
"    return 0;\n"
"}\n"
"\n"
"int upsampling_convolution_full(const __global double* input, int N,\n"
"                                const __global double* filter, int F,\n"
"                                __global double* output, int O)\n"
"{\n"
"    int o = 0;\n"
"    if(F < 2) return -1;\n"
"    if(F % 2) return -3;\n"
"    for(o=0; o < O; ++o) output[o]=0.0;\n"
"    // direct formula for upsampled convolution\n"
"    for(o=0; o < O; ++o){\n"
"        double sum = 0.0;\n"
"        for(int j=0;j<F;++j){\n"
"            int idx = o - j;\n"
"            if((idx & 1) != 0) continue; // odd index => zero\n"
"            int k = idx >> 1;\n"
"            if(k < 0 || k >= N) continue;\n"
"            sum += filter[j] * input[k];\n"
"        }\n"
"        output[o] = sum;\n"
"    }\n"
"    return 0;\n"
"}\n"
"\n"
"int upsampling_convolution_valid_sf_periodization(const __global double* input, int N,\n"
"                                                  const __global double* filter, int F,\n"
"                                                  __global double* output)\n"
"{\n"
"    // TODO? Allow for non-2 step\n"
"    int start = F/4;\n"
"    int i = start;\n"
"    int end = N + start - (((F/2)%2) ? 0 : 1);\n"
"    int o = 0;\n"
"\n"
"    if(F%2) return -3; /* Filter must have even-length. */\n"
"\n"
"    // zero output\n"
"    for(int t=0;t<2*N;++t) output[t]=0.0;\n"
"\n"
"    if ((F/2)%2 == 0){\n"
"        // Shift output one element right. This is necessary for perfect reconstruction.\n"
"\n"
"        // i = N-1; even element goes to output[O-1], odd element goes to output[0]\n"
"        int j = 0;\n"
"        while(j <= start-1){\n"
"            int k;\n"
"            for (k = 0; k < N && j <= start-1; ++k, ++j){\n"
"                output[2*N-1] += filter[2*(start-1-j)] * input[k];\n"
"                output[0] += filter[2*(start-1-j)+1] * input[k];\n"
"            }\n"
"        }\n"
"        for (; j <= N+start-1 && j < F/2; ++j){\n"
"            output[2*N-1] += filter[2*j] * input[N+start-1-j];\n"
"            output[0] += filter[2*j+1] * input[N+start-1-j];\n"
"        }\n"
"        while (j < F / 2){\n"
"            int k;\n"
"            for (k = 0; k < N && j < F/2; ++k, ++j){\n"
"                output[2*N-1] += filter[2*j] * input[N-1-k];\n"
"                output[0] += filter[2*j+1] * input[N-1-k];\n"
"            }\n"
"        }\n"
"\n"
"        o += 1;\n"
"    }\n"
"\n"
"    for (; i < F/2 && i < N; ++i, o += 2){\n"
"        int j = 0;\n"
"        double sum_even=0.0; double sum_odd=0.0;\n"
"        for(; j <= i; ++j){\n"
"            sum_even += filter[2*j] * input[i-j];\n"
"            sum_odd += filter[2*j+1] * input[i-j];\n"
"        }\n"
"        while (j < F/2){\n"
"            int k;\n"
"            for(k = 0; k < N && j < F/2; ++k, ++j){\n"
"                sum_even += filter[2*j] * input[N-1-k];\n"
"                sum_odd += filter[2*j+1] * input[N-1-k];\n"
"            }\n"
"        }\n"
"        output[o] += sum_even;\n"
"        output[o+1] += sum_odd;\n"
"    }\n"
"\n"
"    for (; i < N; ++i, o += 2){\n"
"        double sum_even=0.0; double sum_odd=0.0;\n"
"        for(int j = 0; j < F/2; ++j){\n"
"            sum_even += filter[2*j] * input[i-j];\n"
"            sum_odd += filter[2*j+1] * input[i-j];\n"
"        }\n"
"        output[o] += sum_even;\n"
"        output[o+1] += sum_odd;\n"
"    }\n"
"\n"
"    for (; i < F/2 && i < end; ++i, o += 2){\n"
"        int j = 0;\n"
"        double sum_even=0.0; double sum_odd=0.0;\n"
"        while(i-j >= N){\n"
"            int k;\n"
"            for (k = 0; k < N && i-j >= N; ++k, ++j){\n"
"                sum_even += filter[2*(i-N-j)] * input[k];\n"
"                sum_odd += filter[2*(i-N-j)+1] * input[k];\n"
"            }\n"
"        }\n"
"        for (; j <= i && j < F/2; ++j){\n"
"            sum_even += filter[2*j] * input[i-j];\n"
"            sum_odd += filter[2*j+1] * input[i-j];\n"
"        }\n"
"        while (j < F / 2){\n"
"            int k;\n"
"            for (k = 0; k < N && j < F/2; ++k, ++j){\n"
"                sum_even += filter[2*j] * input[N-1-k];\n"
"                sum_odd += filter[2*j+1] * input[N-1-k];\n"
"            }\n"
"        }\n"
"        output[o] += sum_even;\n"
"        output[o+1] += sum_odd;\n"
"    }\n"
"\n"
"    for (; i < end; ++i, o += 2){\n"
"        int j = 0;\n"
"        double sum_even=0.0; double sum_odd=0.0;\n"
"        while(i-j >= N){\n"
"            int k;\n"
"            for (k = 0; k < N && i-j >= N; ++k, ++j){\n"
"                sum_even += filter[2*(i-N-j)] * input[k];\n"
"                sum_odd += filter[2*(i-N-j)+1] * input[k];\n"
"            }\n"
"        }\n"
"        for (; j <= i && j < F/2; ++j){\n"
"            sum_even += filter[2*j] * input[i-j];\n"
"            sum_odd += filter[2*j+1] * input[i-j];\n"
"        }\n"
"        output[o] += sum_even;\n"
"        output[o+1] += sum_odd;\n"
"    }\n"
"\n"
"    return 0;\n"
"}\n"
"\n"
"int upsampling_convolution_valid_sf(const __global double* input, int N,\n"
"                                    const __global double* filter, int F,\n"
"                                    __global double* output, int mode)\n"
"{\n"
"    if(mode == MODE_PERIODIZATION)\n"
"        return upsampling_convolution_valid_sf_periodization(input, N, filter, F, output);\n"
"\n"
"    if((F%2) || (N < F/2))\n"
"        return -1;\n"
"\n"
"    // output length should be 2*N - F + 2\n"
"    int o = 0;\n"
"    int i = F/2 - 1;\n"
"    int O = 2*N - F + 2;\n"
"    for(o=0; o<O; ++o) output[o]=0.0;\n"
"\n"
"    for(o = 0, i = F/2 - 1; i < N; ++i, o += 2){\n"
"        double sum_even = 0.0;\n"
"        double sum_odd = 0.0;\n"
"        for(int j = 0; j < F/2; ++j){\n"
"            sum_even += filter[j*2] * input[i-j];\n"
"            sum_odd  += filter[j*2+1] * input[i-j];\n"
"        }\n"
"        output[o] = sum_even;\n"
"        output[o+1] = sum_odd;\n"
"    }\n"
"\n"
"    return 0;\n"
"}\n"
"\n"
"__kernel void downconv_kernel(__global const double* input, int N,\n"
"                              __global const double* filter, int F,\n"
"                              int step, int mode,\n"
"                              __global double* output)\n"
"{\n"
"    if(get_global_id(0) != 0) return;\n"
"    downsampling_convolution(input, N, filter, F, output, step, mode);\n"
"}\n"
"\n"
"__kernel void downconv_per_kernel(__global const double* input, int N,\n"
"                                  __global const double* filter, int F,\n"
"                                  int step, int fstep,\n"
"                                  __global double* output)\n"
"{\n"
"    if(get_global_id(0) != 0) return;\n"
"    downsampling_convolution_periodization(input, N, filter, F, output, step, fstep);\n"
"}\n"
"\n"
"__kernel void upconv_full_kernel(__global const double* input, int N,\n"
"                                 __global const double* filter, int F,\n"
"                                 __global double* output, int O)\n"
"{\n"
"    if(get_global_id(0) != 0) return;\n"
"    upsampling_convolution_full(input, N, filter, F, output, O);\n"
"}\n"
"\n"
"__kernel void upconv_valid_kernel(__global const double* input, int N,\n"
"                                  __global const double* filter, int F,\n"
"                                  __global double* output, int mode)\n"
"{\n"
"    if(get_global_id(0) != 0) return;\n"
"    upsampling_convolution_valid_sf(input, N, filter, F, output, mode);\n"
"}\n"
"\n"
"__kernel void upconv_valid_per_kernel(__global const double* input, int N,\n"
"                                      __global const double* filter, int F,\n"
"                                      __global double* output)\n"
"{\n"
"    if(get_global_id(0) != 0) return;\n"
"    upsampling_convolution_valid_sf_periodization(input, N, filter, F, output);\n"
"}\n";
   h.prog=CLProgramCreate(h.ctx,code);
   if(h.prog==INVALID_HANDLE) { PyWTConvFree(h); return false; }
   h.kern_down=CLKernelCreate(h.prog,"downconv_kernel");
   h.kern_down_per=CLKernelCreate(h.prog,"downconv_per_kernel");
   h.kern_up_full=CLKernelCreate(h.prog,"upconv_full_kernel");
   h.kern_up_valid=CLKernelCreate(h.prog,"upconv_valid_kernel");
   h.kern_up_valid_per=CLKernelCreate(h.prog,"upconv_valid_per_kernel");
   if(h.kern_down==INVALID_HANDLE || h.kern_down_per==INVALID_HANDLE || h.kern_up_full==INVALID_HANDLE || h.kern_up_valid==INVALID_HANDLE || h.kern_up_valid_per==INVALID_HANDLE)
     { PyWTConvFree(h); return false; }
   h.ready=true;
   return true;
  }

inline bool PyWT_DownsamplingConvolution(const double &in[],const double &filter[],const int step,const int mode,double &output[])
  {
   int N=ArraySize(in);
   int F=ArraySize(filter);
   if(N<=0 || F<=0 || step<=0) return false;
   int use_mode=mode;
   if(use_mode==PYWT_MODE_SMOOTH && N<2) use_mode=PYWT_MODE_CONSTANT_EDGE;
   if(use_mode==PYWT_MODE_PERIODIZATION)
     {
      return PyWT_DownsamplingConvolutionPeriodization(in,filter,step,1,output);
     }
   int O=(N + F - 1)/step;
   ArrayResize(output,O);
   static PyWTConvHandle h; if(!h.ready) PyWTConvReset(h);
   if(!PyWTConvInit(h)) return false;
   if(h.memIn!=INVALID_HANDLE) CLBufferFree(h.memIn);
   if(h.memF!=INVALID_HANDLE) CLBufferFree(h.memF);
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memIn=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   h.memF=CLBufferCreate(h.ctx,F*sizeof(double),CL_MEM_READ_ONLY);
   h.memOut=CLBufferCreate(h.ctx,O*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memIn==INVALID_HANDLE || h.memF==INVALID_HANDLE || h.memOut==INVALID_HANDLE) return false;
   CLBufferWrite(h.memIn,in);
   CLBufferWrite(h.memF,filter);
   CLSetKernelArgMem(h.kern_down,0,h.memIn);
   CLSetKernelArg(h.kern_down,1,N);
   CLSetKernelArgMem(h.kern_down,2,h.memF);
   CLSetKernelArg(h.kern_down,3,F);
   CLSetKernelArg(h.kern_down,4,step);
   CLSetKernelArg(h.kern_down,5,use_mode);
   CLSetKernelArgMem(h.kern_down,6,h.memOut);
   uint offs[1]={0}; uint work[1]={1};
   if(!CLExecute(h.kern_down,1,offs,work)) return false;
   CLBufferRead(h.memOut,output);
   return true;
  }

inline bool PyWT_DownsamplingConvolutionPeriodization(const double &in[],const double &filter[],const int step,const int fstep,double &output[])
  {
   int N=ArraySize(in);
   int F=ArraySize(filter);
   if(N<=0 || F<=0 || step<=0 || fstep<=0) return false;
   int O=(N + step - 1)/step; // ceil
   ArrayResize(output,O);
   static PyWTConvHandle h; if(!h.ready) PyWTConvReset(h);
   if(!PyWTConvInit(h)) return false;
   if(h.memIn!=INVALID_HANDLE) CLBufferFree(h.memIn);
   if(h.memF!=INVALID_HANDLE) CLBufferFree(h.memF);
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memIn=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   h.memF=CLBufferCreate(h.ctx,F*sizeof(double),CL_MEM_READ_ONLY);
   h.memOut=CLBufferCreate(h.ctx,O*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memIn==INVALID_HANDLE || h.memF==INVALID_HANDLE || h.memOut==INVALID_HANDLE) return false;
   CLBufferWrite(h.memIn,in);
   CLBufferWrite(h.memF,filter);
   CLSetKernelArgMem(h.kern_down_per,0,h.memIn);
   CLSetKernelArg(h.kern_down_per,1,N);
   CLSetKernelArgMem(h.kern_down_per,2,h.memF);
   CLSetKernelArg(h.kern_down_per,3,F);
   CLSetKernelArg(h.kern_down_per,4,step);
   CLSetKernelArg(h.kern_down_per,5,fstep);
   CLSetKernelArgMem(h.kern_down_per,6,h.memOut);
   uint offs[1]={0}; uint work[1]={1};
   if(!CLExecute(h.kern_down_per,1,offs,work)) return false;
   CLBufferRead(h.memOut,output);
   return true;
  }

inline bool PyWT_UpsamplingConvolutionFull(const double &in[],const double &filter[],double &output[])
  {
   int N=ArraySize(in);
   int F=ArraySize(filter);
   if(N<=0 || F<=0) return false;
   if(F<2 || (F%2)!=0) return false;
   int O=2*N + F - 2;
   ArrayResize(output,O);
   static PyWTConvHandle h; if(!h.ready) PyWTConvReset(h);
   if(!PyWTConvInit(h)) return false;
   if(h.memIn!=INVALID_HANDLE) CLBufferFree(h.memIn);
   if(h.memF!=INVALID_HANDLE) CLBufferFree(h.memF);
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memIn=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   h.memF=CLBufferCreate(h.ctx,F*sizeof(double),CL_MEM_READ_ONLY);
   h.memOut=CLBufferCreate(h.ctx,O*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memIn==INVALID_HANDLE || h.memF==INVALID_HANDLE || h.memOut==INVALID_HANDLE) return false;
   CLBufferWrite(h.memIn,in);
   CLBufferWrite(h.memF,filter);
   CLSetKernelArgMem(h.kern_up_full,0,h.memIn);
   CLSetKernelArg(h.kern_up_full,1,N);
   CLSetKernelArgMem(h.kern_up_full,2,h.memF);
   CLSetKernelArg(h.kern_up_full,3,F);
   CLSetKernelArgMem(h.kern_up_full,4,h.memOut);
   CLSetKernelArg(h.kern_up_full,5,O);
   uint offs[1]={0}; uint work[1]={1};
   if(!CLExecute(h.kern_up_full,1,offs,work)) return false;
   CLBufferRead(h.memOut,output);
   return true;
  }

inline bool PyWT_UpsamplingConvolutionValidSF(const double &in[],const double &filter[],const int mode,double &output[])
  {
   int N=ArraySize(in);
   int F=ArraySize(filter);
   if(N<=0 || F<=0) return false;
   if(mode==PYWT_MODE_PERIODIZATION)
     {
      if((F%2)!=0) return false;
     }
   else
     {
      if((F%2)!=0 || N < F/2) return false;
     }
   int O=(mode==PYWT_MODE_PERIODIZATION) ? (2*N) : (2*N - F + 2);
   ArrayResize(output,O);
   static PyWTConvHandle h; if(!h.ready) PyWTConvReset(h);
   if(!PyWTConvInit(h)) return false;
   if(h.memIn!=INVALID_HANDLE) CLBufferFree(h.memIn);
   if(h.memF!=INVALID_HANDLE) CLBufferFree(h.memF);
   if(h.memOut!=INVALID_HANDLE) CLBufferFree(h.memOut);
   h.memIn=CLBufferCreate(h.ctx,N*sizeof(double),CL_MEM_READ_ONLY);
   h.memF=CLBufferCreate(h.ctx,F*sizeof(double),CL_MEM_READ_ONLY);
   h.memOut=CLBufferCreate(h.ctx,O*sizeof(double),CL_MEM_WRITE_ONLY);
   if(h.memIn==INVALID_HANDLE || h.memF==INVALID_HANDLE || h.memOut==INVALID_HANDLE) return false;
   CLBufferWrite(h.memIn,in);
   CLBufferWrite(h.memF,filter);
   if(mode==PYWT_MODE_PERIODIZATION)
     {
      CLSetKernelArgMem(h.kern_up_valid_per,0,h.memIn);
      CLSetKernelArg(h.kern_up_valid_per,1,N);
      CLSetKernelArgMem(h.kern_up_valid_per,2,h.memF);
      CLSetKernelArg(h.kern_up_valid_per,3,F);
      CLSetKernelArgMem(h.kern_up_valid_per,4,h.memOut);
      uint offs[1]={0}; uint work[1]={1};
      if(!CLExecute(h.kern_up_valid_per,1,offs,work)) return false;
     }
   else
     {
      CLSetKernelArgMem(h.kern_up_valid,0,h.memIn);
      CLSetKernelArg(h.kern_up_valid,1,N);
      CLSetKernelArgMem(h.kern_up_valid,2,h.memF);
      CLSetKernelArg(h.kern_up_valid,3,F);
      CLSetKernelArgMem(h.kern_up_valid,4,h.memOut);
      CLSetKernelArg(h.kern_up_valid,5,mode);
      uint offs[1]={0}; uint work[1]={1};
      if(!CLExecute(h.kern_up_valid,1,offs,work)) return false;
     }
   CLBufferRead(h.memOut,output);
   return true;
  }

#endif // __SPECTRAL_PYWT_CONV_MQH__
