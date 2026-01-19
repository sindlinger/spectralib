#ifndef __SPECTRAL_OPENCL_FFT_MQH__
#define __SPECTRAL_OPENCL_FFT_MQH__

#include "SpectralCommon.mqh"
#include "SpectralOpenCLCommon.mqh"

struct CLFFTPlan
  {
   int ctx;
   int prog;
   int kern_bitrev;
   int kern_stage;
   int kern_scale;
   int kern_dft;
   int kern_load;
   int kern_sum;
   int kern_load_dt;
   int kern_bitrev_b;
   int kern_stage_b;
   int kern_scale_b;
   int kern_dft_b;
   int kern_load_b;
   int kern_sum_b;
   int kern_load_dt_b;
   int kern_overlap;
   int kern_norm;
   int kern_win_stats;
   int kern_gen_freqs;
   int kern_gen_times;
   int kern_expand_onesided;
   int kern_copy_slice;
   int kern_time_linear;
   int kern_psd_cmul;
   int kern_psd_onesided;
   int kern_psd_mean;
   int kern_psd_median;
   int kern_pack_segments;
   int kern_cabs;
   int kern_carg;
   int kern_unwrap;
   int kern_cola_bins;
   int kern_nola_bins;
   int kern_cola_check;
   int kern_nola_check;
   int kern_coherence;
   int kern_maxmag;
   int kern_copy;
   int kern_hilbert;
   int memA;
   int memB;
   int memX;
   int memWin;
   int memSum;
   int memOutX;
   int memOutNorm;
   int memWinStat;
   int memOutReal;
   int memHalf;
   int memCrop;
   int memPSD;
   int memAvg;
   int memPack;
   int memFinal;
   int memBins;
   int memCheck;
   int memCohX;
   int memCohPxx;
   int memCohPyy;
   int memCohOut;
   int batch;
   int N;
   int lenX;
   int lenWin;
   int lenOutReal;
   int lenHalf;
   int lenCrop;
   int lenPSD;
   int lenAvg;
   int lenPack;
   int lenBins;
   int lenCoh;
   bool ready;
  };

inline void CLFFTReset(CLFFTPlan &p)
  {
   p.ctx=INVALID_HANDLE;
   p.prog=INVALID_HANDLE;
   p.kern_bitrev=INVALID_HANDLE;
   p.kern_stage=INVALID_HANDLE;
   p.kern_scale=INVALID_HANDLE;
   p.kern_dft=INVALID_HANDLE;
   p.kern_load=INVALID_HANDLE;
   p.kern_sum=INVALID_HANDLE;
   p.kern_load_dt=INVALID_HANDLE;
   p.kern_bitrev_b=INVALID_HANDLE;
   p.kern_stage_b=INVALID_HANDLE;
   p.kern_scale_b=INVALID_HANDLE;
   p.kern_dft_b=INVALID_HANDLE;
   p.kern_load_b=INVALID_HANDLE;
   p.kern_sum_b=INVALID_HANDLE;
   p.kern_load_dt_b=INVALID_HANDLE;
   p.kern_overlap=INVALID_HANDLE;
   p.kern_norm=INVALID_HANDLE;
   p.kern_win_stats=INVALID_HANDLE;
   p.kern_gen_freqs=INVALID_HANDLE;
   p.kern_gen_times=INVALID_HANDLE;
   p.kern_expand_onesided=INVALID_HANDLE;
   p.kern_copy_slice=INVALID_HANDLE;
   p.kern_time_linear=INVALID_HANDLE;
   p.kern_psd_cmul=INVALID_HANDLE;
   p.kern_psd_onesided=INVALID_HANDLE;
   p.kern_psd_mean=INVALID_HANDLE;
   p.kern_psd_median=INVALID_HANDLE;
   p.kern_pack_segments=INVALID_HANDLE;
   p.kern_cabs=INVALID_HANDLE;
   p.kern_carg=INVALID_HANDLE;
   p.kern_unwrap=INVALID_HANDLE;
   p.kern_cola_bins=INVALID_HANDLE;
   p.kern_nola_bins=INVALID_HANDLE;
   p.kern_cola_check=INVALID_HANDLE;
   p.kern_nola_check=INVALID_HANDLE;
   p.kern_coherence=INVALID_HANDLE;
   p.kern_maxmag=INVALID_HANDLE;
   p.kern_copy=INVALID_HANDLE;
   p.kern_hilbert=INVALID_HANDLE;
   p.memA=INVALID_HANDLE;
   p.memB=INVALID_HANDLE;
   p.memX=INVALID_HANDLE;
   p.memWin=INVALID_HANDLE;
   p.memSum=INVALID_HANDLE;
   p.memOutX=INVALID_HANDLE;
   p.memOutNorm=INVALID_HANDLE;
   p.memWinStat=INVALID_HANDLE;
   p.memOutReal=INVALID_HANDLE;
   p.memHalf=INVALID_HANDLE;
   p.memCrop=INVALID_HANDLE;
   p.memPSD=INVALID_HANDLE;
   p.memAvg=INVALID_HANDLE;
   p.memPack=INVALID_HANDLE;
   p.memFinal=INVALID_HANDLE;
   p.memBins=INVALID_HANDLE;
   p.memCheck=INVALID_HANDLE;
   p.memCohX=INVALID_HANDLE;
   p.memCohPxx=INVALID_HANDLE;
   p.memCohPyy=INVALID_HANDLE;
   p.memCohOut=INVALID_HANDLE;
   p.batch=1;
   p.N=0;
   p.lenX=0;
   p.lenWin=0;
   p.lenOutReal=0;
   p.lenHalf=0;
   p.lenCrop=0;
   p.lenPSD=0;
   p.lenAvg=0;
   p.lenPack=0;
   p.lenBins=0;
   p.lenCoh=0;
   p.ready=false;
  }

inline void CLFFTFree(CLFFTPlan &p)
  {
   if(p.memA!=INVALID_HANDLE) { CLBufferFree(p.memA); p.memA=INVALID_HANDLE; }
   if(p.memB!=INVALID_HANDLE) { CLBufferFree(p.memB); p.memB=INVALID_HANDLE; }
   if(p.memX!=INVALID_HANDLE) { CLBufferFree(p.memX); p.memX=INVALID_HANDLE; }
   if(p.memWin!=INVALID_HANDLE) { CLBufferFree(p.memWin); p.memWin=INVALID_HANDLE; }
   if(p.memSum!=INVALID_HANDLE) { CLBufferFree(p.memSum); p.memSum=INVALID_HANDLE; }
   if(p.memOutX!=INVALID_HANDLE) { CLBufferFree(p.memOutX); p.memOutX=INVALID_HANDLE; }
   if(p.memOutNorm!=INVALID_HANDLE) { CLBufferFree(p.memOutNorm); p.memOutNorm=INVALID_HANDLE; }
   if(p.memWinStat!=INVALID_HANDLE) { CLBufferFree(p.memWinStat); p.memWinStat=INVALID_HANDLE; }
   if(p.memOutReal!=INVALID_HANDLE) { CLBufferFree(p.memOutReal); p.memOutReal=INVALID_HANDLE; }
   if(p.memHalf!=INVALID_HANDLE) { CLBufferFree(p.memHalf); p.memHalf=INVALID_HANDLE; }
   if(p.memCrop!=INVALID_HANDLE) { CLBufferFree(p.memCrop); p.memCrop=INVALID_HANDLE; }
   if(p.memPSD!=INVALID_HANDLE) { CLBufferFree(p.memPSD); p.memPSD=INVALID_HANDLE; }
   if(p.memAvg!=INVALID_HANDLE) { CLBufferFree(p.memAvg); p.memAvg=INVALID_HANDLE; }
   if(p.memPack!=INVALID_HANDLE) { CLBufferFree(p.memPack); p.memPack=INVALID_HANDLE; }
   if(p.memBins!=INVALID_HANDLE) { CLBufferFree(p.memBins); p.memBins=INVALID_HANDLE; }
   if(p.memCheck!=INVALID_HANDLE) { CLBufferFree(p.memCheck); p.memCheck=INVALID_HANDLE; }
   if(p.memCohX!=INVALID_HANDLE) { CLBufferFree(p.memCohX); p.memCohX=INVALID_HANDLE; }
   if(p.memCohPxx!=INVALID_HANDLE) { CLBufferFree(p.memCohPxx); p.memCohPxx=INVALID_HANDLE; }
   if(p.memCohPyy!=INVALID_HANDLE) { CLBufferFree(p.memCohPyy); p.memCohPyy=INVALID_HANDLE; }
   if(p.memCohOut!=INVALID_HANDLE) { CLBufferFree(p.memCohOut); p.memCohOut=INVALID_HANDLE; }
   p.memFinal=INVALID_HANDLE;
   if(p.kern_bitrev!=INVALID_HANDLE) { CLKernelFree(p.kern_bitrev); p.kern_bitrev=INVALID_HANDLE; }
   if(p.kern_stage!=INVALID_HANDLE) { CLKernelFree(p.kern_stage); p.kern_stage=INVALID_HANDLE; }
   if(p.kern_scale!=INVALID_HANDLE) { CLKernelFree(p.kern_scale); p.kern_scale=INVALID_HANDLE; }
   if(p.kern_dft!=INVALID_HANDLE) { CLKernelFree(p.kern_dft); p.kern_dft=INVALID_HANDLE; }
   if(p.kern_load!=INVALID_HANDLE) { CLKernelFree(p.kern_load); p.kern_load=INVALID_HANDLE; }
   if(p.kern_sum!=INVALID_HANDLE) { CLKernelFree(p.kern_sum); p.kern_sum=INVALID_HANDLE; }
   if(p.kern_load_dt!=INVALID_HANDLE) { CLKernelFree(p.kern_load_dt); p.kern_load_dt=INVALID_HANDLE; }
   if(p.kern_bitrev_b!=INVALID_HANDLE) { CLKernelFree(p.kern_bitrev_b); p.kern_bitrev_b=INVALID_HANDLE; }
   if(p.kern_stage_b!=INVALID_HANDLE) { CLKernelFree(p.kern_stage_b); p.kern_stage_b=INVALID_HANDLE; }
   if(p.kern_scale_b!=INVALID_HANDLE) { CLKernelFree(p.kern_scale_b); p.kern_scale_b=INVALID_HANDLE; }
   if(p.kern_dft_b!=INVALID_HANDLE) { CLKernelFree(p.kern_dft_b); p.kern_dft_b=INVALID_HANDLE; }
   if(p.kern_load_b!=INVALID_HANDLE) { CLKernelFree(p.kern_load_b); p.kern_load_b=INVALID_HANDLE; }
   if(p.kern_sum_b!=INVALID_HANDLE) { CLKernelFree(p.kern_sum_b); p.kern_sum_b=INVALID_HANDLE; }
   if(p.kern_load_dt_b!=INVALID_HANDLE) { CLKernelFree(p.kern_load_dt_b); p.kern_load_dt_b=INVALID_HANDLE; }
   if(p.kern_overlap!=INVALID_HANDLE) { CLKernelFree(p.kern_overlap); p.kern_overlap=INVALID_HANDLE; }
   if(p.kern_norm!=INVALID_HANDLE) { CLKernelFree(p.kern_norm); p.kern_norm=INVALID_HANDLE; }
   if(p.kern_win_stats!=INVALID_HANDLE) { CLKernelFree(p.kern_win_stats); p.kern_win_stats=INVALID_HANDLE; }
   if(p.kern_gen_freqs!=INVALID_HANDLE) { CLKernelFree(p.kern_gen_freqs); p.kern_gen_freqs=INVALID_HANDLE; }
   if(p.kern_gen_times!=INVALID_HANDLE) { CLKernelFree(p.kern_gen_times); p.kern_gen_times=INVALID_HANDLE; }
   if(p.kern_expand_onesided!=INVALID_HANDLE) { CLKernelFree(p.kern_expand_onesided); p.kern_expand_onesided=INVALID_HANDLE; }
   if(p.kern_copy_slice!=INVALID_HANDLE) { CLKernelFree(p.kern_copy_slice); p.kern_copy_slice=INVALID_HANDLE; }
   if(p.kern_time_linear!=INVALID_HANDLE) { CLKernelFree(p.kern_time_linear); p.kern_time_linear=INVALID_HANDLE; }
   if(p.kern_psd_cmul!=INVALID_HANDLE) { CLKernelFree(p.kern_psd_cmul); p.kern_psd_cmul=INVALID_HANDLE; }
   if(p.kern_psd_onesided!=INVALID_HANDLE) { CLKernelFree(p.kern_psd_onesided); p.kern_psd_onesided=INVALID_HANDLE; }
   if(p.kern_psd_mean!=INVALID_HANDLE) { CLKernelFree(p.kern_psd_mean); p.kern_psd_mean=INVALID_HANDLE; }
   if(p.kern_psd_median!=INVALID_HANDLE) { CLKernelFree(p.kern_psd_median); p.kern_psd_median=INVALID_HANDLE; }
   if(p.kern_pack_segments!=INVALID_HANDLE) { CLKernelFree(p.kern_pack_segments); p.kern_pack_segments=INVALID_HANDLE; }
   if(p.kern_cabs!=INVALID_HANDLE) { CLKernelFree(p.kern_cabs); p.kern_cabs=INVALID_HANDLE; }
   if(p.kern_carg!=INVALID_HANDLE) { CLKernelFree(p.kern_carg); p.kern_carg=INVALID_HANDLE; }
   if(p.kern_unwrap!=INVALID_HANDLE) { CLKernelFree(p.kern_unwrap); p.kern_unwrap=INVALID_HANDLE; }
   if(p.kern_cola_bins!=INVALID_HANDLE) { CLKernelFree(p.kern_cola_bins); p.kern_cola_bins=INVALID_HANDLE; }
   if(p.kern_nola_bins!=INVALID_HANDLE) { CLKernelFree(p.kern_nola_bins); p.kern_nola_bins=INVALID_HANDLE; }
   if(p.kern_cola_check!=INVALID_HANDLE) { CLKernelFree(p.kern_cola_check); p.kern_cola_check=INVALID_HANDLE; }
   if(p.kern_nola_check!=INVALID_HANDLE) { CLKernelFree(p.kern_nola_check); p.kern_nola_check=INVALID_HANDLE; }
   if(p.kern_coherence!=INVALID_HANDLE) { CLKernelFree(p.kern_coherence); p.kern_coherence=INVALID_HANDLE; }
   if(p.kern_maxmag!=INVALID_HANDLE) { CLKernelFree(p.kern_maxmag); p.kern_maxmag=INVALID_HANDLE; }
   if(p.kern_copy!=INVALID_HANDLE) { CLKernelFree(p.kern_copy); p.kern_copy=INVALID_HANDLE; }
   if(p.kern_hilbert!=INVALID_HANDLE) { CLKernelFree(p.kern_hilbert); p.kern_hilbert=INVALID_HANDLE; }
   if(p.prog!=INVALID_HANDLE) { CLProgramFree(p.prog); p.prog=INVALID_HANDLE; }
   if(p.ctx!=INVALID_HANDLE) { CLContextFree(p.ctx); p.ctx=INVALID_HANDLE; }
   p.N=0; p.batch=1; p.ready=false;
   p.lenX=0; p.lenWin=0;
   p.lenOutReal=0;
   p.lenHalf=0;
   p.lenCrop=0;
   p.lenPSD=0;
   p.lenAvg=0;
   p.lenPack=0;
   p.lenBins=0;
   p.lenCoh=0;
  }

inline bool CLFFTInit(CLFFTPlan &p,const int N)
  {
   static bool log_ctx=false;
   static bool log_prog=false;
   static bool log_kern=false;
   static bool log_mem=false;
   if(p.ready && p.N==N) return true;
   CLFFTFree(p);
   CLFFTReset(p);
   if(N<=1) return false;
   p.ctx=CLCreateContextGPUFloat64("SpectralOpenCLFFT");
   if(p.ctx==INVALID_HANDLE)
     {
      if(!log_ctx)
        {
         PrintFormat("SpectralOpenCLFFT: CLContextCreate failed (err=%d)", GetLastError());
         log_ctx=true;
        }
      return false;
     }

   string code=
   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"#ifndef M_PI\n"
"#define M_PI 3.1415926535897932384626433832795\n"
"#endif\n"
   "inline uint bitrev(uint x, uint bits){\n"
   "  uint y=0; for(uint i=0;i<bits;i++){ y=(y<<1) | (x & 1); x>>=1; } return y; }\n"
   "__kernel void bit_reverse(__global const double2* in, __global double2* out, int N, int bits){\n"
   "  int i=get_global_id(0); if(i>=N) return; uint r=bitrev((uint)i,(uint)bits); out[r]=in[i]; }\n"
   "__kernel void bit_reverse_batch(__global const double2* in, __global double2* out, int N, int bits){\n"
   "  int gid=get_global_id(0); int seg=gid / N; int i=gid - seg*N; if(i>=N) return;\n"
   "  uint r=bitrev((uint)i,(uint)bits); out[seg*N + r]=in[seg*N + i]; }\n"
   "__kernel void fft_stage(__global const double2* in, __global double2* out, int N, int m, int inverse){\n"
   "  int i=get_global_id(0); int hlf=m>>1; int total=N>>1; if(i>=total) return;\n"
   "  int j=i%hlf; int block=i/hlf; int k=block*m + j;\n"
   "  double angle = (inverse? 2.0 : -2.0) * M_PI * (double)j / (double)m;\n"
   "  double c=cos(angle); double s=sin(angle);\n"
   "  double2 a=in[k]; double2 b=in[k+hlf];\n"
   "  double2 t = (double2)(b.x*c - b.y*s, b.x*s + b.y*c);\n"
   "  out[k] = (double2)(a.x + t.x, a.y + t.y);\n"
   "  out[k+hlf] = (double2)(a.x - t.x, a.y - t.y);\n"
   "}\n"
   "__kernel void fft_stage_batch(__global const double2* in, __global double2* out, int N, int m, int inverse){\n"
   "  int gid=get_global_id(0); int hlf=m>>1; int total=N>>1; int seg=gid / total; int i=gid - seg*total; if(i>=total) return;\n"
   "  int j=i%hlf; int block=i/hlf; int k=block*m + j; int base=seg*N;\n"
   "  double angle = (inverse? 2.0 : -2.0) * M_PI * (double)j / (double)m;\n"
   "  double c=cos(angle); double s=sin(angle);\n"
   "  double2 a=in[base + k]; double2 b=in[base + k + hlf];\n"
   "  double2 t = (double2)(b.x*c - b.y*s, b.x*s + b.y*c);\n"
   "  out[base + k] = (double2)(a.x + t.x, a.y + t.y);\n"
   "  out[base + k + hlf] = (double2)(a.x - t.x, a.y - t.y);\n"
   "}\n"
   "__kernel void fft_scale(__global double2* data, int N, double invN){\n"
   "  int i=get_global_id(0); if(i>=N) return; data[i].x*=invN; data[i].y*=invN; }\n"
   "__kernel void fft_scale_batch(__global double2* data, int N, double invN){\n"
   "  int gid=get_global_id(0); int i=gid; if(i>=N) return; data[i].x*=invN; data[i].y*=invN; }\n"
   "__kernel void dft_complex(__global const double2* in, __global double2* out, int N, int inverse){\n"
   "  int k=get_global_id(0); if(k>=N) return; double sign = (inverse!=0)? 1.0 : -1.0;\n"
   "  double2 sum=(double2)(0.0,0.0);\n"
   "  for(int n=0;n<N;n++){\n"
   "    double ang = sign * 2.0 * M_PI * ((double)k * (double)n) / (double)N;\n"
   "    double c=cos(ang); double s=sin(ang);\n"
   "    double2 v=in[n]; sum.x += v.x*c - v.y*s; sum.y += v.x*s + v.y*c;\n"
   "  }\n"
   "  if(inverse!=0){ sum.x/= (double)N; sum.y/=(double)N; }\n"
   "  out[k]=sum; }\n"
   "__kernel void dft_complex_batch(__global const double2* in, __global double2* out, int N, int inverse){\n"
   "  int gid=get_global_id(0); int seg=gid / N; int k=gid - seg*N; double sign = (inverse!=0)? 1.0 : -1.0;\n"
   "  double2 sum=(double2)(0.0,0.0);\n"
   "  for(int n=0;n<N;n++){\n"
   "    double ang = sign * 2.0 * M_PI * ((double)k * (double)n) / (double)N;\n"
   "    double c=cos(ang); double s=sin(ang);\n"
   "    double2 v=in[seg*N + n]; sum.x += v.x*c - v.y*s; sum.y += v.x*s + v.y*c;\n"
   "  }\n"
   "  if(inverse!=0){ sum.x/= (double)N; sum.y/=(double)N; }\n"
   "  out[seg*N + k]=sum; }\n"
   "inline double ext_val(__global const double* x, int N, int nedge, int btype, int ext_valid, int idx){\n"
   "  if(idx<0 || idx>=ext_valid) return 0.0;\n"
   "  if(btype==0 || nedge<=0) return x[idx];\n"
   "  if(idx>=nedge && idx<nedge+N) return x[idx-nedge];\n"
   "  if(idx<nedge){ int src=nedge-idx; if(src<0) src=0; if(src>=N) src=N-1;\n"
   "    if(btype==1) return x[src]; if(btype==2) return 2.0*x[0]-x[src]; if(btype==3) return x[0]; return 0.0; }\n"
   "  int i=idx-(nedge+N); int src=N-2-i; if(src<0) src=0; if(src>=N) src=N-1;\n"
   "  if(btype==1) return x[src]; if(btype==2) return 2.0*x[N-1]-x[src]; if(btype==3) return x[N-1]; return 0.0; }\n"
   "__kernel void load_real_segment(__global const double* x, __global const double* win, __global double2* out,\n"
   "  int xlen, int start, int nperseg, int nfft, int btype, int nedge, int ext_valid){\n"
   "  int i=get_global_id(0); if(i>=nfft) return; double v=0.0;\n"
   "  if(i<nperseg){ int idx=start+i; v = ext_val(x,xlen,nedge,btype,ext_valid,idx) * win[i]; }\n"
   "  out[i]=(double2)(v,0.0); }\n"
   "__kernel void load_real_segment_batch(__global const double* x, __global const double* win, __global double2* out,\n"
   "  int xlen, int start0, int step, int nperseg, int nfft, int btype, int nedge, int ext_valid){\n"
   "  int gid=get_global_id(0); int seg=gid / nfft; int i=gid - seg*nfft; double v=0.0;\n"
   "  int start = start0 + seg*step;\n"
   "  if(i<nperseg){ int idx=start+i; v = ext_val(x,xlen,nedge,btype,ext_valid,idx) * win[i]; }\n"
   "  out[seg*nfft + i]=(double2)(v,0.0); }\n"
   "__kernel void seg_sums(__global const double* x, int xlen, int start, int nperseg, int btype, int nedge, int ext_valid, __global double* sumout){\n"
   "  double sumx=0.0; double sumix=0.0; for(int i=0;i<nperseg;i++){\n"
   "    int idx=start+i; double v=ext_val(x,xlen,nedge,btype,ext_valid,idx); sumx+=v; sumix+=v*(double)i; }\n"
   "  sumout[0]=sumx; sumout[1]=sumix; }\n"
   "__kernel void seg_sums_batch(__global const double* x, int xlen, int start0, int step, int nperseg, int btype, int nedge, int ext_valid, __global double* sumout){\n"
   "  int seg=get_global_id(0); double sumx=0.0; double sumix=0.0; int start=start0 + seg*step;\n"
   "  for(int i=0;i<nperseg;i++){\n"
   "    int idx=start+i; double v=ext_val(x,xlen,nedge,btype,ext_valid,idx); sumx+=v; sumix+=v*(double)i; }\n"
   "  sumout[2*seg]=sumx; sumout[2*seg+1]=sumix; }\n"
   "__kernel void load_real_segment_detrend(__global const double* x, __global const double* win, __global const double* sumout,\n"
   "  int xlen, int start, int nperseg, int nfft, int detrend_type, double sum_i, double sum_i2, int btype, int nedge, int ext_valid, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=nfft) return; double v=0.0;\n"
   "  if(i<nperseg){ int idx=start+i; double xi=ext_val(x,xlen,nedge,btype,ext_valid,idx);\n"
   "    if(detrend_type==1){ double mean = sumout[0]/(double)nperseg; xi = xi - mean; }\n"
   "    else if(detrend_type==2){ double n=(double)nperseg; double denom = n*sum_i2 - sum_i*sum_i; double m=0.0;\n"
   "      if(denom!=0.0) m=(n*sumout[1] - sum_i*sumout[0])/denom; double b=(sumout[0]-m*sum_i)/n; xi = xi - (m*(double)i + b); }\n"
   "    v = xi*win[i]; } out[i]=(double2)(v,0.0); }\n"
   "__kernel void load_real_segment_detrend_batch(__global const double* x, __global const double* win, __global const double* sumout,\n"
   "  int xlen, int start0, int step, int nperseg, int nfft, int detrend_type, double sum_i, double sum_i2, int btype, int nedge, int ext_valid, __global double2* out){\n"
   "  int gid=get_global_id(0); int seg=gid / nfft; int i=gid - seg*nfft; double v=0.0; int start=start0 + seg*step;\n"
   "  if(i<nperseg){ int idx=start+i; double xi=ext_val(x,xlen,nedge,btype,ext_valid,idx);\n"
   "    double s0=sumout[2*seg]; double s1=sumout[2*seg+1];\n"
   "    if(detrend_type==1){ double mean = s0/(double)nperseg; xi = xi - mean; }\n"
   "    else if(detrend_type==2){ double n=(double)nperseg; double denom = n*sum_i2 - sum_i*sum_i; double m=0.0;\n"
   "      if(denom!=0.0) m=(n*s1 - sum_i*s0)/denom; double b=(s0-m*sum_i)/n; xi = xi - (m*(double)i + b); }\n"
   "    v = xi*win[i]; } out[seg*nfft + i]=(double2)(v,0.0); }\n"
   "__kernel void overlap_add_complex(__global const double2* seg, __global const double* win, int nseg, int nperseg, int nstep, int N, int outlen, double scale,\n"
   "  __global double2* out, __global double* norm){\n"
   "  int n=get_global_id(0); if(n>=outlen) return; double2 sum=(double2)(0.0,0.0); double ns=0.0;\n"
   "  int smin = (n - (nperseg-1) + nstep - 1) / nstep; if(smin<0) smin=0;\n"
   "  int smax = n / nstep; if(smax>nseg-1) smax=nseg-1;\n"
   "  for(int s=smin;s<=smax;s++){\n"
   "    int start=s*nstep; int idx=n-start; if(idx>=0 && idx<nperseg){ double w=win[idx]; double2 v=seg[s*N + idx];\n"
   "      sum.x += v.x*w*scale; sum.y += v.y*w*scale; ns += w*w; }\n"
   "  }\n"
   "  out[n]=sum; norm[n]=ns; }\n"
   "__kernel void overlap_normalize(__global double2* out, __global const double* norm, int n){\n"
   "  int i=get_global_id(0); if(i>=n) return; double d=norm[i]; if(d>1.0e-10){ out[i].x/=d; out[i].y/=d; } }\n"
   "__kernel void win_stats(__global const double* win, int n, __global double* out){\n"
   "  if(get_global_id(0)!=0) return; double s=0.0; double s2=0.0; for(int i=0;i<n;i++){ double v=win[i]; s+=v; s2+=v*v; }\n"
   "  out[0]=s; out[1]=s2; }\n"
   "__kernel void gen_freqs(int nfft, double fs, int onesided, __global double* out){\n"
   "  int k=get_global_id(0); int n = onesided!=0 ? (nfft/2+1) : nfft; if(k>=n) return;\n"
   "  if(onesided!=0){ out[k]= (double)k*fs/(double)nfft; }\n"
   "  else { int kk = (k<=nfft/2)?k:(k-nfft); out[k]= (double)kk*fs/(double)nfft; } }\n"
   "__kernel void gen_times(int nseg, int seglen, int noverlap, double fs, int boundary_type, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=nseg) return; double step=(double)(seglen-noverlap);\n"
   "  double t=((double)i*step + (double)seglen/2.0)/fs; if(boundary_type!=0){ double shift=(double)seglen/2.0/fs; t-=shift; } out[i]=t; }\n"
   "__kernel void expand_onesided(__global const double2* in, __global double2* out, int N, int nfreq, int kmax){\n"
   "  int gid=get_global_id(0); int seg=gid / nfreq; int k=gid - seg*nfreq; if(k>=nfreq) return; int base=seg*N;\n"
   "  double2 v=in[seg*nfreq + k]; out[base + k]=v; if(k>0 && k<=kmax){ out[base + (N-k)] = (double2)(v.x, -v.y); } }\n"
   "__kernel void copy_slice_cplx(__global const double2* in, int start, int n, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=n) return; out[i]=in[start+i]; }\n"
   "__kernel void copy_cplx(__global const double2* in, __global double2* out, int n){\n"
   "  int i=get_global_id(0); if(i>=n) return; out[i]=in[i]; }\n"
   "__kernel void hilbert_mask(__global double2* data, int N){\n"
   "  int k=get_global_id(0); if(k>=N) return; int hlf=N/2;\n"
   "  if((N & 1)==0){ if(k==0 || k==hlf) return; if(k<hlf){ data[k].x*=2.0; data[k].y*=2.0; }\n"
   "    else { data[k]=(double2)(0.0,0.0); } }\n"
   "  else { if(k==0) return; if(k<=hlf){ data[k].x*=2.0; data[k].y*=2.0; }\n"
   "    else { data[k]=(double2)(0.0,0.0); } }\n"
   "}\n"
   "__kernel void gen_time_linear(int n, double fs, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=n) return; out[i]=(double)i/fs; }\n"
   "__kernel void psd_cmul(__global const double2* X, __global const double2* Y, int nseg, int nfreq, __global double2* out){\n"
   "  int gid=get_global_id(0); int total=nseg*nfreq; if(gid>=total) return; double2 a=X[gid]; double2 b=Y[gid];\n"
   "  out[gid]=(double2)(a.x*b.x + a.y*b.y, a.x*b.y - a.y*b.x); }\n"
   "__kernel void psd_onesided(__global double2* data, int nseg, int nfreq, int last){\n"
   "  int gid=get_global_id(0); int total=nseg*nfreq; if(gid>=total) return; int k=gid - (gid/nfreq)*nfreq;\n"
   "  if(k>=1 && k<=last){ data[gid].x*=2.0; data[gid].y*=2.0; } }\n"
   "__kernel void psd_mean(__global const double2* data, int nseg, int nfreq, __global double2* out){\n"
   "  int k=get_global_id(0); if(k>=nfreq) return; double sr=0.0; double si=0.0; for(int s=0;s<nseg;s++){ double2 v=data[s*nfreq + k]; sr+=v.x; si+=v.y; }\n"
   "  double inv=1.0/(double)nseg; out[k]=(double2)(sr*inv, si*inv); }\n"
   "double select_val(__global const double2* data, int nseg, int nfreq, int k, int pos, int imag){\n"
   "  for(int s=0;s<nseg;s++){\n"
   "    double val = imag? data[s*nfreq + k].y : data[s*nfreq + k].x; int less=0; int greater=0;\n"
   "    for(int s2=0;s2<nseg;s2++){\n"
   "      double v2 = imag? data[s2*nfreq + k].y : data[s2*nfreq + k].x; if(v2<val) less++; else if(v2>val) greater++; }\n"
   "    if(less<=pos && greater<= (nseg-1 - pos)) return val; }\n"
   "  return 0.0; }\n"
   "__kernel void psd_median(__global const double2* data, int nseg, int nfreq, double bias, __global double2* out){\n"
   "  int k=get_global_id(0); if(k>=nfreq) return; int posH=nseg/2; int posL=(nseg-1)/2;\n"
   "  double mr = 0.5*(select_val(data,nseg,nfreq,k,posL,0) + select_val(data,nseg,nfreq,k,posH,0));\n"
   "  double mi = 0.5*(select_val(data,nseg,nfreq,k,posL,1) + select_val(data,nseg,nfreq,k,posH,1));\n"
   "  out[k]=(double2)(mr/bias, mi/bias); }\n"
   "__kernel void pack_segments(__global const double2* in, int nseg, int nfft, int nfreq, __global double2* out){\n"
   "  int gid=get_global_id(0); int total=nseg*nfreq; if(gid>=total) return; int s=gid/nfreq; int k=gid - s*nfreq;\n"
   "  out[gid]=in[s*nfft + k]; }\n"
   "__kernel void cabs_cplx(__global const double2* in, __global double* out, int n){\n"
   "  int i=get_global_id(0); if(i>=n) return; double2 v=in[i]; out[i]=sqrt(v.x*v.x + v.y*v.y); }\n"
   "__kernel void carg_cplx(__global const double2* in, __global double* out, int n){\n"
   "  int i=get_global_id(0); if(i>=n) return; double2 v=in[i]; out[i]=atan2(v.y,v.x); }\n"
   "__kernel void unwrap_phase_rows(__global double* phase, int nseg, int nfreq){\n"
   "  int seg=get_global_id(0); if(seg>=nseg) return; int base=seg*nfreq; double prev=phase[base];\n"
   "  for(int k=1;k<nfreq;k++){ double v=phase[base+k]; double dp=v-prev; while(dp> M_PI){ v-=2.0*M_PI; dp=v-prev; }\n"
   "    while(dp< -M_PI){ v+=2.0*M_PI; dp=v-prev; } phase[base+k]=v; prev=v; }\n"
   "}\n"
   "__kernel void cola_binsums(__global const double* win, int nperseg, int step, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=step) return; double sum=0.0; int nblocks=nperseg/step;\n"
   "  for(int b=0;b<nblocks;b++){ int idx=b*step + i; sum += win[idx]; }\n"
   "  int rem = nperseg % step; if(rem!=0 && i<rem){ sum += win[nperseg-rem + i]; }\n"
   "  out[i]=sum; }\n"
   "__kernel void nola_binsums(__global const double* win, int nperseg, int step, __global double* out){\n"
   "  int i=get_global_id(0); if(i>=step) return; double sum=0.0; int nblocks=nperseg/step;\n"
   "  for(int b=0;b<nblocks;b++){ int idx=b*step + i; double v=win[idx]; sum += v*v; }\n"
   "  int rem = nperseg % step; if(rem!=0 && i<rem){ double v=win[nperseg-rem + i]; sum += v*v; }\n"
   "  out[i]=sum; }\n"
   "double select_val_real(__global const double* data, int n, int pos){\n"
   "  for(int i=0;i<n;i++){\n"
   "    double val=data[i]; int less=0; int greater=0;\n"
   "    for(int j=0;j<n;j++){ double v=data[j]; if(v<val) less++; else if(v>val) greater++; }\n"
   "    if(less<=pos && greater<= (n-1-pos)) return val; }\n"
   "  return 0.0; }\n"
   "__kernel void cola_check(__global const double* bins, int step, double tol, __global double* out){\n"
   "  if(get_global_id(0)!=0) return; int posH=step/2; int posL=(step-1)/2;\n"
   "  double med=0.5*(select_val_real(bins,step,posL) + select_val_real(bins,step,posH));\n"
   "  double maxdev=0.0; for(int i=0;i<step;i++){ double dev=fabs(bins[i]-med); if(dev>maxdev) maxdev=dev; }\n"
   "  out[0]=(maxdev<tol)?1.0:0.0; }\n"
   "__kernel void nola_check(__global const double* bins, int step, double tol, __global double* out){\n"
   "  if(get_global_id(0)!=0) return; double minv=bins[0]; for(int i=1;i<step;i++){ if(bins[i]<minv) minv=bins[i]; }\n"
   "  out[0]=(minv>tol)?1.0:0.0; }\n"
   "__kernel void coherence_ratio(__global const double2* Pxy, __global const double* Pxx, __global const double* Pyy, __global double* out, int n){\n"
   "  int i=get_global_id(0); if(i>=n) return; double denom=Pxx[i]*Pyy[i]; if(denom<=0.0){ out[i]=0.0; return; }\n"
   "  double2 v=Pxy[i]; double num=v.x*v.x + v.y*v.y; out[i]=num/denom; }\n"
   "__kernel void max_mag_index(__global const double2* in, int n, __global double* out){\n"
   "  if(get_global_id(0)!=0) return; double maxv=-1.0; int idx=0; for(int i=0;i<n;i++){ double2 v=in[i]; double mag=v.x*v.x+v.y*v.y; if(mag>maxv){ maxv=mag; idx=i; } }\n"
   "  out[0]=maxv; out[1]=(double)idx; }\n";

   string build_log="";
   p.prog=CLProgramCreate(p.ctx,code,build_log);
   if(p.prog==INVALID_HANDLE)
     {
      if(!log_prog)
        {
         PrintFormat("SpectralOpenCLFFT: CLProgramCreate failed (err=%d)", GetLastError());
         if(build_log!="") Print("SpectralOpenCLFFT build log:\n", build_log);
         log_prog=true;
        }
      CLFFTFree(p);
      return false;
     }
   p.kern_bitrev=CLKernelCreate(p.prog,"bit_reverse");
   p.kern_stage=CLKernelCreate(p.prog,"fft_stage");
   p.kern_scale=CLKernelCreate(p.prog,"fft_scale");
   p.kern_dft=CLKernelCreate(p.prog,"dft_complex");
   p.kern_load=CLKernelCreate(p.prog,"load_real_segment");
   p.kern_sum=CLKernelCreate(p.prog,"seg_sums");
   p.kern_load_dt=CLKernelCreate(p.prog,"load_real_segment_detrend");
   p.kern_bitrev_b=CLKernelCreate(p.prog,"bit_reverse_batch");
   p.kern_stage_b=CLKernelCreate(p.prog,"fft_stage_batch");
   p.kern_scale_b=CLKernelCreate(p.prog,"fft_scale_batch");
   p.kern_dft_b=CLKernelCreate(p.prog,"dft_complex_batch");
   p.kern_load_b=CLKernelCreate(p.prog,"load_real_segment_batch");
   p.kern_sum_b=CLKernelCreate(p.prog,"seg_sums_batch");
   p.kern_load_dt_b=CLKernelCreate(p.prog,"load_real_segment_detrend_batch");
   p.kern_overlap=CLKernelCreate(p.prog,"overlap_add_complex");
   p.kern_norm=CLKernelCreate(p.prog,"overlap_normalize");
   p.kern_win_stats=CLKernelCreate(p.prog,"win_stats");
   p.kern_gen_freqs=CLKernelCreate(p.prog,"gen_freqs");
   p.kern_gen_times=CLKernelCreate(p.prog,"gen_times");
   p.kern_expand_onesided=CLKernelCreate(p.prog,"expand_onesided");
   p.kern_copy_slice=CLKernelCreate(p.prog,"copy_slice_cplx");
   p.kern_time_linear=CLKernelCreate(p.prog,"gen_time_linear");
   p.kern_psd_cmul=CLKernelCreate(p.prog,"psd_cmul");
   p.kern_psd_onesided=CLKernelCreate(p.prog,"psd_onesided");
   p.kern_psd_mean=CLKernelCreate(p.prog,"psd_mean");
   p.kern_psd_median=CLKernelCreate(p.prog,"psd_median");
   p.kern_pack_segments=CLKernelCreate(p.prog,"pack_segments");
   p.kern_cabs=CLKernelCreate(p.prog,"cabs_cplx");
   p.kern_carg=CLKernelCreate(p.prog,"carg_cplx");
   p.kern_unwrap=CLKernelCreate(p.prog,"unwrap_phase_rows");
   p.kern_cola_bins=CLKernelCreate(p.prog,"cola_binsums");
   p.kern_nola_bins=CLKernelCreate(p.prog,"nola_binsums");
   p.kern_cola_check=CLKernelCreate(p.prog,"cola_check");
   p.kern_nola_check=CLKernelCreate(p.prog,"nola_check");
   p.kern_coherence=CLKernelCreate(p.prog,"coherence_ratio");
   p.kern_maxmag=CLKernelCreate(p.prog,"max_mag_index");
   p.kern_copy=CLKernelCreate(p.prog,"copy_cplx");
   p.kern_hilbert=CLKernelCreate(p.prog,"hilbert_mask");
   if(p.kern_bitrev==INVALID_HANDLE || p.kern_stage==INVALID_HANDLE || p.kern_scale==INVALID_HANDLE || p.kern_dft==INVALID_HANDLE || p.kern_load==INVALID_HANDLE || p.kern_sum==INVALID_HANDLE || p.kern_load_dt==INVALID_HANDLE ||
      p.kern_bitrev_b==INVALID_HANDLE || p.kern_stage_b==INVALID_HANDLE || p.kern_scale_b==INVALID_HANDLE || p.kern_dft_b==INVALID_HANDLE || p.kern_load_b==INVALID_HANDLE || p.kern_sum_b==INVALID_HANDLE || p.kern_load_dt_b==INVALID_HANDLE ||
      p.kern_overlap==INVALID_HANDLE || p.kern_norm==INVALID_HANDLE ||
      p.kern_win_stats==INVALID_HANDLE || p.kern_gen_freqs==INVALID_HANDLE || p.kern_gen_times==INVALID_HANDLE ||
      p.kern_expand_onesided==INVALID_HANDLE || p.kern_copy_slice==INVALID_HANDLE || p.kern_time_linear==INVALID_HANDLE ||
      p.kern_psd_cmul==INVALID_HANDLE || p.kern_psd_onesided==INVALID_HANDLE || p.kern_psd_mean==INVALID_HANDLE || p.kern_psd_median==INVALID_HANDLE ||
      p.kern_pack_segments==INVALID_HANDLE || p.kern_cabs==INVALID_HANDLE || p.kern_carg==INVALID_HANDLE || p.kern_unwrap==INVALID_HANDLE ||
      p.kern_cola_bins==INVALID_HANDLE || p.kern_nola_bins==INVALID_HANDLE || p.kern_cola_check==INVALID_HANDLE || p.kern_nola_check==INVALID_HANDLE ||
      p.kern_coherence==INVALID_HANDLE || p.kern_maxmag==INVALID_HANDLE || p.kern_copy==INVALID_HANDLE || p.kern_hilbert==INVALID_HANDLE)
      {
       if(!log_kern)
         {
          PrintFormat("SpectralOpenCLFFT: CLKernelCreate failed (err=%d)", GetLastError());
          log_kern=true;
         }
       CLFFTFree(p);
       return false;
      }
   p.memA=CLBufferCreate(p.ctx,N*sizeof(double)*2,CL_MEM_READ_WRITE);
   p.memB=CLBufferCreate(p.ctx,N*sizeof(double)*2,CL_MEM_READ_WRITE);
   if(p.memA==INVALID_HANDLE || p.memB==INVALID_HANDLE)
     {
      if(!log_mem)
        {
         PrintFormat("SpectralOpenCLFFT: CLBufferCreate failed (err=%d)", GetLastError());
         log_mem=true;
        }
      CLFFTFree(p);
      return false;
     }
   p.N=N;
   p.ready=true;
   return true;
  }

inline void _pack_complex(const Complex64 &in[],double &buf[])
  {
   int N=ArraySize(in);
   ArrayResize(buf,2*N);
   for(int i=0;i<N;i++)
     { buf[2*i]=in[i].re; buf[2*i+1]=in[i].im; }
  }

inline void _unpack_complex(const double &buf[],Complex64 &out[])
  {
   int N=ArraySize(buf)/2;
   ArrayResize(out,N);
   for(int i=0;i<N;i++)
     { out[i]=Cx(buf[2*i],buf[2*i+1]); }
  }

inline bool CLFFTExecute(CLFFTPlan &p,const Complex64 &in[],Complex64 &out[],const bool inverse)
  {
   int N=ArraySize(in);
   if(!CLFFTInit(p,N)) return false;
   bool pow2 = ((N & (N-1))==0);
   double buf[];
   _pack_complex(in,buf);
   CLBufferWrite(p.memA,buf);

   if(!pow2)
     {
      CLSetKernelArgMem(p.kern_dft,0,p.memA);
      CLSetKernelArgMem(p.kern_dft,1,p.memB);
      CLSetKernelArg(p.kern_dft,2,N);
      CLSetKernelArg(p.kern_dft,3,(int)(inverse?1:0));
      uint offs0[1]={0}; uint work0[1]={(uint)N};
      if(!CLExecute(p.kern_dft,1,offs0,work0)) return false;
      CLBufferRead(p.memB,buf);
      _unpack_complex(buf,out);
      return true;
     }

   // bit-reversal into memB
   int bits=0; int tmp=N;
   while(tmp>1){ bits++; tmp>>=1; }
   CLSetKernelArgMem(p.kern_bitrev,0,p.memA);
   CLSetKernelArgMem(p.kern_bitrev,1,p.memB);
   CLSetKernelArg(p.kern_bitrev,2,N);
   CLSetKernelArg(p.kern_bitrev,3,bits);
   uint offs[1]={0}; uint work[1]={(uint)N};
   if(!CLExecute(p.kern_bitrev,1,offs,work)) return false;

   // stages ping-pong
   int m=2;
   bool toggle=false;
   while(m<=N)
     {
      int half=m>>1;
      int total=N>>1;
      int inMem= toggle ? p.memA : p.memB;
      int outMem= toggle ? p.memB : p.memA;
      CLSetKernelArgMem(p.kern_stage,0,inMem);
      CLSetKernelArgMem(p.kern_stage,1,outMem);
      CLSetKernelArg(p.kern_stage,2,N);
      CLSetKernelArg(p.kern_stage,3,m);
      CLSetKernelArg(p.kern_stage,4,(int)(inverse?1:0));
      uint work2[1]={(uint)total};
      if(!CLExecute(p.kern_stage,1,offs,work2)) return false;
      toggle=!toggle;
      m<<=1;
     }
   int finalMem = toggle ? p.memB : p.memA;
   if(inverse)
     {
      double invN=1.0/(double)N;
      CLSetKernelArgMem(p.kern_scale,0,finalMem);
      CLSetKernelArg(p.kern_scale,1,N);
      CLSetKernelArg(p.kern_scale,2,invN);
      if(!CLExecute(p.kern_scale,1,offs,work)) return false;
     }
   CLBufferRead(finalMem,buf);
   _unpack_complex(buf,out);
   return true;
  }

inline bool CLFFTUpldRealSeries(CLFFTPlan &p,const double &x[],const double &win[])
  {
   int xlen=ArraySize(x);
   int wlen=ArraySize(win);
   if(xlen<=0 || wlen<=0) return false;
   if(p.memX==INVALID_HANDLE || p.lenX!=xlen)
     {
      if(p.memX!=INVALID_HANDLE) CLBufferFree(p.memX);
      p.memX=CLBufferCreate(p.ctx,xlen*sizeof(double),CL_MEM_READ_ONLY);
      if(p.memX==INVALID_HANDLE) return false;
      p.lenX=xlen;
     }
   if(p.memWin==INVALID_HANDLE || p.lenWin!=wlen)
     {
      if(p.memWin!=INVALID_HANDLE) CLBufferFree(p.memWin);
      p.memWin=CLBufferCreate(p.ctx,wlen*sizeof(double),CL_MEM_READ_ONLY);
      if(p.memWin==INVALID_HANDLE) return false;
      p.lenWin=wlen;
     }
   CLBufferWrite(p.memX,x);
   CLBufferWrite(p.memWin,win);
   if(p.memSum==INVALID_HANDLE)
     {
      p.memSum=CLBufferCreate(p.ctx,2*sizeof(double),CL_MEM_READ_WRITE);
      if(p.memSum==INVALID_HANDLE) return false;
     }
   return true;
  }

inline bool CLFFTEnsureBatchBuffers(CLFFTPlan &p,const int batch)
  {
   if(batch<=0) return false;
   if(p.memA!=INVALID_HANDLE && p.memB!=INVALID_HANDLE && p.batch==batch) return true;
   if(p.memA!=INVALID_HANDLE) { CLBufferFree(p.memA); p.memA=INVALID_HANDLE; }
   if(p.memB!=INVALID_HANDLE) { CLBufferFree(p.memB); p.memB=INVALID_HANDLE; }
   long total = (long)p.N * (long)batch;
   p.memA=CLBufferCreate(p.ctx,(int)(total*sizeof(double)*2),CL_MEM_READ_WRITE);
   p.memB=CLBufferCreate(p.ctx,(int)(total*sizeof(double)*2),CL_MEM_READ_WRITE);
   if(p.memA==INVALID_HANDLE || p.memB==INVALID_HANDLE) return false;
   p.batch=batch;
   return true;
  }

inline bool CLFFTLoadRealSegment(CLFFTPlan &p,const double &x[],const double &win[],const int start,const int nperseg,const int nfft,
                                 const int boundary_type,const int nedge,const int ext_valid)
  {
   if(!CLFFTInit(p,nfft)) return false;
   if(!CLFFTUpldRealSeries(p,x,win)) return false;
   CLSetKernelArgMem(p.kern_load,0,p.memX);
   CLSetKernelArgMem(p.kern_load,1,p.memWin);
   CLSetKernelArgMem(p.kern_load,2,p.memA);
   CLSetKernelArg(p.kern_load,3,p.lenX);
   CLSetKernelArg(p.kern_load,4,start);
   CLSetKernelArg(p.kern_load,5,nperseg);
   CLSetKernelArg(p.kern_load,6,nfft);
   CLSetKernelArg(p.kern_load,7,boundary_type);
   CLSetKernelArg(p.kern_load,8,nedge);
   CLSetKernelArg(p.kern_load,9,ext_valid);
   uint offs[1]={0};
   uint work[1]={(uint)nfft};
   return CLExecute(p.kern_load,1,offs,work);
  }

inline bool CLFFTLoadRealSegmentDetrend(CLFFTPlan &p,const double &x[],const double &win[],const int start,const int nperseg,const int nfft,
                                        const int detrend_type,const int boundary_type,const int nedge,const int ext_valid)
  {
   if(!CLFFTInit(p,nfft)) return false;
   if(!CLFFTUpldRealSeries(p,x,win)) return false;
   if(detrend_type==0)
      return CLFFTLoadRealSegment(p,x,win,start,nperseg,nfft,boundary_type,nedge,ext_valid);

   // compute sums on GPU (single work-item)
   CLSetKernelArgMem(p.kern_sum,0,p.memX);
   CLSetKernelArg(p.kern_sum,1,p.lenX);
   CLSetKernelArg(p.kern_sum,2,start);
   CLSetKernelArg(p.kern_sum,3,nperseg);
   CLSetKernelArg(p.kern_sum,4,boundary_type);
   CLSetKernelArg(p.kern_sum,5,nedge);
   CLSetKernelArg(p.kern_sum,6,ext_valid);
   CLSetKernelArgMem(p.kern_sum,7,p.memSum);
   uint offs0[1]={0}; uint work0[1]={1};
   if(!CLExecute(p.kern_sum,1,offs0,work0)) return false;

   // precomputed sums of i and i^2
   double sum_i=0.0, sum_i2=0.0;
   for(int i=0;i<nperseg;i++){ sum_i += (double)i; sum_i2 += (double)i*(double)i; }

   CLSetKernelArgMem(p.kern_load_dt,0,p.memX);
   CLSetKernelArgMem(p.kern_load_dt,1,p.memWin);
   CLSetKernelArgMem(p.kern_load_dt,2,p.memSum);
   CLSetKernelArg(p.kern_load_dt,3,p.lenX);
   CLSetKernelArg(p.kern_load_dt,4,start);
   CLSetKernelArg(p.kern_load_dt,5,nperseg);
   CLSetKernelArg(p.kern_load_dt,6,nfft);
   CLSetKernelArg(p.kern_load_dt,7,detrend_type);
   CLSetKernelArg(p.kern_load_dt,8,sum_i);
   CLSetKernelArg(p.kern_load_dt,9,sum_i2);
   CLSetKernelArg(p.kern_load_dt,10,boundary_type);
   CLSetKernelArg(p.kern_load_dt,11,nedge);
   CLSetKernelArg(p.kern_load_dt,12,ext_valid);
   CLSetKernelArgMem(p.kern_load_dt,13,p.memA);
   uint offs[1]={0}; uint work[1]={(uint)nfft};
   return CLExecute(p.kern_load_dt,1,offs,work);
  }

inline bool CLFFTLoadRealSegmentDetrendMem(CLFFTPlan &p,const int start,const int nperseg,const int nfft,const int detrend_type,
                                           const int boundary_type,const int nedge,const int ext_valid)
  {
   if(!p.ready || p.memX==INVALID_HANDLE || p.memWin==INVALID_HANDLE) return false;
   if(p.N!=nfft) return false;
   if(detrend_type==0)
     {
      CLSetKernelArgMem(p.kern_load,0,p.memX);
      CLSetKernelArgMem(p.kern_load,1,p.memWin);
      CLSetKernelArgMem(p.kern_load,2,p.memA);
      CLSetKernelArg(p.kern_load,3,p.lenX);
      CLSetKernelArg(p.kern_load,4,start);
      CLSetKernelArg(p.kern_load,5,nperseg);
      CLSetKernelArg(p.kern_load,6,nfft);
      CLSetKernelArg(p.kern_load,7,boundary_type);
      CLSetKernelArg(p.kern_load,8,nedge);
      CLSetKernelArg(p.kern_load,9,ext_valid);
      uint offs[1]={0}; uint work[1]={(uint)nfft};
      return CLExecute(p.kern_load,1,offs,work);
     }

   // compute sums on GPU (single work-item)
   CLSetKernelArgMem(p.kern_sum,0,p.memX);
   CLSetKernelArg(p.kern_sum,1,p.lenX);
   CLSetKernelArg(p.kern_sum,2,start);
   CLSetKernelArg(p.kern_sum,3,nperseg);
   CLSetKernelArg(p.kern_sum,4,boundary_type);
   CLSetKernelArg(p.kern_sum,5,nedge);
   CLSetKernelArg(p.kern_sum,6,ext_valid);
   CLSetKernelArgMem(p.kern_sum,7,p.memSum);
   uint offs0[1]={0}; uint work0[1]={1};
   if(!CLExecute(p.kern_sum,1,offs0,work0)) return false;

   // precomputed sums of i and i^2
   double sum_i=0.0, sum_i2=0.0;
   for(int i=0;i<nperseg;i++){ sum_i += (double)i; sum_i2 += (double)i*(double)i; }

   CLSetKernelArgMem(p.kern_load_dt,0,p.memX);
   CLSetKernelArgMem(p.kern_load_dt,1,p.memWin);
   CLSetKernelArgMem(p.kern_load_dt,2,p.memSum);
   CLSetKernelArg(p.kern_load_dt,3,p.lenX);
   CLSetKernelArg(p.kern_load_dt,4,start);
   CLSetKernelArg(p.kern_load_dt,5,nperseg);
   CLSetKernelArg(p.kern_load_dt,6,nfft);
   CLSetKernelArg(p.kern_load_dt,7,detrend_type);
   CLSetKernelArg(p.kern_load_dt,8,sum_i);
   CLSetKernelArg(p.kern_load_dt,9,sum_i2);
   CLSetKernelArg(p.kern_load_dt,10,boundary_type);
   CLSetKernelArg(p.kern_load_dt,11,nedge);
   CLSetKernelArg(p.kern_load_dt,12,ext_valid);
   CLSetKernelArgMem(p.kern_load_dt,13,p.memA);
   uint offs[1]={0}; uint work[1]={(uint)nfft};
   return CLExecute(p.kern_load_dt,1,offs,work);
  }

inline bool CLFFTLoadRealSegmentsDetrendBatch(CLFFTPlan &p,const double &x[],const double &win[],const int start0,const int step,const int nperseg,const int nfft,
                                              const int detrend_type,const int nseg,const int boundary_type,const int nedge,const int ext_valid)
  {
   if(nseg<=0) return false;
   if(!CLFFTInit(p,nfft)) return false;
   if(!CLFFTUpldRealSeries(p,x,win)) return false;
   if(!CLFFTEnsureBatchBuffers(p,nseg)) return false;

   // ensure sum buffer size
   if(p.memSum!=INVALID_HANDLE) CLBufferFree(p.memSum);
   p.memSum=CLBufferCreate(p.ctx,(int)(2*nseg*sizeof(double)),CL_MEM_READ_WRITE);
   if(p.memSum==INVALID_HANDLE) return false;

   uint offs[1]={0};
   if(detrend_type==0)
     {
      CLSetKernelArgMem(p.kern_load_b,0,p.memX);
      CLSetKernelArgMem(p.kern_load_b,1,p.memWin);
      CLSetKernelArgMem(p.kern_load_b,2,p.memA);
      CLSetKernelArg(p.kern_load_b,3,p.lenX);
      CLSetKernelArg(p.kern_load_b,4,start0);
      CLSetKernelArg(p.kern_load_b,5,step);
      CLSetKernelArg(p.kern_load_b,6,nperseg);
      CLSetKernelArg(p.kern_load_b,7,nfft);
      CLSetKernelArg(p.kern_load_b,8,boundary_type);
      CLSetKernelArg(p.kern_load_b,9,nedge);
      CLSetKernelArg(p.kern_load_b,10,ext_valid);
      uint work0[1]={(uint)(nseg*nfft)};
      return CLExecute(p.kern_load_b,1,offs,work0);
     }

   // sums per segment
   CLSetKernelArgMem(p.kern_sum_b,0,p.memX);
   CLSetKernelArg(p.kern_sum_b,1,p.lenX);
   CLSetKernelArg(p.kern_sum_b,2,start0);
   CLSetKernelArg(p.kern_sum_b,3,step);
   CLSetKernelArg(p.kern_sum_b,4,nperseg);
   CLSetKernelArg(p.kern_sum_b,5,boundary_type);
   CLSetKernelArg(p.kern_sum_b,6,nedge);
   CLSetKernelArg(p.kern_sum_b,7,ext_valid);
   CLSetKernelArgMem(p.kern_sum_b,8,p.memSum);
   uint workS[1]={(uint)nseg};
   if(!CLExecute(p.kern_sum_b,1,offs,workS)) return false;

   double sum_i=0.0, sum_i2=0.0;
   for(int i=0;i<nperseg;i++){ sum_i += (double)i; sum_i2 += (double)i*(double)i; }

   CLSetKernelArgMem(p.kern_load_dt_b,0,p.memX);
   CLSetKernelArgMem(p.kern_load_dt_b,1,p.memWin);
   CLSetKernelArgMem(p.kern_load_dt_b,2,p.memSum);
   CLSetKernelArg(p.kern_load_dt_b,3,p.lenX);
   CLSetKernelArg(p.kern_load_dt_b,4,start0);
   CLSetKernelArg(p.kern_load_dt_b,5,step);
   CLSetKernelArg(p.kern_load_dt_b,6,nperseg);
   CLSetKernelArg(p.kern_load_dt_b,7,nfft);
   CLSetKernelArg(p.kern_load_dt_b,8,detrend_type);
   CLSetKernelArg(p.kern_load_dt_b,9,sum_i);
   CLSetKernelArg(p.kern_load_dt_b,10,sum_i2);
   CLSetKernelArg(p.kern_load_dt_b,11,boundary_type);
   CLSetKernelArg(p.kern_load_dt_b,12,nedge);
   CLSetKernelArg(p.kern_load_dt_b,13,ext_valid);
   CLSetKernelArgMem(p.kern_load_dt_b,14,p.memA);
   uint workL[1]={(uint)(nseg*nfft)};
   return CLExecute(p.kern_load_dt_b,1,offs,workL);
  }

inline bool CLFFTExecuteFromMemA(CLFFTPlan &p,Complex64 &out[],const bool inverse)
  {
   if(!p.ready) return false;
   int N=p.N;
   bool pow2 = ((N & (N-1))==0);
   double buf[];
   ArrayResize(buf,2*N);

   if(!pow2)
     {
      CLSetKernelArgMem(p.kern_dft,0,p.memA);
      CLSetKernelArgMem(p.kern_dft,1,p.memB);
      CLSetKernelArg(p.kern_dft,2,N);
      CLSetKernelArg(p.kern_dft,3,(int)(inverse?1:0));
      uint offs0[1]={0}; uint work0[1]={(uint)N};
      if(!CLExecute(p.kern_dft,1,offs0,work0)) return false;
      CLBufferRead(p.memB,buf);
      _unpack_complex(buf,out);
      return true;
     }

   int bits=0; int tmp=N;
   while(tmp>1){ bits++; tmp>>=1; }
   CLSetKernelArgMem(p.kern_bitrev,0,p.memA);
   CLSetKernelArgMem(p.kern_bitrev,1,p.memB);
   CLSetKernelArg(p.kern_bitrev,2,N);
   CLSetKernelArg(p.kern_bitrev,3,bits);
   uint offs[1]={0}; uint work[1]={(uint)N};
   if(!CLExecute(p.kern_bitrev,1,offs,work)) return false;

   int m=2;
   bool toggle=false;
   while(m<=N)
     {
      int total=N>>1;
      int inMem= toggle ? p.memA : p.memB;
      int outMem= toggle ? p.memB : p.memA;
      CLSetKernelArgMem(p.kern_stage,0,inMem);
      CLSetKernelArgMem(p.kern_stage,1,outMem);
      CLSetKernelArg(p.kern_stage,2,N);
      CLSetKernelArg(p.kern_stage,3,m);
      CLSetKernelArg(p.kern_stage,4,(int)(inverse?1:0));
      uint work2[1]={(uint)total};
      if(!CLExecute(p.kern_stage,1,offs,work2)) return false;
      toggle=!toggle;
      m<<=1;
     }
   int finalMem = toggle ? p.memB : p.memA;
   if(inverse)
     {
      double invN=1.0/(double)N;
      CLSetKernelArgMem(p.kern_scale,0,finalMem);
      CLSetKernelArg(p.kern_scale,1,N);
      CLSetKernelArg(p.kern_scale,2,invN);
      if(!CLExecute(p.kern_scale,1,offs,work)) return false;
     }
   CLBufferRead(finalMem,buf);
   _unpack_complex(buf,out);
   return true;
  }

inline bool CLFFTExecuteBatchFromMemA(CLFFTPlan &p,const int batch,Complex64 &outFlat[],const bool inverse)
  {
   if(!p.ready || batch<=0) return false;
   if(p.batch!=batch) return false;
   int N=p.N;
   int total = batch * N;
   bool pow2 = ((N & (N-1))==0);
   double buf[];
   ArrayResize(buf,2*total);

   if(!pow2)
     {
      CLSetKernelArgMem(p.kern_dft_b,0,p.memA);
      CLSetKernelArgMem(p.kern_dft_b,1,p.memB);
      CLSetKernelArg(p.kern_dft_b,2,N);
      CLSetKernelArg(p.kern_dft_b,3,(int)(inverse?1:0));
      uint offs0[1]={0}; uint work0[1]={(uint)total};
      if(!CLExecute(p.kern_dft_b,1,offs0,work0)) return false;
      CLBufferRead(p.memB,buf);
      _unpack_complex(buf,outFlat);
      p.memFinal=p.memB;
      return true;
     }

   int bits=0; int tmp=N;
   while(tmp>1){ bits++; tmp>>=1; }
   CLSetKernelArgMem(p.kern_bitrev_b,0,p.memA);
   CLSetKernelArgMem(p.kern_bitrev_b,1,p.memB);
   CLSetKernelArg(p.kern_bitrev_b,2,N);
   CLSetKernelArg(p.kern_bitrev_b,3,bits);
   uint offs[1]={0}; uint work[1]={(uint)total};
   if(!CLExecute(p.kern_bitrev_b,1,offs,work)) return false;

   int m=2;
   bool toggle=false;
   while(m<=N)
     {
      int totalHalf = (N>>1) * batch;
      int inMem= toggle ? p.memA : p.memB;
      int outMem= toggle ? p.memB : p.memA;
      CLSetKernelArgMem(p.kern_stage_b,0,inMem);
      CLSetKernelArgMem(p.kern_stage_b,1,outMem);
      CLSetKernelArg(p.kern_stage_b,2,N);
      CLSetKernelArg(p.kern_stage_b,3,m);
      CLSetKernelArg(p.kern_stage_b,4,(int)(inverse?1:0));
      uint work2[1]={(uint)totalHalf};
      if(!CLExecute(p.kern_stage_b,1,offs,work2)) return false;
      toggle=!toggle;
      m<<=1;
     }
   int finalMem = toggle ? p.memB : p.memA;
   if(inverse)
     {
      double invN=1.0/(double)N;
      CLSetKernelArgMem(p.kern_scale_b,0,finalMem);
      CLSetKernelArg(p.kern_scale_b,1,(int)total);
      CLSetKernelArg(p.kern_scale_b,2,invN);
      if(!CLExecute(p.kern_scale_b,1,offs,work)) return false;
     }
   CLBufferRead(finalMem,buf);
   _unpack_complex(buf,outFlat);
   p.memFinal=finalMem;
   return true;
  }

inline bool CLFFTExecuteBatchFromMemA_NoRead(CLFFTPlan &p,const int batch,const bool inverse)
  {
   if(!p.ready || batch<=0) return false;
   if(p.batch!=batch) return false;
   int N=p.N;
   long total = (long)batch * (long)N;
   bool pow2 = ((N & (N-1))==0);

   if(!pow2)
     {
      CLSetKernelArgMem(p.kern_dft_b,0,p.memA);
      CLSetKernelArgMem(p.kern_dft_b,1,p.memB);
      CLSetKernelArg(p.kern_dft_b,2,N);
      CLSetKernelArg(p.kern_dft_b,3,(int)(inverse?1:0));
      uint offs0[1]={0}; uint work0[1]={(uint)total};
      if(!CLExecute(p.kern_dft_b,1,offs0,work0)) return false;
      p.memFinal=p.memB;
      return true;
     }

   int bits=0; int tmp=N;
   while(tmp>1){ bits++; tmp>>=1; }
   CLSetKernelArgMem(p.kern_bitrev_b,0,p.memA);
   CLSetKernelArgMem(p.kern_bitrev_b,1,p.memB);
   CLSetKernelArg(p.kern_bitrev_b,2,N);
   CLSetKernelArg(p.kern_bitrev_b,3,bits);
   uint offs[1]={0}; uint work[1]={(uint)total};
   if(!CLExecute(p.kern_bitrev_b,1,offs,work)) return false;

   int m=2;
   bool toggle=false;
   while(m<=N)
     {
      int totalHalf = (N>>1) * batch;
      int inMem= toggle ? p.memA : p.memB;
      int outMem= toggle ? p.memB : p.memA;
      CLSetKernelArgMem(p.kern_stage_b,0,inMem);
      CLSetKernelArgMem(p.kern_stage_b,1,outMem);
      CLSetKernelArg(p.kern_stage_b,2,N);
      CLSetKernelArg(p.kern_stage_b,3,m);
      CLSetKernelArg(p.kern_stage_b,4,(int)(inverse?1:0));
      uint work2[1]={(uint)totalHalf};
      if(!CLExecute(p.kern_stage_b,1,offs,work2)) return false;
      toggle=!toggle;
      m<<=1;
     }
   int finalMem = toggle ? p.memB : p.memA;
   if(inverse)
     {
      double invN=1.0/(double)N;
      CLSetKernelArgMem(p.kern_scale_b,0,finalMem);
      CLSetKernelArg(p.kern_scale_b,1,(int)total);
      CLSetKernelArg(p.kern_scale_b,2,invN);
      if(!CLExecute(p.kern_scale_b,1,offs,work)) return false;
     }
   p.memFinal=finalMem;
   return true;
  }

inline bool CLFFTExecuteFromMemA_NoRead(CLFFTPlan &p,const bool inverse)
  {
   if(!p.ready) return false;
   if(p.batch!=1)
     {
      if(!CLFFTEnsureBatchBuffers(p,1)) return false;
     }
   return CLFFTExecuteBatchFromMemA_NoRead(p,1,inverse);
  }

inline bool CLFFTCopyCplx(CLFFTPlan &p,const int srcMem,const int dstMem,const int n)
  {
   if(!p.ready || srcMem==INVALID_HANDLE || dstMem==INVALID_HANDLE || n<=0) return false;
   CLSetKernelArgMem(p.kern_copy,0,srcMem);
   CLSetKernelArgMem(p.kern_copy,1,dstMem);
   CLSetKernelArg(p.kern_copy,2,n);
   uint offs[1]={0}; uint work[1]={(uint)n};
   return CLExecute(p.kern_copy,1,offs,work);
  }

inline bool CLFFTHilbertMask(CLFFTPlan &p,const int mem,const int n)
  {
   if(!p.ready || mem==INVALID_HANDLE || n<=0) return false;
   CLSetKernelArgMem(p.kern_hilbert,0,mem);
   CLSetKernelArg(p.kern_hilbert,1,n);
   uint offs[1]={0}; uint work[1]={(uint)n};
   return CLExecute(p.kern_hilbert,1,offs,work);
  }

inline bool CLFFTScaleBatchFromFinal(CLFFTPlan &p,const int batch,const double scale)
  {
   if(!p.ready || p.memFinal==INVALID_HANDLE || batch<=0) return false;
   if(p.batch!=batch) return false;
   long total = (long)batch * (long)p.N;
   CLSetKernelArgMem(p.kern_scale_b,0,p.memFinal);
   CLSetKernelArg(p.kern_scale_b,1,(int)total);
   CLSetKernelArg(p.kern_scale_b,2,scale);
   uint offs[1]={0}; uint work[1]={(uint)total};
   return CLExecute(p.kern_scale_b,1,offs,work);
  }

inline bool CLFFTEnsureHalfBuffer(CLFFTPlan &p,const int len)
  {
   if(len<=0) return false;
   if(p.memHalf!=INVALID_HANDLE && p.lenHalf==len) return true;
   if(p.memHalf!=INVALID_HANDLE) { CLBufferFree(p.memHalf); p.memHalf=INVALID_HANDLE; }
   p.memHalf=CLBufferCreate(p.ctx,len*sizeof(double)*2,CL_MEM_READ_WRITE);
   if(p.memHalf==INVALID_HANDLE) return false;
   p.lenHalf=len;
   return true;
  }

inline bool CLFFTExpandOnesidedToMemA(CLFFTPlan &p,const Complex64 &inHalf[],const int batch,const int nfreq)
  {
   if(!p.ready || batch<=0 || nfreq<=0) return false;
   if(!CLFFTEnsureBatchBuffers(p,batch)) return false;
   int totalHalf=batch*nfreq;
   if(!CLFFTEnsureHalfBuffer(p,totalHalf)) return false;
   double buf[];
   ArrayResize(buf,2*totalHalf);
   for(int i=0;i<totalHalf;i++){ buf[2*i]=inHalf[i].re; buf[2*i+1]=inHalf[i].im; }
   CLBufferWrite(p.memHalf,buf);
   int N=p.N;
   int kmax=(N%2==0)? (nfreq-2) : (nfreq-1);
   if(kmax<0) kmax=0;
   CLSetKernelArgMem(p.kern_expand_onesided,0,p.memHalf);
   CLSetKernelArgMem(p.kern_expand_onesided,1,p.memA);
   CLSetKernelArg(p.kern_expand_onesided,2,N);
   CLSetKernelArg(p.kern_expand_onesided,3,nfreq);
   CLSetKernelArg(p.kern_expand_onesided,4,kmax);
   uint offs[1]={0}; uint work[1]={(uint)totalHalf};
   return CLExecute(p.kern_expand_onesided,1,offs,work);
  }

inline bool CLFFTCropComplexFromMem(CLFFTPlan &p,const int srcMem,const int start,const int newlen)
  {
   if(!p.ready || newlen<=0) return false;
   if(p.memCrop!=INVALID_HANDLE && p.lenCrop==newlen) { /* reuse */ }
   else
     {
      if(p.memCrop!=INVALID_HANDLE) { CLBufferFree(p.memCrop); p.memCrop=INVALID_HANDLE; }
      p.memCrop=CLBufferCreate(p.ctx,newlen*sizeof(double)*2,CL_MEM_READ_WRITE);
      if(p.memCrop==INVALID_HANDLE) return false;
      p.lenCrop=newlen;
     }
   CLSetKernelArgMem(p.kern_copy_slice,0,srcMem);
   CLSetKernelArg(p.kern_copy_slice,1,start);
   CLSetKernelArg(p.kern_copy_slice,2,newlen);
   CLSetKernelArgMem(p.kern_copy_slice,3,p.memCrop);
   uint offs[1]={0}; uint work[1]={(uint)newlen};
   if(!CLExecute(p.kern_copy_slice,1,offs,work)) return false;
   p.memFinal=p.memCrop;
   return true;
  }

inline bool CLFFTOverlapAddFromFinal_NoRead(CLFFTPlan &p,const int nseg,const int nperseg,const int nstep,const int N,
                                            const double &win[],const double scale,const int outlen)
  {
   if(p.memFinal==INVALID_HANDLE) return false;
   if(p.memWin==INVALID_HANDLE || p.lenWin!=nperseg)
     {
      if(p.memWin!=INVALID_HANDLE) CLBufferFree(p.memWin);
      p.memWin=CLBufferCreate(p.ctx,nperseg*sizeof(double),CL_MEM_READ_ONLY);
      if(p.memWin==INVALID_HANDLE) return false;
      p.lenWin=nperseg;
      CLBufferWrite(p.memWin,win);
     }
   else
     {
      CLBufferWrite(p.memWin,win);
     }
   if(p.memOutX==INVALID_HANDLE || p.lenOutReal!=outlen)
     {
      if(p.memOutX!=INVALID_HANDLE) CLBufferFree(p.memOutX);
      if(p.memOutNorm!=INVALID_HANDLE) CLBufferFree(p.memOutNorm);
      p.memOutX=CLBufferCreate(p.ctx,outlen*sizeof(double)*2,CL_MEM_READ_WRITE);
      p.memOutNorm=CLBufferCreate(p.ctx,outlen*sizeof(double),CL_MEM_READ_WRITE);
      if(p.memOutX==INVALID_HANDLE || p.memOutNorm==INVALID_HANDLE) return false;
      p.lenOutReal=outlen;
     }
   CLSetKernelArgMem(p.kern_overlap,0,p.memFinal);
   CLSetKernelArgMem(p.kern_overlap,1,p.memWin);
   CLSetKernelArg(p.kern_overlap,2,nseg);
   CLSetKernelArg(p.kern_overlap,3,nperseg);
   CLSetKernelArg(p.kern_overlap,4,nstep);
   CLSetKernelArg(p.kern_overlap,5,N);
   CLSetKernelArg(p.kern_overlap,6,outlen);
   CLSetKernelArg(p.kern_overlap,7,scale);
   CLSetKernelArgMem(p.kern_overlap,8,p.memOutX);
   CLSetKernelArgMem(p.kern_overlap,9,p.memOutNorm);
   uint offs[1]={0}; uint work[1]={(uint)outlen};
   if(!CLExecute(p.kern_overlap,1,offs,work)) return false;
   CLSetKernelArgMem(p.kern_norm,0,p.memOutX);
   CLSetKernelArgMem(p.kern_norm,1,p.memOutNorm);
   CLSetKernelArg(p.kern_norm,2,outlen);
   if(!CLExecute(p.kern_norm,1,offs,work)) return false;
   p.memFinal=p.memOutX;
   return true;
  }

inline bool CLFFTGenerateTimeLinear(CLFFTPlan &p,const int n,const double fs,double &t[])
  {
   if(!p.ready) return false;
   if(n<=0) { ArrayResize(t,0); return true; }
   if(p.memOutReal==INVALID_HANDLE || p.lenOutReal!=n)
     {
      if(p.memOutReal!=INVALID_HANDLE) { CLBufferFree(p.memOutReal); p.memOutReal=INVALID_HANDLE; }
      p.memOutReal=CLBufferCreate(p.ctx,n*sizeof(double),CL_MEM_READ_WRITE);
      if(p.memOutReal==INVALID_HANDLE) return false;
      p.lenOutReal=n;
     }
   CLSetKernelArg(p.kern_time_linear,0,n);
   CLSetKernelArg(p.kern_time_linear,1,fs);
   CLSetKernelArgMem(p.kern_time_linear,2,p.memOutReal);
   uint offs[1]={0}; uint work[1]={(uint)n};
   if(!CLExecute(p.kern_time_linear,1,offs,work)) return false;
   ArrayResize(t,n);
   CLBufferRead(p.memOutReal,t);
   return true;
  }

inline bool CLFFTPSDEnsureBuffers(CLFFTPlan &p,const int total,const int nfreq)
  {
   if(!p.ready) return false;
   if(total<=0 || nfreq<=0) return false;
   if(p.memPSD==INVALID_HANDLE || p.lenPSD!=total)
     {
      if(p.memPSD!=INVALID_HANDLE) { CLBufferFree(p.memPSD); p.memPSD=INVALID_HANDLE; }
      p.memPSD=CLBufferCreate(p.ctx,total*sizeof(double)*2,CL_MEM_READ_WRITE);
      if(p.memPSD==INVALID_HANDLE) return false;
      p.lenPSD=total;
     }
   if(p.memAvg==INVALID_HANDLE || p.lenAvg!=nfreq)
     {
      if(p.memAvg!=INVALID_HANDLE) { CLBufferFree(p.memAvg); p.memAvg=INVALID_HANDLE; }
      p.memAvg=CLBufferCreate(p.ctx,nfreq*sizeof(double)*2,CL_MEM_READ_WRITE);
      if(p.memAvg==INVALID_HANDLE) return false;
      p.lenAvg=nfreq;
     }
   return true;
  }

inline bool CLFFTPackEnsure(CLFFTPlan &p,const int total)
  {
   if(!p.ready) return false;
   if(total<=0) return false;
   if(p.memPack!=INVALID_HANDLE && p.lenPack==total) return true;
   if(p.memPack!=INVALID_HANDLE) { CLBufferFree(p.memPack); p.memPack=INVALID_HANDLE; }
   p.memPack=CLBufferCreate(p.ctx,total*sizeof(double)*2,CL_MEM_READ_WRITE);
   if(p.memPack==INVALID_HANDLE) return false;
   p.lenPack=total;
   return true;
  }

inline bool CLFFTPackSegments(CLFFTPlan &p,const int srcMem,const int nseg,const int nfft,const int nfreq,const int dstMem)
  {
   if(!p.ready) return false;
   if(nseg<=0 || nfft<=0 || nfreq<=0) return false;
   int total=nseg*nfreq;
   CLSetKernelArgMem(p.kern_pack_segments,0,srcMem);
   CLSetKernelArg(p.kern_pack_segments,1,nseg);
   CLSetKernelArg(p.kern_pack_segments,2,nfft);
   CLSetKernelArg(p.kern_pack_segments,3,nfreq);
   CLSetKernelArgMem(p.kern_pack_segments,4,dstMem);
   uint offs[1]={0}; uint work[1]={(uint)total};
   return CLExecute(p.kern_pack_segments,1,offs,work);
  }

inline bool CLFFTEnsureOutReal(CLFFTPlan &p,const int n)
  {
   if(!p.ready) return false;
   if(n<=0) return false;
   if(p.memOutReal!=INVALID_HANDLE && p.lenOutReal==n) return true;
   if(p.memOutReal!=INVALID_HANDLE){ CLBufferFree(p.memOutReal); p.memOutReal=INVALID_HANDLE; }
   p.memOutReal=CLBufferCreate(p.ctx,n*sizeof(double),CL_MEM_READ_WRITE);
   if(p.memOutReal==INVALID_HANDLE) return false;
   p.lenOutReal=n;
   return true;
  }

inline bool CLFFTComputeMag(CLFFTPlan &p,const int memIn,const int n)
  {
   if(!p.ready) return false;
   if(n<=0) return false;
   if(!CLFFTEnsureOutReal(p,n)) return false;
   CLSetKernelArgMem(p.kern_cabs,0,memIn);
   CLSetKernelArgMem(p.kern_cabs,1,p.memOutReal);
   CLSetKernelArg(p.kern_cabs,2,n);
   uint offs[1]={0}; uint work[1]={(uint)n};
   return CLExecute(p.kern_cabs,1,offs,work);
  }

inline bool CLFFTComputePhase(CLFFTPlan &p,const int memIn,const int n)
  {
   if(!p.ready) return false;
   if(n<=0) return false;
   if(!CLFFTEnsureOutReal(p,n)) return false;
   CLSetKernelArgMem(p.kern_carg,0,memIn);
   CLSetKernelArgMem(p.kern_carg,1,p.memOutReal);
   CLSetKernelArg(p.kern_carg,2,n);
   uint offs[1]={0}; uint work[1]={(uint)n};
   return CLExecute(p.kern_carg,1,offs,work);
  }

inline bool CLFFTUnwrapPhase(CLFFTPlan &p,const int nseg,const int nfreq)
  {
   if(!p.ready) return false;
   if(nseg<=0 || nfreq<=0) return false;
   int total=nseg*nfreq;
   if(p.memOutReal==INVALID_HANDLE || p.lenOutReal!=total) return false;
   CLSetKernelArgMem(p.kern_unwrap,0,p.memOutReal);
   CLSetKernelArg(p.kern_unwrap,1,nseg);
   CLSetKernelArg(p.kern_unwrap,2,nfreq);
   uint offs[1]={0}; uint work[1]={(uint)nseg};
   return CLExecute(p.kern_unwrap,1,offs,work);
  }

inline bool CLFFTUploadWin(CLFFTPlan &p,const double &win[])
  {
   int wlen=ArraySize(win);
   if(!p.ready || wlen<=0) return false;
   if(p.memWin==INVALID_HANDLE || p.lenWin!=wlen)
     {
      if(p.memWin!=INVALID_HANDLE) { CLBufferFree(p.memWin); p.memWin=INVALID_HANDLE; }
      p.memWin=CLBufferCreate(p.ctx,wlen*sizeof(double),CL_MEM_READ_ONLY);
      if(p.memWin==INVALID_HANDLE) return false;
      p.lenWin=wlen;
     }
   CLBufferWrite(p.memWin,win);
   return true;
  }

inline bool CLFFTEnsureBins(CLFFTPlan &p,const int step)
  {
   if(!p.ready || step<=0) return false;
   if(p.memBins!=INVALID_HANDLE && p.lenBins==step) return true;
   if(p.memBins!=INVALID_HANDLE) { CLBufferFree(p.memBins); p.memBins=INVALID_HANDLE; }
   p.memBins=CLBufferCreate(p.ctx,step*sizeof(double),CL_MEM_READ_WRITE);
   if(p.memBins==INVALID_HANDLE) return false;
   p.lenBins=step;
   return true;
  }

inline bool CLFFTEnsureCheck(CLFFTPlan &p)
  {
   if(!p.ready) return false;
   if(p.memCheck!=INVALID_HANDLE) return true;
   p.memCheck=CLBufferCreate(p.ctx,sizeof(double),CL_MEM_READ_WRITE);
   return (p.memCheck!=INVALID_HANDLE);
  }

inline bool CLFFTCheckCOLA(CLFFTPlan &p,const double &win[],const int nperseg,const int noverlap,const double tol,bool &ok)
  {
   if(!p.ready) return false;
   if(nperseg<=0 || noverlap<0 || noverlap>=nperseg) return false;
   int step=nperseg-noverlap;
   if(step<=0) return false;
   if(!CLFFTUploadWin(p,win)) return false;
   if(!CLFFTEnsureBins(p,step)) return false;
   if(!CLFFTEnsureCheck(p)) return false;
   CLSetKernelArgMem(p.kern_cola_bins,0,p.memWin);
   CLSetKernelArg(p.kern_cola_bins,1,nperseg);
   CLSetKernelArg(p.kern_cola_bins,2,step);
   CLSetKernelArgMem(p.kern_cola_bins,3,p.memBins);
   uint offs[1]={0}; uint work[1]={(uint)step};
   if(!CLExecute(p.kern_cola_bins,1,offs,work)) return false;
   CLSetKernelArgMem(p.kern_cola_check,0,p.memBins);
   CLSetKernelArg(p.kern_cola_check,1,step);
   CLSetKernelArg(p.kern_cola_check,2,tol);
   CLSetKernelArgMem(p.kern_cola_check,3,p.memCheck);
   uint work1[1]={1};
   if(!CLExecute(p.kern_cola_check,1,offs,work1)) return false;
   double buf[]; ArrayResize(buf,1);
   CLBufferRead(p.memCheck,buf);
   ok = (buf[0] > 0.5);
   return true;
  }

inline bool CLFFTCheckNOLA(CLFFTPlan &p,const double &win[],const int nperseg,const int noverlap,const double tol,bool &ok)
  {
   if(!p.ready) return false;
   if(nperseg<=0 || noverlap<0 || noverlap>=nperseg) return false;
   int step=nperseg-noverlap;
   if(step<=0) return false;
   if(!CLFFTUploadWin(p,win)) return false;
   if(!CLFFTEnsureBins(p,step)) return false;
   if(!CLFFTEnsureCheck(p)) return false;
   CLSetKernelArgMem(p.kern_nola_bins,0,p.memWin);
   CLSetKernelArg(p.kern_nola_bins,1,nperseg);
   CLSetKernelArg(p.kern_nola_bins,2,step);
   CLSetKernelArgMem(p.kern_nola_bins,3,p.memBins);
   uint offs[1]={0}; uint work[1]={(uint)step};
   if(!CLExecute(p.kern_nola_bins,1,offs,work)) return false;
   CLSetKernelArgMem(p.kern_nola_check,0,p.memBins);
   CLSetKernelArg(p.kern_nola_check,1,step);
   CLSetKernelArg(p.kern_nola_check,2,tol);
   CLSetKernelArgMem(p.kern_nola_check,3,p.memCheck);
   uint work1[1]={1};
   if(!CLExecute(p.kern_nola_check,1,offs,work1)) return false;
   double buf[]; ArrayResize(buf,1);
   CLBufferRead(p.memCheck,buf);
   ok = (buf[0] > 0.5);
   return true;
  }

inline bool CLFFTEnsureCoherenceBuffers(CLFFTPlan &p,const int nfreq)
  {
   if(!p.ready || nfreq<=0) return false;
   if(p.memCohX==INVALID_HANDLE || p.lenCoh!=nfreq)
     {
      if(p.memCohX!=INVALID_HANDLE) { CLBufferFree(p.memCohX); p.memCohX=INVALID_HANDLE; }
      p.memCohX=CLBufferCreate(p.ctx,nfreq*sizeof(double)*2,CL_MEM_READ_WRITE);
      if(p.memCohX==INVALID_HANDLE) return false;
     }
   if(p.memCohPxx==INVALID_HANDLE || p.lenCoh!=nfreq)
     {
      if(p.memCohPxx!=INVALID_HANDLE) { CLBufferFree(p.memCohPxx); p.memCohPxx=INVALID_HANDLE; }
      p.memCohPxx=CLBufferCreate(p.ctx,nfreq*sizeof(double),CL_MEM_READ_WRITE);
      if(p.memCohPxx==INVALID_HANDLE) return false;
     }
   if(p.memCohPyy==INVALID_HANDLE || p.lenCoh!=nfreq)
     {
      if(p.memCohPyy!=INVALID_HANDLE) { CLBufferFree(p.memCohPyy); p.memCohPyy=INVALID_HANDLE; }
      p.memCohPyy=CLBufferCreate(p.ctx,nfreq*sizeof(double),CL_MEM_READ_WRITE);
      if(p.memCohPyy==INVALID_HANDLE) return false;
     }
   if(p.memCohOut==INVALID_HANDLE || p.lenCoh!=nfreq)
     {
      if(p.memCohOut!=INVALID_HANDLE) { CLBufferFree(p.memCohOut); p.memCohOut=INVALID_HANDLE; }
      p.memCohOut=CLBufferCreate(p.ctx,nfreq*sizeof(double),CL_MEM_READ_WRITE);
      if(p.memCohOut==INVALID_HANDLE) return false;
     }
   p.lenCoh=nfreq;
   return true;
  }

inline bool CLFFTComputeCoherenceFromArrays(CLFFTPlan &p,const Complex64 &Pxy[],const double &Pxx[],const double &Pyy[],double &Cxy[])
  {
   int nfreq=ArraySize(Pxy);
   if(nfreq<=0 || ArraySize(Pxx)!=nfreq || ArraySize(Pyy)!=nfreq) return false;
   if(!p.ready) return false;
   if(!CLFFTEnsureCoherenceBuffers(p,nfreq)) return false;
   double bufC[]; ArrayResize(bufC,2*nfreq);
   for(int i=0;i<nfreq;i++){ bufC[2*i]=Pxy[i].re; bufC[2*i+1]=Pxy[i].im; }
   CLBufferWrite(p.memCohX,bufC);
   CLBufferWrite(p.memCohPxx,Pxx);
   CLBufferWrite(p.memCohPyy,Pyy);
   CLSetKernelArgMem(p.kern_coherence,0,p.memCohX);
   CLSetKernelArgMem(p.kern_coherence,1,p.memCohPxx);
   CLSetKernelArgMem(p.kern_coherence,2,p.memCohPyy);
   CLSetKernelArgMem(p.kern_coherence,3,p.memCohOut);
   CLSetKernelArg(p.kern_coherence,4,nfreq);
   uint offs[1]={0}; uint work[1]={(uint)nfreq};
   if(!CLExecute(p.kern_coherence,1,offs,work)) return false;
   ArrayResize(Cxy,nfreq);
   CLBufferRead(p.memCohOut,Cxy);
   return true;
  }

inline bool CLFFTMaxMagIndexFromMem(CLFFTPlan &p,const int memIn,const int n,int &idx,double &maxv)
  {
   if(!p.ready || n<=0) return false;
   if(!CLFFTEnsureOutReal(p,2)) return false;
   CLSetKernelArgMem(p.kern_maxmag,0,memIn);
   CLSetKernelArg(p.kern_maxmag,1,n);
   CLSetKernelArgMem(p.kern_maxmag,2,p.memOutReal);
   uint offs[1]={0}; uint work[1]={1};
   if(!CLExecute(p.kern_maxmag,1,offs,work)) return false;
   double buf[]; ArrayResize(buf,2);
   CLBufferRead(p.memOutReal,buf);
   maxv=buf[0];
   idx=(int)MathRound(buf[1]);
   if(idx<0) idx=0; if(idx>=n) idx=n-1;
   return true;
  }

inline bool CLFFTPSDCompute(CLFFTPlan &p,const int memX,const int memY,const int nseg,const int nfreq,const int nfft,const bool onesided)
  {
   if(!p.ready) return false;
   if(nseg<=0 || nfreq<=0) return false;
   int total=nseg*nfreq;
   if(!CLFFTPSDEnsureBuffers(p,total,nfreq)) return false;
   CLSetKernelArgMem(p.kern_psd_cmul,0,memX);
   CLSetKernelArgMem(p.kern_psd_cmul,1,memY);
   CLSetKernelArg(p.kern_psd_cmul,2,nseg);
   CLSetKernelArg(p.kern_psd_cmul,3,nfreq);
   CLSetKernelArgMem(p.kern_psd_cmul,4,p.memPSD);
   uint offs[1]={0}; uint work[1]={(uint)total};
   if(!CLExecute(p.kern_psd_cmul,1,offs,work)) return false;
   if(onesided)
     {
      int last = ((nfft%2)!=0) ? (nfreq-1) : (nfreq-2);
      if(last>0)
        {
         CLSetKernelArgMem(p.kern_psd_onesided,0,p.memPSD);
         CLSetKernelArg(p.kern_psd_onesided,1,nseg);
         CLSetKernelArg(p.kern_psd_onesided,2,nfreq);
         CLSetKernelArg(p.kern_psd_onesided,3,last);
         if(!CLExecute(p.kern_psd_onesided,1,offs,work)) return false;
        }
     }
   return true;
  }

inline bool CLFFTPSDReduceMean(CLFFTPlan &p,const int nseg,const int nfreq,Complex64 &out[])
  {
   if(!p.ready) return false;
   if(nseg<=0 || nfreq<=0) return false;
   if(!CLFFTPSDEnsureBuffers(p,nseg*nfreq,nfreq)) return false;
   CLSetKernelArgMem(p.kern_psd_mean,0,p.memPSD);
   CLSetKernelArg(p.kern_psd_mean,1,nseg);
   CLSetKernelArg(p.kern_psd_mean,2,nfreq);
   CLSetKernelArgMem(p.kern_psd_mean,3,p.memAvg);
   uint offs[1]={0}; uint work[1]={(uint)nfreq};
   if(!CLExecute(p.kern_psd_mean,1,offs,work)) return false;
   double buf[]; ArrayResize(buf,2*nfreq);
   CLBufferRead(p.memAvg,buf);
   ArrayResize(out,nfreq);
   for(int k=0;k<nfreq;k++) out[k]=Cx(buf[2*k],buf[2*k+1]);
   return true;
  }

inline bool CLFFTPSDReduceMedian(CLFFTPlan &p,const int nseg,const int nfreq,const double bias,Complex64 &out[])
  {
   if(!p.ready) return false;
   if(nseg<=0 || nfreq<=0) return false;
   if(!CLFFTPSDEnsureBuffers(p,nseg*nfreq,nfreq)) return false;
   CLSetKernelArgMem(p.kern_psd_median,0,p.memPSD);
   CLSetKernelArg(p.kern_psd_median,1,nseg);
   CLSetKernelArg(p.kern_psd_median,2,nfreq);
   CLSetKernelArg(p.kern_psd_median,3,bias);
   CLSetKernelArgMem(p.kern_psd_median,4,p.memAvg);
   uint offs[1]={0}; uint work[1]={(uint)nfreq};
   if(!CLExecute(p.kern_psd_median,1,offs,work)) return false;
   double buf[]; ArrayResize(buf,2*nfreq);
   CLBufferRead(p.memAvg,buf);
   ArrayResize(out,nfreq);
   for(int k=0;k<nfreq;k++) out[k]=Cx(buf[2*k],buf[2*k+1]);
   return true;
  }

inline bool CLFFTComputeWinStats(CLFFTPlan &p,double &wsum,double &winpow)
  {
   if(!p.ready || p.memWin==INVALID_HANDLE) return false;
   if(p.memWinStat==INVALID_HANDLE)
     {
      p.memWinStat=CLBufferCreate(p.ctx,2*sizeof(double),CL_MEM_READ_WRITE);
      if(p.memWinStat==INVALID_HANDLE) return false;
     }
   CLSetKernelArgMem(p.kern_win_stats,0,p.memWin);
   CLSetKernelArg(p.kern_win_stats,1,p.lenWin);
   CLSetKernelArgMem(p.kern_win_stats,2,p.memWinStat);
   uint offs[1]={0}; uint work[1]={1};
   if(!CLExecute(p.kern_win_stats,1,offs,work)) return false;
   double buf[2];
   CLBufferRead(p.memWinStat,buf);
   wsum=buf[0]; winpow=buf[1];
   return true;
  }

inline bool CLFFTGenerateFreqs(CLFFTPlan &p,const int nfft,const double fs,const bool onesided,double &freqs[])
  {
   if(!p.ready) return false;
   int nfreq = onesided ? (nfft/2+1) : nfft;
   if(nfreq<=0) return false;
   if(p.memOutReal==INVALID_HANDLE || p.lenOutReal!=nfreq)
     {
      if(p.memOutReal!=INVALID_HANDLE) { CLBufferFree(p.memOutReal); p.memOutReal=INVALID_HANDLE; }
      p.memOutReal=CLBufferCreate(p.ctx,nfreq*sizeof(double),CL_MEM_READ_WRITE);
      if(p.memOutReal==INVALID_HANDLE) return false;
      p.lenOutReal=nfreq;
     }
   CLSetKernelArg(p.kern_gen_freqs,0,nfft);
   CLSetKernelArg(p.kern_gen_freqs,1,fs);
   CLSetKernelArg(p.kern_gen_freqs,2,(int)(onesided?1:0));
   CLSetKernelArgMem(p.kern_gen_freqs,3,p.memOutReal);
   uint offs[1]={0}; uint work[1]={(uint)nfreq};
   if(!CLExecute(p.kern_gen_freqs,1,offs,work)) return false;
   ArrayResize(freqs,nfreq);
   CLBufferRead(p.memOutReal,freqs);
   return true;
  }

inline bool CLFFTGenerateTimes(CLFFTPlan &p,const int nseg,const int seglen,const int noverlap,const double fs,const int boundary_type,double &t[])
  {
   if(!p.ready) return false;
   if(nseg<=0) { ArrayResize(t,0); return true; }
   if(p.memOutReal==INVALID_HANDLE || p.lenOutReal!=nseg)
     {
      if(p.memOutReal!=INVALID_HANDLE) { CLBufferFree(p.memOutReal); p.memOutReal=INVALID_HANDLE; }
      p.memOutReal=CLBufferCreate(p.ctx,nseg*sizeof(double),CL_MEM_READ_WRITE);
      if(p.memOutReal==INVALID_HANDLE) return false;
      p.lenOutReal=nseg;
     }
   CLSetKernelArg(p.kern_gen_times,0,nseg);
   CLSetKernelArg(p.kern_gen_times,1,seglen);
   CLSetKernelArg(p.kern_gen_times,2,noverlap);
   CLSetKernelArg(p.kern_gen_times,3,fs);
   CLSetKernelArg(p.kern_gen_times,4,boundary_type);
   CLSetKernelArgMem(p.kern_gen_times,5,p.memOutReal);
   uint offs[1]={0}; uint work[1]={(uint)nseg};
   if(!CLExecute(p.kern_gen_times,1,offs,work)) return false;
   ArrayResize(t,nseg);
   CLBufferRead(p.memOutReal,t);
   return true;
  }

inline bool CLFFTOverlapAddFromFinal(CLFFTPlan &p,const int nseg,const int nperseg,const int nstep,const int N,const double &win[],const double scale,
                                     Complex64 &out[],double &norm[])
  {
   if(p.memFinal==INVALID_HANDLE) return false;
   if(p.memWin==INVALID_HANDLE || p.lenWin!=nperseg)
     {
      if(p.memWin!=INVALID_HANDLE) CLBufferFree(p.memWin);
      p.memWin=CLBufferCreate(p.ctx,nperseg*sizeof(double),CL_MEM_READ_ONLY);
      if(p.memWin==INVALID_HANDLE) return false;
      p.lenWin=nperseg;
      CLBufferWrite(p.memWin,win);
     }
   int outlen = nperseg + (nseg-1)*nstep;
   if(p.memOutX!=INVALID_HANDLE) CLBufferFree(p.memOutX);
   if(p.memOutNorm!=INVALID_HANDLE) CLBufferFree(p.memOutNorm);
   p.memOutX=CLBufferCreate(p.ctx,outlen*sizeof(double)*2,CL_MEM_READ_WRITE);
   p.memOutNorm=CLBufferCreate(p.ctx,outlen*sizeof(double),CL_MEM_READ_WRITE);
   if(p.memOutX==INVALID_HANDLE || p.memOutNorm==INVALID_HANDLE) return false;

   CLSetKernelArgMem(p.kern_overlap,0,p.memFinal);
   CLSetKernelArgMem(p.kern_overlap,1,p.memWin);
   CLSetKernelArg(p.kern_overlap,2,nseg);
   CLSetKernelArg(p.kern_overlap,3,nperseg);
   CLSetKernelArg(p.kern_overlap,4,nstep);
   CLSetKernelArg(p.kern_overlap,5,N);
   CLSetKernelArg(p.kern_overlap,6,outlen);
   CLSetKernelArg(p.kern_overlap,7,scale);
   CLSetKernelArgMem(p.kern_overlap,8,p.memOutX);
   CLSetKernelArgMem(p.kern_overlap,9,p.memOutNorm);
   uint offs[1]={0}; uint work[1]={(uint)outlen};
   if(!CLExecute(p.kern_overlap,1,offs,work)) return false;

   CLSetKernelArgMem(p.kern_norm,0,p.memOutX);
   CLSetKernelArgMem(p.kern_norm,1,p.memOutNorm);
   CLSetKernelArg(p.kern_norm,2,outlen);
   if(!CLExecute(p.kern_norm,1,offs,work)) return false;

   double buf[];
   ArrayResize(buf,2*outlen);
   ArrayResize(norm,outlen);
   CLBufferRead(p.memOutX,buf);
   CLBufferRead(p.memOutNorm,norm);
   ArrayResize(out,outlen);
   for(int i=0;i<outlen;i++) out[i]=Cx(buf[2*i],buf[2*i+1]);
   return true;
  }

inline bool CLFFTUploadComplexBatch(CLFFTPlan &p,const Complex64 &inFlat[],const int batch)
  {
   if(!p.ready || batch<=0) return false;
   if(!CLFFTEnsureBatchBuffers(p,batch)) return false;
   int N=p.N;
   int total=batch*N;
   double buf[];
   ArrayResize(buf,2*total);
   for(int i=0;i<total;i++)
     {
      buf[2*i]=inFlat[i].re;
      buf[2*i+1]=inFlat[i].im;
     }
   CLBufferWrite(p.memA,buf);
   return true;
  }

inline void CLFFTRealForward(CLFFTPlan &p,const double &x[],Complex64 &out[])
  {
   int N=ArraySize(x);
   Complex64 tmp[];
   ArrayResize(tmp,N);
   for(int i=0;i<N;i++) tmp[i]=Cx(x[i],0.0);
   CLFFTExecute(p,tmp,out,false);
  }

inline void CLFFTRealInverse(CLFFTPlan &p,const Complex64 &Xhalf[],double &out[])
  {
   int Nh=ArraySize(Xhalf);
   int N=(Nh-1)*2;
   Complex64 full[];
   ArrayResize(full,N);
   // fill
   for(int k=0;k<Nh;k++) full[k]=Xhalf[k];
   for(int k=1;k<Nh-1;k++) full[N-k]=Cx(Xhalf[k].re,-Xhalf[k].im);
   Complex64 tmp[];
   CLFFTExecute(p,full,tmp,true);
   ArrayResize(out,N);
   for(int i=0;i<N;i++) out[i]=tmp[i].re;
  }

#endif
