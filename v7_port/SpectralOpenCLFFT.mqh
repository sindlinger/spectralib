#ifndef __SPECTRAL_OPENCL_FFT_MQH__
#define __SPECTRAL_OPENCL_FFT_MQH__

#include "SpectralCommon.mqh"

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
   int memA;
   int memB;
   int memX;
   int memWin;
   int memSum;
   int memOutX;
   int memOutNorm;
   int memFinal;
   int batch;
   int N;
   int lenX;
   int lenWin;
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
   p.memA=INVALID_HANDLE;
   p.memB=INVALID_HANDLE;
   p.memX=INVALID_HANDLE;
   p.memWin=INVALID_HANDLE;
   p.memSum=INVALID_HANDLE;
   p.memOutX=INVALID_HANDLE;
   p.memOutNorm=INVALID_HANDLE;
   p.memFinal=INVALID_HANDLE;
   p.batch=1;
   p.N=0;
   p.lenX=0;
   p.lenWin=0;
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
   if(p.prog!=INVALID_HANDLE) { CLProgramFree(p.prog); p.prog=INVALID_HANDLE; }
   if(p.ctx!=INVALID_HANDLE) { CLContextFree(p.ctx); p.ctx=INVALID_HANDLE; }
   p.N=0; p.batch=1; p.ready=false;
   p.lenX=0; p.lenWin=0;
  }

inline bool CLFFTInit(CLFFTPlan &p,const int N)
  {
   if(p.ready && p.N==N) return true;
   CLFFTFree(p);
   CLFFTReset(p);
   if(N<=1) return false;
   p.ctx=CLContextCreate(CL_USE_GPU_DOUBLE_ONLY);
   if(p.ctx==INVALID_HANDLE) return false;

   string code=
   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
   "inline uint bitrev(uint x, uint bits){\n"
   "  uint y=0; for(uint i=0;i<bits;i++){ y=(y<<1) | (x & 1); x>>=1; } return y; }\n"
   "__kernel void bit_reverse(__global const double2* in, __global double2* out, int N, int bits){\n"
   "  int i=get_global_id(0); if(i>=N) return; uint r=bitrev((uint)i,(uint)bits); out[r]=in[i]; }\n"
   "__kernel void bit_reverse_batch(__global const double2* in, __global double2* out, int N, int bits){\n"
   "  int gid=get_global_id(0); int seg=gid / N; int i=gid - seg*N; if(i>=N) return;\n"
   "  uint r=bitrev((uint)i,(uint)bits); out[seg*N + r]=in[seg*N + i]; }\n"
   "__kernel void fft_stage(__global const double2* in, __global double2* out, int N, int m, int inverse){\n"
   "  int i=get_global_id(0); int half=m>>1; int total=N>>1; if(i>=total) return;\n"
   "  int j=i%half; int block=i/half; int k=block*m + j;\n"
   "  double angle = (inverse? 2.0 : -2.0) * M_PI * (double)j / (double)m;\n"
   "  double c=cos(angle); double s=sin(angle);\n"
   "  double2 a=in[k]; double2 b=in[k+half];\n"
   "  double2 t = (double2)(b.x*c - b.y*s, b.x*s + b.y*c);\n"
   "  out[k] = (double2)(a.x + t.x, a.y + t.y);\n"
   "  out[k+half] = (double2)(a.x - t.x, a.y - t.y);\n"
   "}\n"
   "__kernel void fft_stage_batch(__global const double2* in, __global double2* out, int N, int m, int inverse){\n"
   "  int gid=get_global_id(0); int half=m>>1; int total=N>>1; int seg=gid / total; int i=gid - seg*total; if(i>=total) return;\n"
   "  int j=i%half; int block=i/half; int k=block*m + j; int base=seg*N;\n"
   "  double angle = (inverse? 2.0 : -2.0) * M_PI * (double)j / (double)m;\n"
   "  double c=cos(angle); double s=sin(angle);\n"
   "  double2 a=in[base + k]; double2 b=in[base + k + half];\n"
   "  double2 t = (double2)(b.x*c - b.y*s, b.x*s + b.y*c);\n"
   "  out[base + k] = (double2)(a.x + t.x, a.y + t.y);\n"
   "  out[base + k + half] = (double2)(a.x - t.x, a.y - t.y);\n"
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
   "__kernel void load_real_segment(__global const double* x, __global const double* win, __global double2* out,\n"
   "  int xlen, int start, int nperseg, int nfft){\n"
   "  int i=get_global_id(0); if(i>=nfft) return; double v=0.0;\n"
   "  if(i<nperseg){ int idx=start+i; if(idx>=0 && idx<xlen){ v = x[idx]*win[i]; }}\n"
   "  out[i]=(double2)(v,0.0); }\n"
   "__kernel void load_real_segment_batch(__global const double* x, __global const double* win, __global double2* out,\n"
   "  int xlen, int start0, int step, int nperseg, int nfft){\n"
   "  int gid=get_global_id(0); int seg=gid / nfft; int i=gid - seg*nfft; double v=0.0;\n"
   "  int start = start0 + seg*step;\n"
   "  if(i<nperseg){ int idx=start+i; if(idx>=0 && idx<xlen){ v = x[idx]*win[i]; }}\n"
   "  out[seg*nfft + i]=(double2)(v,0.0); }\n"
   "__kernel void seg_sums(__global const double* x, int xlen, int start, int nperseg, __global double* sumout){\n"
   "  double sumx=0.0; double sumix=0.0; for(int i=0;i<nperseg;i++){\n"
   "    int idx=start+i; if(idx>=0 && idx<xlen){ double v=x[idx]; sumx+=v; sumix+=v*(double)i; }\n"
   "  } sumout[0]=sumx; sumout[1]=sumix; }\n"
   "__kernel void seg_sums_batch(__global const double* x, int xlen, int start0, int step, int nperseg, __global double* sumout){\n"
   "  int seg=get_global_id(0); double sumx=0.0; double sumix=0.0; int start=start0 + seg*step;\n"
   "  for(int i=0;i<nperseg;i++){\n"
   "    int idx=start+i; if(idx>=0 && idx<xlen){ double v=x[idx]; sumx+=v; sumix+=v*(double)i; }\n"
   "  } sumout[2*seg]=sumx; sumout[2*seg+1]=sumix; }\n"
   "__kernel void load_real_segment_detrend(__global const double* x, __global const double* win, __global const double* sumout,\n"
   "  int xlen, int start, int nperseg, int nfft, int detrend_type, double sum_i, double sum_i2, __global double2* out){\n"
   "  int i=get_global_id(0); if(i>=nfft) return; double v=0.0;\n"
   "  if(i<nperseg){ int idx=start+i; if(idx>=0 && idx<xlen){ double xi=x[idx];\n"
   "    if(detrend_type==1){ double mean = sumout[0]/(double)nperseg; xi = xi - mean; }\n"
   "    else if(detrend_type==2){ double n=(double)nperseg; double denom = n*sum_i2 - sum_i*sum_i; double m=0.0;\n"
   "      if(denom!=0.0) m=(n*sumout[1] - sum_i*sumout[0])/denom; double b=(sumout[0]-m*sum_i)/n; xi = xi - (m*(double)i + b); }\n"
   "    v = xi*win[i]; }} out[i]=(double2)(v,0.0); }\n"
   "__kernel void load_real_segment_detrend_batch(__global const double* x, __global const double* win, __global const double* sumout,\n"
   "  int xlen, int start0, int step, int nperseg, int nfft, int detrend_type, double sum_i, double sum_i2, __global double2* out){\n"
   "  int gid=get_global_id(0); int seg=gid / nfft; int i=gid - seg*nfft; double v=0.0; int start=start0 + seg*step;\n"
   "  if(i<nperseg){ int idx=start+i; if(idx>=0 && idx<xlen){ double xi=x[idx];\n"
   "    double s0=sumout[2*seg]; double s1=sumout[2*seg+1];\n"
   "    if(detrend_type==1){ double mean = s0/(double)nperseg; xi = xi - mean; }\n"
   "    else if(detrend_type==2){ double n=(double)nperseg; double denom = n*sum_i2 - sum_i*sum_i; double m=0.0;\n"
   "      if(denom!=0.0) m=(n*s1 - sum_i*s0)/denom; double b=(s0-m*sum_i)/n; xi = xi - (m*(double)i + b); }\n"
   "    v = xi*win[i]; }} out[seg*nfft + i]=(double2)(v,0.0); }\n";
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
   "  int i=get_global_id(0); if(i>=n) return; double d=norm[i]; if(d>1.0e-10){ out[i].x/=d; out[i].y/=d; } }\n";

   p.prog=CLProgramCreate(p.ctx,code);
   if(p.prog==INVALID_HANDLE) { CLFFTFree(p); return false; }
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
   if(p.kern_bitrev==INVALID_HANDLE || p.kern_stage==INVALID_HANDLE || p.kern_scale==INVALID_HANDLE || p.kern_dft==INVALID_HANDLE || p.kern_load==INVALID_HANDLE || p.kern_sum==INVALID_HANDLE || p.kern_load_dt==INVALID_HANDLE ||
      p.kern_bitrev_b==INVALID_HANDLE || p.kern_stage_b==INVALID_HANDLE || p.kern_scale_b==INVALID_HANDLE || p.kern_dft_b==INVALID_HANDLE || p.kern_load_b==INVALID_HANDLE || p.kern_sum_b==INVALID_HANDLE || p.kern_load_dt_b==INVALID_HANDLE ||
      p.kern_overlap==INVALID_HANDLE || p.kern_norm==INVALID_HANDLE)
     { CLFFTFree(p); return false; }
   p.memA=CLBufferCreate(p.ctx,N*sizeof(double)*2,CL_MEM_READ_WRITE);
   p.memB=CLBufferCreate(p.ctx,N*sizeof(double)*2,CL_MEM_READ_WRITE);
   if(p.memA==INVALID_HANDLE || p.memB==INVALID_HANDLE)
     { CLFFTFree(p); return false; }
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

inline bool CLFFTLoadRealSegment(CLFFTPlan &p,const double &x[],const double &win[],const int start,const int nperseg,const int nfft)
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
   uint offs[1]={0};
   uint work[1]={(uint)nfft};
   return CLExecute(p.kern_load,1,offs,work);
  }

inline bool CLFFTLoadRealSegmentDetrend(CLFFTPlan &p,const double &x[],const double &win[],const int start,const int nperseg,const int nfft,const int detrend_type)
  {
   if(!CLFFTInit(p,nfft)) return false;
   if(!CLFFTUpldRealSeries(p,x,win)) return false;
   if(detrend_type==0)
      return CLFFTLoadRealSegment(p,x,win,start,nperseg,nfft);

   // compute sums on GPU (single work-item)
   CLSetKernelArgMem(p.kern_sum,0,p.memX);
   CLSetKernelArg(p.kern_sum,1,p.lenX);
   CLSetKernelArg(p.kern_sum,2,start);
   CLSetKernelArg(p.kern_sum,3,nperseg);
   CLSetKernelArgMem(p.kern_sum,4,p.memSum);
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
   CLSetKernelArgMem(p.kern_load_dt,10,p.memA);
   uint offs[1]={0}; uint work[1]={(uint)nfft};
   return CLExecute(p.kern_load_dt,1,offs,work);
  }

inline bool CLFFTLoadRealSegmentDetrendMem(CLFFTPlan &p,const int start,const int nperseg,const int nfft,const int detrend_type)
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
      uint offs[1]={0}; uint work[1]={(uint)nfft};
      return CLExecute(p.kern_load,1,offs,work);
     }

   // compute sums on GPU (single work-item)
   CLSetKernelArgMem(p.kern_sum,0,p.memX);
   CLSetKernelArg(p.kern_sum,1,p.lenX);
   CLSetKernelArg(p.kern_sum,2,start);
   CLSetKernelArg(p.kern_sum,3,nperseg);
   CLSetKernelArgMem(p.kern_sum,4,p.memSum);
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
   CLSetKernelArgMem(p.kern_load_dt,10,p.memA);
   uint offs[1]={0}; uint work[1]={(uint)nfft};
   return CLExecute(p.kern_load_dt,1,offs,work);
  }

inline bool CLFFTLoadRealSegmentsDetrendBatch(CLFFTPlan &p,const double &x[],const double &win[],const int start0,const int step,const int nperseg,const int nfft,const int detrend_type,const int nseg)
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
      uint work0[1]={(uint)(nseg*nfft)};
      return CLExecute(p.kern_load_b,1,offs,work0);
     }

   // sums per segment
   CLSetKernelArgMem(p.kern_sum_b,0,p.memX);
   CLSetKernelArg(p.kern_sum_b,1,p.lenX);
   CLSetKernelArg(p.kern_sum_b,2,start0);
   CLSetKernelArg(p.kern_sum_b,3,step);
   CLSetKernelArg(p.kern_sum_b,4,nperseg);
   CLSetKernelArgMem(p.kern_sum_b,5,p.memSum);
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
   CLSetKernelArgMem(p.kern_load_dt_b,11,p.memA);
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
   long total = (long)batch * (long)N;
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
   long total=(long)batch*(long)N;
   double buf[];
   ArrayResize(buf,2*total);
   for(long i=0;i<total;i++)
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
