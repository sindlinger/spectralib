//+------------------------------------------------------------------+
//| DominantCycles_STFT_Kalman_Hilbert_SG.mq5                        |
//| Multi-cycle oscillator using STFT + Kalman + Hilbert + SG        |
//+------------------------------------------------------------------+
#property indicator_separate_window
#property indicator_buffers 72
#property indicator_plots 71

#define PI 3.14159265358979323846

enum PriceSource
  {
   PRICE_SRC_CLOSE=0,
   PRICE_SRC_HLC3=1,
   PRICE_SRC_OHLC4=2
  };

enum DetrendMode
  {
   DETREND_KALMAN_LLT=0,
   DETREND_SG_ENDPOINT=1
  };

enum WindowType
  {
   WINDOW_HANN=0,
   WINDOW_BLACKMAN=1
  };

enum SmoothMode
  {
   SMOOTH_CAUSAL=0,
   SMOOTH_REPAINT=1
  };

enum NormMode
  {
   NORM_NONE=0,
   NORM_RMS=1,
   NORM_STD=2
  };

enum PlotMode
  {
   PLOT_SUM_TOP_K=0,
   PLOT_SUM_TOP2=1,
   PLOT_SINGLE_INDEX=2,
   PLOT_MASK=3
  };

// ---- inputs (all are used) ----
input PriceSource InpPriceSource=PRICE_SRC_CLOSE;
input DetrendMode InpDetrendMode=DETREND_KALMAN_LLT;

input double InpLLT_ProcessVar=1e-5;      // Kalman LLT process var (level)
input double InpLLT_SlopeVar=1e-6;        // Kalman LLT process var (slope)
input double InpLLT_MeasureVar=1e-4;      // Kalman LLT measurement var

input int    InpSGDetrendWindow=21;       // SG detrend window (odd)
input int    InpSGDetrendOrder=2;         // SG detrend order

input int    InpSTFTWindow=128;           // STFT window length
input WindowType InpWindowType=WINDOW_HANN;
input int    InpMinPeriodBars=10;
input int    InpMaxPeriodBars=80;
input int    InpBinOversample=2;          // bins per FFT bin

input int    InpKCycles=5;                // 3..7
input int    InpKcand=12;                 // candidate peaks >= 2*K
input int    InpMinPeakSeparation=1;      // min bin separation

input double InpAssocSigmaScale=2.0;      // association sigma scale
input double InpAssocSNRPenalty=0.5;      // SNR penalty in association
input double InpAssocCostThreshold=6.0;   // association cost threshold
input double InpSNRMin=0.5;               // minimum SNR for match

input double InpOmegaProcessVar=1e-4;     // omega process var
input double InpOmegaDotVar=1e-5;         // omega_dot process var
input double InpOmegaMeasureVar=1e-3;     // omega measurement var
input double InpAmpProcessVar=1e-3;       // amplitude process var
input double InpAmpDotVar=1e-4;           // amplitude_dot process var
input double InpAmpMeasureVar=1e-2;       // amplitude measurement var

input int    InpMissBarsToReplace=15;     // missing bars before replace
input double InpConfDecay=0.9;            // confidence decay
input double InpConfMin=0.1;              // minimum confidence

input int    InpHilbertLen=31;            // Hilbert FIR length (odd)
input double InpHilbertBlend=0.5;         // blend weight for Hilbert
input bool   InpEnableHilbert=true;
input double InpResonatorPole=0.95;       // resonator pole (0..1)

input int    InpSGSmoothWindow=11;        // SG smoothing window (odd)
input int    InpSGSmoothOrder=2;          // SG smoothing order
input SmoothMode InpSmoothMode=SMOOTH_CAUSAL;
input bool   InpEnableSmooth=true;

input NormMode InpNormMode=NORM_NONE; // output normalization (explicit)
input int    InpNormWindow=128;       // normalization window (bars)
input double InpNormTarget=1.0;       // target scale after normalization
input double InpPhaseK=1.0;           // phase PLL gain (0..1)
input bool   InpUseOpenCL=true;       // use OpenCL for STFT bins
input int    InpCLDevice=0;           // OpenCL device index
input double InpIQProcessVar=1e-4;    // I/Q process var
input double InpIQMeasureVar=1e-2;    // I/Q measurement var

input PlotMode InpPlotMode=PLOT_SUM_TOP_K;
input int    InpCycleIndex=1;             // rank index (1..7)
input int    InpCycleMask=127;            // bitmask 1..7

input bool   InpCalcOnTick=false;
input int    InpForecastH=3;              // forecast horizon (bars)
input double InpForecastAmpDecay=1.0;     // amplitude decay per bar
input bool   InpForecastShift=false;      // shift forecast plot
input bool   InpShowCycles=false;         // show cycle plots
input bool   InpShowForecastPlot=false;   // show forecast plot
input bool   InpDebugLog=false;           // DEBUG: print time/index/value info

// ---- constants ----
#define MAX_TRACKS 7

#define BUF_MAIN 0
#define BUF_COLOR 1
#define BUF_FORECAST 2
#define BUF_DIR 3
#define BUF_FLIP 4
#define BUF_CYCLE_BASE 5
#define BUFS_PER_CYCLE 9
#define BUF_TREND (BUF_CYCLE_BASE + BUFS_PER_CYCLE*MAX_TRACKS)
#define BUF_RESID (BUF_TREND + 1)
#define BUF_HILBERT_DELAY (BUF_RESID + 1)
#define BUF_SG_DELAY (BUF_RESID + 2)
#define TOTAL_PLOTS 71

// ---- buffers ----
double g_outMain[];
double g_colorIdx[];
double g_forecast[];
double g_dir[];
double g_flip[];
double g_outRaw[];
double g_forecastRaw[];
double g_trend[];
double g_resid[];
double g_hilbertDelayBuf[];
double g_sgDelayBuf[];

// per-cycle buffers (max 7)
struct CycleBuffers
  {
   double cycle[];
   double omega[];
   double period[];
   double amp[];
   double phaseUnw[];
   double phaseW[];
   double snr[];
   double snrDb[];
   double conf[];
   double omegaRaw[];
   double ampRaw[];
   double bpHist[];
   double prevBpHist[];
  };
CycleBuffers g_bufs[MAX_TRACKS];

// ---- precomputed ----
double g_win[];
double g_winSum=1.0;

double g_binOmega[];
double g_binCos[];
double g_binSin[];
int    g_binCount=0;

// ---- OpenCL ----
bool   g_clReady=false;
int    g_clCtx=INVALID_HANDLE;
int    g_clPrg=INVALID_HANDLE;
int    g_clKrn=INVALID_HANDLE;
int    g_clMemX=INVALID_HANDLE;
int    g_clMemCos=INVALID_HANDLE;
int    g_clMemSin=INVALID_HANDLE;
int    g_clMemRe=INVALID_HANDLE;
int    g_clMemIm=INVALID_HANDLE;
int    g_clMemP=INVALID_HANDLE;
int    g_clWinLen=0;
int    g_clBinCount=0;
float  g_clX[];
float  g_clCos[];
float  g_clSin[];
float  g_clRe[];
float  g_clIm[];
float  g_clP[];

// SG coefficients
bool   g_sgDetrendOk=false;
bool   g_sgSmoothOk=false;
double g_sgDetrendCoeff[];
double g_sgSmoothCoeff[];
double g_sgSmoothCoeffSym[];
int    g_sgDetrendW=0;
int    g_sgSmoothW=0;

// Hilbert coefficients
int    g_hilbertW=0;
double g_hilbertCoeff[];
int    g_hilbertDelay=0;

// ---- tracking state (current bar) ----
double g_omegaState[MAX_TRACKS][2];
double g_omegaP[MAX_TRACKS][2][2];
double g_ampState[MAX_TRACKS][2];
double g_ampP[MAX_TRACKS][2][2];
double g_phaseState[MAX_TRACKS];
double g_confState[MAX_TRACKS];
int    g_missCount[MAX_TRACKS];
double g_iState[MAX_TRACKS];
double g_qState[MAX_TRACKS];
double g_iP[MAX_TRACKS];
double g_qP[MAX_TRACKS];

// bandpass/Hilbert history
double g_bpY1[MAX_TRACKS];
double g_bpY2[MAX_TRACKS];
int    g_bpFill[MAX_TRACKS];

// LLT state (current bar)
double g_llt_level=0.0;
double g_llt_slope=0.0;
double g_lltP[2][2];

// previous closed bar state for tick recalculation
bool   g_prevValid=false;
double g_prevOmegaState[MAX_TRACKS][2];
double g_prevOmegaP[MAX_TRACKS][2][2];
double g_prevAmpState[MAX_TRACKS][2];
double g_prevAmpP[MAX_TRACKS][2][2];
double g_prevPhaseState[MAX_TRACKS];
double g_prevConfState[MAX_TRACKS];
int    g_prevMissCount[MAX_TRACKS];
double g_prevBpY1[MAX_TRACKS];
double g_prevBpY2[MAX_TRACKS];
int    g_prevBpFill[MAX_TRACKS];
double g_prevIState[MAX_TRACKS];
double g_prevQState[MAX_TRACKS];
double g_prevIP[MAX_TRACKS];
double g_prevQP[MAX_TRACKS];

double g_prevLltLevel=0.0;
double g_prevLltSlope=0.0;
double g_prevLltP[2][2];

datetime g_lastBarTime=0;
bool g_tracksInitialized=false;

// ---- helpers ----
int MinInt(int a,int b){return a<b?a:b;}
int MaxInt(int a,int b){return a>b?a:b;}
double MaxDouble(double a,double b){return a>b?a:b;}

int PlotIndexFromBuffer(int bufferIndex)
  {
   if(bufferIndex<=BUF_COLOR) return bufferIndex;
   return bufferIndex-1; // color buffer at index 1 does not map to a plot
  }

double WrapPhase(double p)
  {
   while(p>PI) p-=2.0*PI;
   while(p<-PI) p+=2.0*PI;
   return p;
  }

double UnwrapAround(double p,double ref)
  {
   // Bring phase p close to reference (which may be unwrapped)
   double twoPI=2.0*PI;
   double diff=p-ref;
   if(diff>PI || diff<-PI)
     {
      // shift by integer multiples of 2pi to minimize |p-ref|
      double k=MathFloor((diff+PI)/twoPI);
      p-=k*twoPI;
      diff=p-ref;
      if(diff>PI) p-=twoPI;
      else if(diff<-PI) p+=twoPI;
     }
   return p;
  }

void NormalizeAmpPhase(double &amp,double &phase)
  {
   // keep amplitude non-negative; negative amp is equivalent to phase shift of PI
   if(amp<0.0)
     {
      amp=-amp;
      phase+=PI;
     }
  }

double PhaseUpdatePLL(const double pred,const double meas,const double wIn)
  {
   double w=wIn;
   if(w<0.0) w=0.0; if(w>1.0) w=1.0;
   double err=WrapPhase(meas-pred);
   return pred + w*err;
  }

void OpenCLFree()
  {
   if(g_clMemP!=INVALID_HANDLE) { CLBufferFree(g_clMemP); g_clMemP=INVALID_HANDLE; }
   if(g_clMemIm!=INVALID_HANDLE){ CLBufferFree(g_clMemIm); g_clMemIm=INVALID_HANDLE; }
   if(g_clMemRe!=INVALID_HANDLE){ CLBufferFree(g_clMemRe); g_clMemRe=INVALID_HANDLE; }
   if(g_clMemSin!=INVALID_HANDLE){ CLBufferFree(g_clMemSin); g_clMemSin=INVALID_HANDLE; }
   if(g_clMemCos!=INVALID_HANDLE){ CLBufferFree(g_clMemCos); g_clMemCos=INVALID_HANDLE; }
   if(g_clMemX!=INVALID_HANDLE) { CLBufferFree(g_clMemX); g_clMemX=INVALID_HANDLE; }
   if(g_clKrn!=INVALID_HANDLE)  { CLKernelFree(g_clKrn); g_clKrn=INVALID_HANDLE; }
   if(g_clPrg!=INVALID_HANDLE)  { CLProgramFree(g_clPrg); g_clPrg=INVALID_HANDLE; }
   if(g_clCtx!=INVALID_HANDLE)  { CLContextFree(g_clCtx); g_clCtx=INVALID_HANDLE; }
   g_clReady=false;
   g_clWinLen=0;
   g_clBinCount=0;
  }

bool OpenCLInit(const int windowLen,const int binCount)
  {
   if(!InpUseOpenCL) { OpenCLFree(); return false; }
   if(windowLen<8 || binCount<3) return false;
   if(g_clReady && g_clWinLen==windowLen && g_clBinCount==binCount) return true;
   OpenCLFree();

   g_clCtx=CLContextCreate(InpCLDevice);
   if(g_clCtx==INVALID_HANDLE) { g_clReady=false; return false; }

   string clSrc=
      "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\\n"
      "#define N "+IntegerToString(windowLen)+"\\n"
      "__kernel void goertzel(__global const float* x, __global const float* cosw, __global const float* sinw, "
      "__global float* re, __global float* im, __global float* power)\\n"
      "{\\n"
      "  int b = get_global_id(0);\\n"
      "  float coeff = 2.0f * cosw[b];\\n"
      "  float s_prev = 0.0f;\\n"
      "  float s_prev2 = 0.0f;\\n"
      "  for(int n=0;n<N;n++){\\n"
      "    float s = x[n] + coeff*s_prev - s_prev2;\\n"
      "    s_prev2 = s_prev;\\n"
      "    s_prev = s;\\n"
      "  }\\n"
      "  float r = s_prev - s_prev2*cosw[b];\\n"
      "  float ii = s_prev2*sinw[b];\\n"
      "  re[b]=r; im[b]=ii; power[b]=r*r+ii*ii;\\n"
      "}\\n";

   g_clPrg=CLProgramCreate(g_clCtx,clSrc);
   if(g_clPrg==INVALID_HANDLE) { OpenCLFree(); return false; }
   g_clKrn=CLKernelCreate(g_clPrg,"goertzel");
   if(g_clKrn==INVALID_HANDLE) { OpenCLFree(); return false; }

   g_clMemX=CLBufferCreate(g_clCtx,windowLen*sizeof(float),CL_MEM_READ_ONLY);
   g_clMemCos=CLBufferCreate(g_clCtx,binCount*sizeof(float),CL_MEM_READ_ONLY);
   g_clMemSin=CLBufferCreate(g_clCtx,binCount*sizeof(float),CL_MEM_READ_ONLY);
   g_clMemRe=CLBufferCreate(g_clCtx,binCount*sizeof(float),CL_MEM_WRITE_ONLY);
   g_clMemIm=CLBufferCreate(g_clCtx,binCount*sizeof(float),CL_MEM_WRITE_ONLY);
   g_clMemP=CLBufferCreate(g_clCtx,binCount*sizeof(float),CL_MEM_WRITE_ONLY);
   if(g_clMemX==INVALID_HANDLE || g_clMemCos==INVALID_HANDLE || g_clMemSin==INVALID_HANDLE ||
      g_clMemRe==INVALID_HANDLE || g_clMemIm==INVALID_HANDLE || g_clMemP==INVALID_HANDLE)
     { OpenCLFree(); return false; }

   // upload bin cos/sin (float)
   ArrayResize(g_clCos,binCount);
   ArrayResize(g_clSin,binCount);
   for(int i=0;i<binCount;i++){ g_clCos[i]=(float)g_binCos[i]; g_clSin[i]=(float)g_binSin[i]; }
   CLBufferWrite(g_clMemCos,g_clCos);
   CLBufferWrite(g_clMemSin,g_clSin);

   CLSetKernelArgMem(g_clKrn,0,g_clMemX);
   CLSetKernelArgMem(g_clKrn,1,g_clMemCos);
   CLSetKernelArgMem(g_clKrn,2,g_clMemSin);
   CLSetKernelArgMem(g_clKrn,3,g_clMemRe);
   CLSetKernelArgMem(g_clKrn,4,g_clMemIm);
   CLSetKernelArgMem(g_clKrn,5,g_clMemP);

   g_clWinLen=windowLen;
   g_clBinCount=binCount;
   g_clReady=true;
   return true;
  }

bool ComputeBinsOpenCL(const int idx,const int windowLen,const double &resid[],
                       double &binRe[],double &binIm[],double &binPower[],double &sumPower)
  {
   if(!OpenCLInit(windowLen,g_binCount)) return false;
   ArrayResize(g_clX,windowLen);
   for(int n=0;n<windowLen;n++)
     {
      int j=SeriesWindowIndex(idx,windowLen,n);
      g_clX[n]=(float)(resid[j]*g_win[n]);
     }
   CLBufferWrite(g_clMemX,g_clX);

   const uint offs[1]={0};
   const uint works[1]={(uint)g_binCount};
   if(!CLExecute(g_clKrn,1,offs,works)) return false;

   ArrayResize(g_clRe,g_binCount);
   ArrayResize(g_clIm,g_binCount);
   ArrayResize(g_clP,g_binCount);
   CLBufferRead(g_clMemRe,g_clRe);
   CLBufferRead(g_clMemIm,g_clIm);
   CLBufferRead(g_clMemP,g_clP);

   ArrayResize(binRe,g_binCount);
   ArrayResize(binIm,g_binCount);
   ArrayResize(binPower,g_binCount);
   sumPower=0.0;
   for(int b=0;b<g_binCount;b++)
     {
      binRe[b]=(double)g_clRe[b];
      binIm[b]=(double)g_clIm[b];
      binPower[b]=(double)g_clP[b];
      sumPower+=binPower[b];
     }
   return true;
  }

double NormScaleAt(const int i,const int W,const double &arr[],const int rates_total)
  {
   if(i<0 || i>=rates_total) return 0.0;
   if(i+W-1>=rates_total) return 0.0;
   int count=0;
   double mean=0.0;
   double sumsq=0.0;
   for(int j=i;j<i+W;j++)
     {
      double v=arr[j];
      if(v==EMPTY_VALUE) continue;
      count++;
      mean+=v;
      sumsq+=v*v;
     }
   if(count<2) return 0.0;
   mean/=count;
   if(InpNormMode==NORM_STD)
     {
      double var=(sumsq/count) - mean*mean;
      if(var<0.0) var=0.0;
      return MathSqrt(var);
     }
   // RMS (default)
   return MathSqrt(sumsq/count);
  }

void RecomputeDirFlip(const int start,const int end,const int rates_total)
  {
   int maxIdx=MaxInt(start,end);
   int minIdx=MinInt(start,end);
   if(maxIdx>rates_total-1) maxIdx=rates_total-1;
   if(minIdx<0) minIdx=0;
   for(int i=maxIdx;i>=minIdx;i--)
     {
      if(i+1<rates_total && g_outMain[i]!=EMPTY_VALUE && g_outMain[i+1]!=EMPTY_VALUE)
        {
         double d=g_outMain[i]-g_outMain[i+1];
         double dir=(d>0.0)?1.0:((d<0.0)?-1.0:0.0);
         g_dir[i]=dir;
         double prevDir=g_dir[i+1];
         if(dir>0 && prevDir<0) g_flip[i]=1.0;
         else if(dir<0 && prevDir>0) g_flip[i]=-1.0;
         else g_flip[i]=0.0;

         if(dir>0) g_colorIdx[i]=0.0; else if(dir<0) g_colorIdx[i]=1.0; else g_colorIdx[i]=2.0;
        }
      else
        {
         g_dir[i]=0.0;
         g_flip[i]=0.0;
         g_colorIdx[i]=2.0;
   }
  }

void KalmanPredict1(double &x,double &P,double q)
  {
   x=x;
   P=P+q;
  }

void KalmanUpdate1(double &x,double &P,double z,double r)
  {
   double y=z-x;
   double S=P+r;
   double invS=1.0/MaxDouble(S,1e-12);
   double K=P*invS;
   x=x+K*y;
   P=(1.0-K)*P;
  }

void KalmanPredict2(double &x0,double &x1,double &P00,double &P01,double &P10,double &P11,double q0,double q1)
  {
   // x = F*x, F=[[1,1],[0,1]]
   double x0n=x0+x1;
   double x1n=x1;

   double P00n=P00+P01+P10+P11+q0;
   double P01n=P01+P11;
   double P10n=P10+P11;
   double P11n=P11+q1;

   x0=x0n;
   x1=x1n;
   P00=P00n; P01=P01n; P10=P10n; P11=P11n;
  }

void KalmanUpdate2(double &x0,double &x1,double &P00,double &P01,double &P10,double &P11,double z,double r)
  {
   // H=[1,0]
   double y=z-x0;
   double S=P00+r;
   double invS=1.0/MaxDouble(S,1e-12);
   double K0=P00*invS;
   double K1=P10*invS;

   x0=x0+K0*y;
   x1=x1+K1*y;

   double P00n=(1.0-K0)*P00;
   double P01n=(1.0-K0)*P01;
   double P10n=P10-K1*P00;
   double P11n=P11-K1*P01;

   P00=P00n; P01=P01n; P10=P10n; P11=P11n;
  }

// matrix inversion (Gauss-Jordan) for small matrices
bool InvertMatrix(const double &A[],double &Ainv[],int n)
  {
   // A and Ainv are flat arrays size n*n
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
           {
            pivot=tmp[r*(2*n)+i];
            piv=r;
           }
        }
      if(MathAbs(pivot)<1e-12)
         return false;
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
      for(int c=0;c<2*n;c++)
         tmp[i*(2*n)+c]*=invp;
      for(int r=0;r<n;r++)
        {
         if(r==i) continue;
         double f=tmp[r*(2*n)+i];
         if(MathAbs(f)<1e-12) continue;
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

bool BuildSGCoeffs(int window,int order,bool endpoint,double &coeff[])
  {
   if(window<order+2) return false;
   int m=order+1;
   double A[];
   ArrayResize(A,window*m);
   double x0=endpoint ? (double)(window-1) : 0.0;
   for(int i=0;i<window;i++)
     {
      double x=endpoint ? (double)i : (double)(i-(window-1)/2);
      double xp=1.0;
      for(int j=0;j<m;j++)
        {
         A[i*m+j]=xp;
         xp*=x;
        }
     }

   double ATA[];
   ArrayResize(ATA,m*m);
   ArrayInitialize(ATA,0.0);
   for(int r=0;r<m;r++)
     {
      for(int c=0;c<m;c++)
        {
         double sum=0.0;
         for(int i=0;i<window;i++)
            sum+=A[i*m+r]*A[i*m+c];
         ATA[r*m+c]=sum;
        }
     }
   double ATAinv[];
   if(!InvertMatrix(ATA,ATAinv,m)) return false;

   // B = ATAinv * A^T => m x window
   double B[];
   ArrayResize(B,m*window);
   for(int r=0;r<m;r++)
     {
      for(int c=0;c<window;c++)
        {
         double sum=0.0;
         for(int k=0;k<m;k++)
            sum+=ATAinv[r*m+k]*A[c*m+k];
         B[r*window+c]=sum;
        }
     }

   double v[];
   ArrayResize(v,m);
   double xp=1.0;
   for(int j=0;j<m;j++)
     {
      v[j]=xp;
      xp*=x0;
     }

   ArrayResize(coeff,window);
   for(int c=0;c<window;c++)
     {
      double sum=0.0;
      for(int r=0;r<m;r++)
         sum+=v[r]*B[r*window+c];
      coeff[c]=sum;
     }
   return true;
  }

void BuildWindow(int N,WindowType wtype)
  {
   ArrayResize(g_win,N);
   g_winSum=0.0;
   for(int n=0;n<N;n++)
     {
      double w=1.0;
      if(wtype==WINDOW_HANN)
         w=0.5-0.5*MathCos(2.0*PI*n/(N-1));
      else if(wtype==WINDOW_BLACKMAN)
         w=0.42-0.5*MathCos(2.0*PI*n/(N-1))+0.08*MathCos(4.0*PI*n/(N-1));
      g_win[n]=w;
      g_winSum+=w;
     }
   if(g_winSum<1e-12) g_winSum=1.0;
  }

void BuildBins(int windowLen,int minPeriod,int maxPeriod,int oversample)
  {
   if(minPeriod<2) minPeriod=2;
   if(maxPeriod<minPeriod) maxPeriod=minPeriod;
   if(oversample<1) oversample=1;
   int kmax=(int)MathFloor((windowLen/2.0)*oversample);
   ArrayResize(g_binOmega,0);
   ArrayResize(g_binCos,0);
   ArrayResize(g_binSin,0);
   g_binCount=0;
   for(int k=1;k<=kmax;k++)
     {
      double omega=2.0*PI*k/(windowLen*oversample);
      if(omega<=0.0) continue;
      double period=2.0*PI/omega;
      if(period<minPeriod || period>maxPeriod) continue;
      int idx=g_binCount;
      ArrayResize(g_binOmega,idx+1);
      ArrayResize(g_binCos,idx+1);
      ArrayResize(g_binSin,idx+1);
      g_binOmega[idx]=omega;
      g_binCos[idx]=MathCos(omega);
      g_binSin[idx]=MathSin(omega);
      g_binCount++;
     }
  }

void BuildHilbert(int L)
  {
   if(L<3) L=3;
   if((L%2)==0) L++;
   g_hilbertW=L;
   g_hilbertDelay=(L-1)/2;
   ArrayResize(g_hilbertCoeff,L);
   int M=(L-1)/2;
   for(int n=0;n<L;n++)
     {
      int k=n-M;
      double h=0.0;
      if(k==0) h=0.0;
      else if((k%2)!=0)
         h=2.0/(PI*k);
      else
         h=0.0;
      // Hamming window
      double w=0.54-0.46*MathCos(2.0*PI*n/(L-1));
      g_hilbertCoeff[n]=h*w;
     }
  }

void ResetStates()
  {
   g_tracksInitialized=false;
   g_prevValid=false;
   g_lastBarTime=0;

   g_llt_level=0.0;
   g_llt_slope=0.0;
   g_lltP[0][0]=1.0; g_lltP[0][1]=0.0; g_lltP[1][0]=0.0; g_lltP[1][1]=1.0;

   for(int k=0;k<MAX_TRACKS;k++)
     {
      g_omegaState[k][0]=0.0; g_omegaState[k][1]=0.0;
      g_omegaP[k][0][0]=1.0; g_omegaP[k][0][1]=0.0; g_omegaP[k][1][0]=0.0; g_omegaP[k][1][1]=1.0;
      g_ampState[k][0]=0.0; g_ampState[k][1]=0.0;
      g_ampP[k][0][0]=1.0; g_ampP[k][0][1]=0.0; g_ampP[k][1][0]=0.0; g_ampP[k][1][1]=1.0;
      g_phaseState[k]=0.0;
      g_confState[k]=0.0;
      g_missCount[k]=0;
      g_iState[k]=0.0;
      g_qState[k]=0.0;
      g_iP[k]=1.0;
      g_qP[k]=1.0;
      g_bpY1[k]=0.0;
      g_bpY2[k]=0.0;
      g_bpFill[k]=0;
      if(g_hilbertW>0)
        {
         ArrayResize(g_bufs[k].bpHist,g_hilbertW);
         ArrayInitialize(g_bufs[k].bpHist,0.0);
        }
     }
  }

// ---- STFT and peaks ----
int FindPeaks(int idx,int windowLen,const double &resid[],int Kcand,int minSep,
              double &peakOmega[],double &peakAmp[],double &peakPhase[],double &peakPower[],double &peakSNR[])
  {
   if(g_binCount<3) return 0;
   double binPower[];
   double binRe[];
   double binIm[];
   ArrayResize(binPower,g_binCount);
   ArrayResize(binRe,g_binCount);
   ArrayResize(binIm,g_binCount);

   double sumPower=0.0;
   bool usedCL=false;
   if(InpUseOpenCL)
     usedCL=ComputeBinsOpenCL(idx,windowLen,resid,binRe,binIm,binPower,sumPower);
   if(!usedCL)
     {
      for(int b=0;b<g_binCount;b++)
        {
         double omega=g_binOmega[b];
         double coeff=2.0*g_binCos[b];
         double s_prev=0.0, s_prev2=0.0;
         for(int n=0;n<windowLen;n++)
           {
            int j=SeriesWindowIndex(idx,windowLen,n);
            double x=resid[j]*g_win[n];
            double s=x+coeff*s_prev-s_prev2;
            s_prev2=s_prev;
            s_prev=s;
           }
         double re=s_prev-s_prev2*g_binCos[b];
         double im=s_prev2*g_binSin[b];
         double p=re*re+im*im;
         binRe[b]=re;
         binIm[b]=im;
         binPower[b]=p;
         sumPower+=p;
        }
     }

   // collect candidate peaks
   int candIdx[];
   ArrayResize(candIdx,0);
   for(int b=1;b<g_binCount-1;b++)
     {
      if(binPower[b]>binPower[b-1] && binPower[b]>binPower[b+1])
        {
         int n=ArraySize(candIdx);
         ArrayResize(candIdx,n+1);
         candIdx[n]=b;
        }
     }

   // select top Kcand with separation
   int selected=0;
   ArrayResize(peakOmega,Kcand);
   ArrayResize(peakAmp,Kcand);
   ArrayResize(peakPhase,Kcand);
   ArrayResize(peakPower,Kcand);
   ArrayResize(peakSNR,Kcand);

   bool usedCand[];
   ArrayResize(usedCand,ArraySize(candIdx));
   ArrayInitialize(usedCand,false);

   for(int pick=0;pick<Kcand;pick++)
     {
      double bestP=-1.0;
      int bestIdx=-1;
      for(int c=0;c<ArraySize(candIdx);c++)
        {
         if(usedCand[c]) continue;
         int b=candIdx[c];
         double p=binPower[b];
         if(p>bestP)
           {
            // check separation
            bool ok=true;
            for(int s=0;s<selected;s++)
              {
               int bsel=(int)MathRound(peakPower[s]); // stored later as bin index in peakPower temporarily
               if(MathAbs(b-bsel)<minSep)
                 { ok=false; break; }
              }
            if(!ok) continue;
            bestP=p;
            bestIdx=c;
           }
        }
      if(bestIdx<0) break;
      usedCand[bestIdx]=true;
      int b=candIdx[bestIdx];

      // refine omega using parabolic interpolation on log power
      double lp0=MathLog(binPower[b-1]+1e-12);
      double lp1=MathLog(binPower[b]+1e-12);
      double lp2=MathLog(binPower[b+1]+1e-12);
      double denom=(lp0-2.0*lp1+lp2);
      double delta=0.0;
      if(MathAbs(denom)>1e-12)
         delta=0.5*(lp0-lp2)/denom;
      double omega=g_binOmega[b];
      double omegaStep=g_binOmega[b+1]-g_binOmega[b];
      double omegaRef=omega+delta*omegaStep;

      // recompute complex at refined omega
      double coeff=2.0*MathCos(omegaRef);
      double s_prev=0.0, s_prev2=0.0;
      for(int n=0;n<windowLen;n++)
        {
         int j=SeriesWindowIndex(idx,windowLen,n);
         double x=resid[j]*g_win[n];
         double s=x+coeff*s_prev-s_prev2;
         s_prev2=s_prev;
         s_prev=s;
        }
      double re=s_prev-s_prev2*MathCos(omegaRef);
      double im=s_prev2*MathSin(omegaRef);
      double p=re*re+im*im;
      double amp=2.0*MathSqrt(p)/g_winSum;
      double phase=MathArctan2(im,re);
      // align phase to the window end (current bar index)
      phase+=omegaRef*(windowLen-1);

      // SNR
      double meanP=(sumPower-p)/MaxDouble(1.0,(double)(g_binCount-1));
      if(meanP<1e-12) meanP=1e-12;
      double snr=p/meanP;

      // store
      peakOmega[selected]=omegaRef;
      peakAmp[selected]=amp;
      peakPhase[selected]=phase;
      peakSNR[selected]=snr;
      peakPower[selected]=(double)b; // store bin index temporarily for separation check
      selected++;
     }

   // fix peakPower to actual power for outputs
   for(int i=0;i<selected;i++)
     {
      int b=(int)MathRound(peakPower[i]);
      peakPower[i]=binPower[b];
     }

   return selected;
  }

// ---- price source ----
double GetPrice(int i,const double &open[],const double &high[],const double &low[],const double &close[])
  {
   if(InpPriceSource==PRICE_SRC_CLOSE)
      return close[i];
   if(InpPriceSource==PRICE_SRC_HLC3)
      return (high[i]+low[i]+close[i])/3.0;
   // OHLC4
   return (open[i]+high[i]+low[i]+close[i])/4.0;
  }

int SeriesWindowIndex(const int i,const int windowLen,const int n)
  {
   // For series arrays (0 = most recent), map n=0..windowLen-1 to oldest->newest
   return i + (windowLen-1-n);
  }

// ---- debug dump ----
void DebugDump(const datetime &time[],const double &close[],const int rates_total,const int prev_calculated,
               const bool fullRecalc,const bool newBar,const int start,const int end)
  {
   if(rates_total<=0) return;
   int last=rates_total-1;
   int seriesTime=ArrayGetAsSeries(time)?1:0;
   int seriesClose=ArrayGetAsSeries(close)?1:0;
   double c0=close[0];
   double clast=close[last];
   string t0s=TimeToString(time[0],TIME_DATE|TIME_MINUTES|TIME_SECONDS);
   string tlasts=TimeToString(time[last],TIME_DATE|TIME_MINUTES|TIME_SECONDS);
   PrintFormat("DBG series time=%d close=%d | idx0 time=%s close=%.6f | idxLast(%d) time=%s close=%.6f",
               seriesTime,seriesClose,t0s,c0,last,tlasts,clast);
   PrintFormat("DBG rates_total=%d prev_calculated=%d fullRecalc=%d newBar=%d start=%d end=%d",
               rates_total,prev_calculated,(int)fullRecalc,(int)newBar,start,end);
   if(last>=1)
     {
      double c1=close[1];
      double clast1=close[last-1];
      string t1s=TimeToString(time[1],TIME_DATE|TIME_MINUTES|TIME_SECONDS);
      string tlast1s=TimeToString(time[last-1],TIME_DATE|TIME_MINUTES|TIME_SECONDS);
      PrintFormat("DBG check: idx1 time=%s close=%.6f | idxLast-1(%d) time=%s close=%.6f",
                  t1s,c1,last-1,tlast1s,clast1);
     }

   int n=MinInt(10,rates_total);
   string s="DBG last10 (idx/time/close/out):";
   for(int i=0;i<n;i++)
     {
      string ti=TimeToString(time[i],TIME_DATE|TIME_MINUTES);
      s+=StringFormat(" [%d %s %.6f %.6f]",i,ti,close[i],g_outMain[i]);
     }
   Print(s);

   string s2="DBG first10 (oldest idx/time/close/out):";
   for(int j=0;j<n;j++)
     {
      int idx=last-(n-1-j);
      string ti=TimeToString(time[idx],TIME_DATE|TIME_MINUTES);
      s2+=StringFormat(" [%d %s %.6f %.6f]",idx,ti,close[idx],g_outMain[idx]);
     }
   Print(s2);
  }

// ---- init ----
int OnInit()
  {
   int kActive=MaxInt(3,MinInt(7,InpKCycles));

   // ensure odd lengths
   g_sgDetrendW=InpSGDetrendWindow;
   if(g_sgDetrendW<3) g_sgDetrendW=3;
   if((g_sgDetrendW%2)==0) g_sgDetrendW++;

   g_sgSmoothW=InpSGSmoothWindow;
   if(g_sgSmoothW<3) g_sgSmoothW=3;
   if((g_sgSmoothW%2)==0) g_sgSmoothW++;

   BuildWindow(InpSTFTWindow,InpWindowType);
   BuildBins(InpSTFTWindow,InpMinPeriodBars,InpMaxPeriodBars,InpBinOversample);
   BuildHilbert(InpHilbertLen);
   OpenCLInit(InpSTFTWindow,g_binCount);

   g_sgDetrendOk=BuildSGCoeffs(g_sgDetrendW,InpSGDetrendOrder,true,g_sgDetrendCoeff);
   g_sgSmoothOk=BuildSGCoeffs(g_sgSmoothW,InpSGSmoothOrder,true,g_sgSmoothCoeff);
   bool sgSymOk=BuildSGCoeffs(g_sgSmoothW,InpSGSmoothOrder,false,g_sgSmoothCoeffSym);
   if(!sgSymOk) g_sgSmoothOk=false;

   ResetStates();

   // buffers
   SetIndexBuffer(BUF_MAIN,g_outMain,INDICATOR_DATA);
   SetIndexBuffer(BUF_COLOR,g_colorIdx,INDICATOR_COLOR_INDEX);
   SetIndexBuffer(BUF_FORECAST,g_forecast,INDICATOR_DATA);
   SetIndexBuffer(BUF_DIR,g_dir,INDICATOR_DATA);
   SetIndexBuffer(BUF_FLIP,g_flip,INDICATOR_DATA);

   for(int k=0;k<MAX_TRACKS;k++)
     {
      ArraySetAsSeries(g_bufs[k].cycle,true);
      ArraySetAsSeries(g_bufs[k].omega,true);
      ArraySetAsSeries(g_bufs[k].period,true);
      ArraySetAsSeries(g_bufs[k].amp,true);
      ArraySetAsSeries(g_bufs[k].phaseUnw,true);
      ArraySetAsSeries(g_bufs[k].phaseW,true);
      ArraySetAsSeries(g_bufs[k].snr,true);
      ArraySetAsSeries(g_bufs[k].snrDb,true);
      ArraySetAsSeries(g_bufs[k].conf,true);
      ArraySetAsSeries(g_bufs[k].omegaRaw,true);
      ArraySetAsSeries(g_bufs[k].ampRaw,true);

      int base=BUF_CYCLE_BASE+k*BUFS_PER_CYCLE;
      SetIndexBuffer(base+0,g_bufs[k].cycle,INDICATOR_DATA);
      SetIndexBuffer(base+1,g_bufs[k].omega,INDICATOR_DATA);
      SetIndexBuffer(base+2,g_bufs[k].period,INDICATOR_DATA);
      SetIndexBuffer(base+3,g_bufs[k].amp,INDICATOR_DATA);
      SetIndexBuffer(base+4,g_bufs[k].phaseUnw,INDICATOR_DATA);
      SetIndexBuffer(base+5,g_bufs[k].phaseW,INDICATOR_DATA);
      SetIndexBuffer(base+6,g_bufs[k].snr,INDICATOR_DATA);
      SetIndexBuffer(base+7,g_bufs[k].snrDb,INDICATOR_DATA);
      SetIndexBuffer(base+8,g_bufs[k].conf,INDICATOR_DATA);
    }
   SetIndexBuffer(BUF_TREND,g_trend,INDICATOR_DATA);
   SetIndexBuffer(BUF_RESID,g_resid,INDICATOR_DATA);
   SetIndexBuffer(BUF_HILBERT_DELAY,g_hilbertDelayBuf,INDICATOR_DATA);
   SetIndexBuffer(BUF_SG_DELAY,g_sgDelayBuf,INDICATOR_DATA);

   for(int p=0;p<TOTAL_PLOTS;p++)
     {
      PlotIndexSetInteger(p,PLOT_DRAW_TYPE,DRAW_NONE);
      PlotIndexSetInteger(p,PLOT_SHOW_DATA,false);
     }

   int plotMain=PlotIndexFromBuffer(BUF_MAIN);
   PlotIndexSetInteger(plotMain,PLOT_DRAW_TYPE,DRAW_COLOR_LINE);
   PlotIndexSetInteger(plotMain,PLOT_COLOR_INDEXES,3);
   PlotIndexSetInteger(plotMain,PLOT_LINE_COLOR,0,clrLime);
   PlotIndexSetInteger(plotMain,PLOT_LINE_COLOR,1,clrRed);
   PlotIndexSetInteger(plotMain,PLOT_LINE_COLOR,2,clrSilver);
   PlotIndexSetInteger(plotMain,PLOT_LINE_WIDTH,2);
   PlotIndexSetInteger(plotMain,PLOT_SHOW_DATA,true);

   int plotForecast=PlotIndexFromBuffer(BUF_FORECAST);
   PlotIndexSetInteger(plotForecast,PLOT_DRAW_TYPE,InpShowForecastPlot?DRAW_LINE:DRAW_NONE);
   PlotIndexSetInteger(plotForecast,PLOT_LINE_COLOR,clrDodgerBlue);
   PlotIndexSetInteger(plotForecast,PLOT_LINE_STYLE,STYLE_DASH);
   PlotIndexSetInteger(plotForecast,PLOT_LINE_WIDTH,1);
   PlotIndexSetInteger(plotForecast,PLOT_SHOW_DATA,InpShowForecastPlot);
   int fshift=InpForecastShift?MaxInt(1,InpForecastH):0;
   PlotIndexSetInteger(plotForecast,PLOT_SHIFT,fshift);

   for(int k=0;k<MAX_TRACKS;k++)
     {
      int base=BUF_CYCLE_BASE+k*BUFS_PER_CYCLE;
      int plotCycle=PlotIndexFromBuffer(base);
      PlotIndexSetInteger(plotCycle,PLOT_DRAW_TYPE,InpShowCycles?DRAW_LINE:DRAW_NONE);
      PlotIndexSetInteger(plotCycle,PLOT_LINE_COLOR,(color)(0x0050A0FF-0x00080808*k));
      PlotIndexSetInteger(plotCycle,PLOT_LINE_WIDTH,1);
     }

   ArraySetAsSeries(g_outMain,true);
   ArraySetAsSeries(g_colorIdx,true);
   ArraySetAsSeries(g_forecast,true);
   ArraySetAsSeries(g_dir,true);
   ArraySetAsSeries(g_flip,true);
   ArraySetAsSeries(g_outRaw,true);
   ArraySetAsSeries(g_forecastRaw,true);
   ArraySetAsSeries(g_trend,true);
   ArraySetAsSeries(g_resid,true);
   ArraySetAsSeries(g_hilbertDelayBuf,true);
   ArraySetAsSeries(g_sgDelayBuf,true);

   IndicatorSetString(INDICATOR_SHORTNAME,"DominantCycles_STFT_Kalman_Hilbert_SG");
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   OpenCLFree();
  }

// ---- OnCalculate ----
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   if(rates_total<InpSTFTWindow)
      return 0;

   // ensure series orientation (index 0 = most recent)
   ArraySetAsSeries(time,true);
   ArraySetAsSeries(open,true);
   ArraySetAsSeries(high,true);
   ArraySetAsSeries(low,true);
   ArraySetAsSeries(close,true);
   ArraySetAsSeries(tick_volume,true);
   ArraySetAsSeries(volume,true);
   ArraySetAsSeries(spread,true);

   bool fullRecalc=(prev_calculated==0 || rates_total<prev_calculated);
   bool newBar=(time[0]!=g_lastBarTime);

   int start=fullRecalc ? rates_total-1 : (InpCalcOnTick ? 0 : 1);
   int end=InpCalcOnTick ? 0 : 1;
   if(rates_total<=1) { start=0; end=0; }

   if(!InpCalcOnTick && !newBar && !fullRecalc)
     {
      if(InpDebugLog)
         DebugDump(time,close,rates_total,prev_calculated,fullRecalc,newBar,start,end);
      return rates_total;
     }

   int kActive=MaxInt(3,MinInt(7,InpKCycles));
   int kcand=MaxInt(InpKcand,2*kActive);
   int minSep=MaxInt(1,InpMinPeakSeparation);

   if(fullRecalc)
     {
      ResetStates();
      OpenCLInit(InpSTFTWindow,g_binCount);
      // resize raw arrays
      for(int k=0;k<MAX_TRACKS;k++)
        {
         ArrayResize(g_bufs[k].omegaRaw,rates_total);
         ArrayResize(g_bufs[k].ampRaw,rates_total);
         ArraySetAsSeries(g_bufs[k].omegaRaw,true);
         ArraySetAsSeries(g_bufs[k].ampRaw,true);
         ArrayInitialize(g_bufs[k].omegaRaw,0.0);
         ArrayInitialize(g_bufs[k].ampRaw,0.0);
        }
      ArrayResize(g_outRaw,rates_total);
      ArrayResize(g_forecastRaw,rates_total);
      ArraySetAsSeries(g_outRaw,true);
      ArraySetAsSeries(g_forecastRaw,true);
      ArrayInitialize(g_outRaw,EMPTY_VALUE);
      ArrayInitialize(g_forecastRaw,EMPTY_VALUE);
     }
   else if(newBar)
     {
      // shift raw arrays for new bar
      for(int k=0;k<MAX_TRACKS;k++)
        {
         int sz=ArraySize(g_bufs[k].omegaRaw);
         int need=rates_total;
         if(need<1) need=1;
         if(sz<need)
           {
            ArrayResize(g_bufs[k].omegaRaw,need);
            ArrayResize(g_bufs[k].ampRaw,need);
            ArraySetAsSeries(g_bufs[k].omegaRaw,true);
            ArraySetAsSeries(g_bufs[k].ampRaw,true);
           }
         int shift_sz=sz;
         if(shift_sz>need) shift_sz=need;
         for(int i=shift_sz-1;i>0;i--)
           {
            g_bufs[k].omegaRaw[i]=g_bufs[k].omegaRaw[i-1];
            g_bufs[k].ampRaw[i]=g_bufs[k].ampRaw[i-1];
           }
         g_bufs[k].omegaRaw[0]=0.0;
         g_bufs[k].ampRaw[0]=0.0;
        }

      // shift output raw arrays for new bar
      int szOut=ArraySize(g_outRaw);
      int needOut=rates_total;
      if(needOut<1) needOut=1;
      if(szOut<needOut)
        {
         ArrayResize(g_outRaw,needOut);
         ArrayResize(g_forecastRaw,needOut);
         ArraySetAsSeries(g_outRaw,true);
         ArraySetAsSeries(g_forecastRaw,true);
        }
      int shiftOut=szOut;
      if(shiftOut>needOut) shiftOut=needOut;
      for(int i=shiftOut-1;i>0;i--)
        {
         g_outRaw[i]=g_outRaw[i-1];
         g_forecastRaw[i]=g_forecastRaw[i-1];
        }
      g_outRaw[0]=EMPTY_VALUE;
      g_forecastRaw[0]=EMPTY_VALUE;

      // store prev state for tick recalculation
      g_prevValid=true;
      g_prevLltLevel=g_llt_level;
      g_prevLltSlope=g_llt_slope;
      g_prevLltP[0][0]=g_lltP[0][0]; g_prevLltP[0][1]=g_lltP[0][1];
      g_prevLltP[1][0]=g_lltP[1][0]; g_prevLltP[1][1]=g_lltP[1][1];

      for(int k=0;k<MAX_TRACKS;k++)
        {
         g_prevOmegaState[k][0]=g_omegaState[k][0]; g_prevOmegaState[k][1]=g_omegaState[k][1];
         g_prevOmegaP[k][0][0]=g_omegaP[k][0][0]; g_prevOmegaP[k][0][1]=g_omegaP[k][0][1];
         g_prevOmegaP[k][1][0]=g_omegaP[k][1][0]; g_prevOmegaP[k][1][1]=g_omegaP[k][1][1];

         g_prevAmpState[k][0]=g_ampState[k][0]; g_prevAmpState[k][1]=g_ampState[k][1];
         g_prevAmpP[k][0][0]=g_ampP[k][0][0]; g_prevAmpP[k][0][1]=g_ampP[k][0][1];
         g_prevAmpP[k][1][0]=g_ampP[k][1][0]; g_prevAmpP[k][1][1]=g_ampP[k][1][1];

         g_prevPhaseState[k]=g_phaseState[k];
         g_prevConfState[k]=g_confState[k];
         g_prevMissCount[k]=g_missCount[k];
         g_prevIState[k]=g_iState[k];
         g_prevQState[k]=g_qState[k];
         g_prevIP[k]=g_iP[k];
         g_prevQP[k]=g_qP[k];

         g_prevBpY1[k]=g_bpY1[k];
         g_prevBpY2[k]=g_bpY2[k];
         g_prevBpFill[k]=g_bpFill[k];

         int L=g_hilbertW;
         ArrayResize(g_bufs[k].prevBpHist,L);
         for(int n=0;n<L;n++)
            g_bufs[k].prevBpHist[n]=g_bufs[k].bpHist[n];
        }
     }
   else if(InpCalcOnTick && g_prevValid)
     {
      // same bar: restore state from previous closed bar
      g_llt_level=g_prevLltLevel;
      g_llt_slope=g_prevLltSlope;
      g_lltP[0][0]=g_prevLltP[0][0]; g_lltP[0][1]=g_prevLltP[0][1];
      g_lltP[1][0]=g_prevLltP[1][0]; g_lltP[1][1]=g_prevLltP[1][1];

      for(int k=0;k<MAX_TRACKS;k++)
        {
         g_omegaState[k][0]=g_prevOmegaState[k][0]; g_omegaState[k][1]=g_prevOmegaState[k][1];
         g_omegaP[k][0][0]=g_prevOmegaP[k][0][0]; g_omegaP[k][0][1]=g_prevOmegaP[k][0][1];
         g_omegaP[k][1][0]=g_prevOmegaP[k][1][0]; g_omegaP[k][1][1]=g_prevOmegaP[k][1][1];

         g_ampState[k][0]=g_prevAmpState[k][0]; g_ampState[k][1]=g_prevAmpState[k][1];
         g_ampP[k][0][0]=g_prevAmpP[k][0][0]; g_ampP[k][0][1]=g_prevAmpP[k][0][1];
         g_ampP[k][1][0]=g_prevAmpP[k][1][0]; g_ampP[k][1][1]=g_prevAmpP[k][1][1];

         g_phaseState[k]=g_prevPhaseState[k];
         g_confState[k]=g_prevConfState[k];
         g_missCount[k]=g_prevMissCount[k];
         g_iState[k]=g_prevIState[k];
         g_qState[k]=g_prevQState[k];
         g_iP[k]=g_prevIP[k];
         g_qP[k]=g_prevQP[k];

         g_bpY1[k]=g_prevBpY1[k];
         g_bpY2[k]=g_prevBpY2[k];
         g_bpFill[k]=g_prevBpFill[k];
         int L=g_hilbertW;
         ArrayResize(g_bufs[k].bpHist,L);
         for(int n=0;n<L;n++)
            g_bufs[k].bpHist[n]=g_bufs[k].prevBpHist[n];
        }
     }

   for(int i=start;i>=end;i--)
     {
      // compute price, trend, resid
      double price=GetPrice(i,open,high,low,close);
      double trend=price;
      if(InpDetrendMode==DETREND_KALMAN_LLT)
        {
         // Kalman LLT
         if(i==rates_total-1)
           {
            g_llt_level=price;
            g_llt_slope=0.0;
            g_lltP[0][0]=1.0; g_lltP[0][1]=0.0; g_lltP[1][0]=0.0; g_lltP[1][1]=1.0;
           }
         KalmanPredict2(g_llt_level,g_llt_slope,g_lltP[0][0],g_lltP[0][1],g_lltP[1][0],g_lltP[1][1],
                        InpLLT_ProcessVar,InpLLT_SlopeVar);
         KalmanUpdate2(g_llt_level,g_llt_slope,g_lltP[0][0],g_lltP[0][1],g_lltP[1][0],g_lltP[1][1],
                       price,InpLLT_MeasureVar);
         trend=g_llt_level;
        }
      else
        {
         // SG endpoint detrend
         if(g_sgDetrendOk && i+g_sgDetrendW-1<rates_total)
           {
            double acc=0.0;
            for(int n=0;n<g_sgDetrendW;n++)
               acc+=g_sgDetrendCoeff[n]*GetPrice(SeriesWindowIndex(i,g_sgDetrendW,n),open,high,low,close);
            trend=acc;
           }
         else
            trend=price;
        }

      double resid=price-trend;
      g_trend[i]=trend;
      g_resid[i]=resid;

      g_hilbertDelayBuf[i]=(double)g_hilbertDelay;
      g_sgDelayBuf[i]=(InpSmoothMode==SMOOTH_REPAINT)?(double)((g_sgSmoothW-1)/2):0.0;

      // not enough data for STFT
      if(i>rates_total-InpSTFTWindow)
        {
         g_outMain[i]=EMPTY_VALUE;
         g_forecast[i]=EMPTY_VALUE;
         g_dir[i]=0.0;
         g_flip[i]=0.0;
         g_colorIdx[i]=2.0;

         for(int k=0;k<MAX_TRACKS;k++)
           {
            g_bufs[k].cycle[i]=EMPTY_VALUE;
            g_bufs[k].omega[i]=EMPTY_VALUE;
            g_bufs[k].period[i]=EMPTY_VALUE;
            g_bufs[k].amp[i]=EMPTY_VALUE;
            g_bufs[k].phaseUnw[i]=EMPTY_VALUE;
            g_bufs[k].phaseW[i]=EMPTY_VALUE;
            g_bufs[k].snr[i]=EMPTY_VALUE;
            g_bufs[k].snrDb[i]=EMPTY_VALUE;
            g_bufs[k].conf[i]=EMPTY_VALUE;
           }
         continue;
        }

      // STFT peaks
      double peakOmega[];
      double peakAmp[];
      double peakPhase[];
      double peakPower[];
      double peakSNR[];
      int peakCount=FindPeaks(i,InpSTFTWindow,g_resid,kcand,minSep,
                              peakOmega,peakAmp,peakPhase,peakPower,peakSNR);

      // initialize tracks on first valid bar
      if(!g_tracksInitialized)
        {
         for(int k=0;k<kActive;k++)
           {
            if(k<peakCount)
              {
               g_omegaState[k][0]=peakOmega[k];
               g_omegaState[k][1]=0.0;
               g_ampState[k][0]=peakAmp[k];
               g_ampState[k][1]=0.0;
               g_phaseState[k]=peakPhase[k];
               g_confState[k]=peakSNR[k]/(peakSNR[k]+1.0);
               g_missCount[k]=0;
              }
            else
              {
               g_omegaState[k][0]=0.0;
               g_omegaState[k][1]=0.0;
               g_ampState[k][0]=0.0;
               g_ampState[k][1]=0.0;
               g_phaseState[k]=0.0;
               g_confState[k]=0.0;
               g_missCount[k]=0;
              }
           }
         g_tracksInitialized=true;
      }

      // predict and prepare Hilbert measurements
      double hilbertPhase[MAX_TRACKS];
      double hilbertAmp[MAX_TRACKS];
      bool hilbertValid[MAX_TRACKS];
      ArrayInitialize(hilbertValid,false);

      for(int k=0;k<kActive;k++)
        {
         // omega/amp prediction
         KalmanPredict2(g_omegaState[k][0],g_omegaState[k][1],
                        g_omegaP[k][0][0],g_omegaP[k][0][1],g_omegaP[k][1][0],g_omegaP[k][1][1],
                        InpOmegaProcessVar,InpOmegaDotVar);
         KalmanPredict2(g_ampState[k][0],g_ampState[k][1],
                        g_ampP[k][0][0],g_ampP[k][0][1],g_ampP[k][1][0],g_ampP[k][1][1],
                        InpAmpProcessVar,InpAmpDotVar);

         // I/Q demodulation for phase (robust quadrature)
         double omegaPred=g_omegaState[k][0];
         double phasePred=g_phaseState[k]+omegaPred;
         double iMeas=resid*MathCos(phasePred);
         double qMeas=-resid*MathSin(phasePred);
         KalmanPredict1(g_iState[k],g_iP[k],InpIQProcessVar);
         KalmanPredict1(g_qState[k],g_qP[k],InpIQProcessVar);
         KalmanUpdate1(g_iState[k],g_iP[k],iMeas,InpIQMeasureVar);
         KalmanUpdate1(g_qState[k],g_qP[k],qMeas,InpIQMeasureVar);

         // bandpass resonator using predicted omega (Hilbert assist)
         double r=InpResonatorPole;
         if(r<0.01) r=0.01;
         if(r>0.999) r=0.999;
         double a1=2.0*r*MathCos(omegaPred);
         double a2=-r*r;
         double y=(1.0-r)*resid + a1*g_bpY1[k] + a2*g_bpY2[k];
         g_bpY2[k]=g_bpY1[k];
         g_bpY1[k]=y;

         int L=g_hilbertW;
         if(ArraySize(g_bufs[k].bpHist)!=L)
            ArrayResize(g_bufs[k].bpHist,L);
         for(int n=L-1;n>0;n--)
            g_bufs[k].bpHist[n]=g_bufs[k].bpHist[n-1];
         g_bufs[k].bpHist[0]=y;
         if(g_bpFill[k]<L) g_bpFill[k]++;

         if(InpEnableHilbert && g_bpFill[k]>=L)
           {
            double im=0.0;
            for(int n=0;n<L;n++)
               im+=g_hilbertCoeff[n]*g_bufs[k].bpHist[n];
            double re=g_bufs[k].bpHist[g_hilbertDelay];
            double ph=MathArctan2(im,re);
            // compensate Hilbert group delay using causal prediction
            ph+=omegaPred*g_hilbertDelay;
            hilbertPhase[k]=ph;
            hilbertAmp[k]=MathSqrt(re*re+im*im);
            hilbertValid[k]=true;
           }
        }

      // data association
      int assignPeak[MAX_TRACKS];
      for(int k=0;k<MAX_TRACKS;k++) assignPeak[k]=-1;
      bool peakUsed[];
      ArrayResize(peakUsed,peakCount);
      ArrayInitialize(peakUsed,false);

      // order tracks by confidence
      int trackOrder[MAX_TRACKS];
      for(int k=0;k<kActive;k++) trackOrder[k]=k;
      for(int a=0;a<kActive-1;a++)
        {
         int best=a;
         for(int b=a+1;b<kActive;b++)
           {
            if(g_confState[trackOrder[b]]>g_confState[trackOrder[best]])
               best=b;
           }
         if(best!=a)
           {
            int tmp=trackOrder[a];
            trackOrder[a]=trackOrder[best];
            trackOrder[best]=tmp;
           }
        }

      for(int oi=0;oi<kActive;oi++)
        {
         int k=trackOrder[oi];
         double omegaPred=g_omegaState[k][0];
         double sigma=MathSqrt(MathMax(g_omegaP[k][0][0],1e-12))*InpAssocSigmaScale;
         double bestCost=1e50;
         int bestPeak=-1;

         for(int p=0;p<peakCount;p++)
           {
            if(peakUsed[p]) continue;
            double d=MathAbs(peakOmega[p]-omegaPred)/MathMax(sigma,1e-6);
            double snr=peakSNR[p];
            double cost=d + InpAssocSNRPenalty/(snr+1e-6);
            if(cost<bestCost)
              {
               bestCost=cost;
               bestPeak=p;
              }
           }
         if(bestPeak>=0 && bestCost<=InpAssocCostThreshold && peakSNR[bestPeak]>=InpSNRMin)
           {
            assignPeak[k]=bestPeak;
            peakUsed[bestPeak]=true;
           }
        }

      // update tracks
      double minOmega=2.0*PI/MathMax(2.0,(double)InpMaxPeriodBars);
      double maxOmega=2.0*PI/MathMax(2.0,(double)InpMinPeriodBars);
      for(int k=0;k<kActive;k++)
        {
         int p=assignPeak[k];
         double omegaPred=g_omegaState[k][0];
         double phasePred=g_phaseState[k]+omegaPred; // dt=1 bar
         double phaseIQ=MathArctan2(g_qState[k],g_iState[k]);
         double phaseMeasIQ=phasePred+phaseIQ;
         if(p>=0)
           {
            double snr=peakSNR[p];
            double measOmega=peakOmega[p];
            double measAmp=peakAmp[p];

            if(InpEnableHilbert && hilbertValid[k])
              {
               double snrWeight=snr/(snr+1.0);
               double confWeight=g_confState[k];
               double w=InpHilbertBlend*0.5*(snrWeight+confWeight);
               if(w<0.0) w=0.0; if(w>1.0) w=1.0;
               measAmp=(1.0-w)*measAmp + w*hilbertAmp[k];
              }

            double rOmega=InpOmegaMeasureVar/MaxDouble(1.0,snr);
            double rAmp=InpAmpMeasureVar/MaxDouble(1.0,snr);

            KalmanUpdate2(g_omegaState[k][0],g_omegaState[k][1],
                          g_omegaP[k][0][0],g_omegaP[k][0][1],g_omegaP[k][1][0],g_omegaP[k][1][1],
                          measOmega,rOmega);
            KalmanUpdate2(g_ampState[k][0],g_ampState[k][1],
                          g_ampP[k][0][0],g_ampP[k][0][1],g_ampP[k][1][0],g_ampP[k][1][1],
                          measAmp,rAmp);

            // robust phase from I/Q quadrature
            double wPhase=InpPhaseK;
            g_phaseState[k]=PhaseUpdatePLL(phasePred,phaseMeasIQ,wPhase);
            g_confState[k]=MathMin(1.0,g_confState[k]*InpConfDecay + (snr/(snr+1.0)));
            g_missCount[k]=0;
           }
         else
           {
            double wPhase=InpPhaseK;
            g_phaseState[k]=PhaseUpdatePLL(phasePred,phaseMeasIQ,wPhase);
            g_confState[k]=g_confState[k]*InpConfDecay;
            g_missCount[k]++;
           }

         // constrain omega and amplitude to stable ranges
         double o=g_omegaState[k][0];
         if(o<0.0) o=-o;
         if(o<minOmega) o=minOmega;
         if(o>maxOmega) o=maxOmega;
         g_omegaState[k][0]=o;
         NormalizeAmpPhase(g_ampState[k][0],g_phaseState[k]);

         // replacement
         if(g_missCount[k]>=InpMissBarsToReplace || g_confState[k]<InpConfMin)
           {
            // find best unassigned peak
            int best=-1;
            double bestS=-1.0;
            for(int p2=0;p2<peakCount;p2++)
              {
               if(peakUsed[p2]) continue;
               if(peakSNR[p2]>bestS)
                 { bestS=peakSNR[p2]; best=p2; }
              }
            if(best>=0)
              {
               peakUsed[best]=true;
               g_omegaState[k][0]=peakOmega[best];
               g_omegaState[k][1]=0.0;
               g_ampState[k][0]=peakAmp[best];
               g_ampState[k][1]=0.0;
               g_phaseState[k]=peakPhase[best];
               NormalizeAmpPhase(g_ampState[k][0],g_phaseState[k]);
               g_confState[k]=peakSNR[best]/(peakSNR[best]+1.0);
               g_missCount[k]=0;
               g_omegaP[k][0][0]=1.0; g_omegaP[k][0][1]=0.0; g_omegaP[k][1][0]=0.0; g_omegaP[k][1][1]=1.0;
               g_ampP[k][0][0]=1.0; g_ampP[k][0][1]=0.0; g_ampP[k][1][0]=0.0; g_ampP[k][1][1]=1.0;
               g_bpY1[k]=0.0; g_bpY2[k]=0.0; g_bpFill[k]=0;
               g_iState[k]=0.0; g_qState[k]=0.0; g_iP[k]=1.0; g_qP[k]=1.0;
              }
           }
        }

      // output per cycle
      for(int k=0;k<MAX_TRACKS;k++)
        {
         if(k>=kActive)
           {
            g_bufs[k].cycle[i]=EMPTY_VALUE;
            g_bufs[k].omega[i]=EMPTY_VALUE;
            g_bufs[k].period[i]=EMPTY_VALUE;
            g_bufs[k].amp[i]=EMPTY_VALUE;
            g_bufs[k].phaseUnw[i]=EMPTY_VALUE;
            g_bufs[k].phaseW[i]=EMPTY_VALUE;
            g_bufs[k].snr[i]=EMPTY_VALUE;
            g_bufs[k].snrDb[i]=EMPTY_VALUE;
            g_bufs[k].conf[i]=EMPTY_VALUE;
            continue;
           }

         double omega=g_omegaState[k][0];
         double amp=g_ampState[k][0];
         double phase=g_phaseState[k];

         g_bufs[k].omegaRaw[i]=omega;
         g_bufs[k].ampRaw[i]=amp;

         // SG smoothing (causal)
         if(InpEnableSmooth && g_sgSmoothOk && InpSmoothMode==SMOOTH_CAUSAL && i+g_sgSmoothW-1<rates_total)
           {
            double accO=0.0, accA=0.0;
            for(int n=0;n<g_sgSmoothW;n++)
              {
               int j=SeriesWindowIndex(i,g_sgSmoothW,n);
               accO+=g_sgSmoothCoeff[n]*g_bufs[k].omegaRaw[j];
               accA+=g_sgSmoothCoeff[n]*g_bufs[k].ampRaw[j];
              }
            omega=accO;
            amp=accA;
           }

         g_bufs[k].omega[i]=omega;
         g_bufs[k].period[i]=(MathAbs(omega)>1e-6)?(2.0*PI/omega):0.0;
         g_bufs[k].amp[i]=amp;
         g_bufs[k].phaseUnw[i]=phase;
         g_bufs[k].phaseW[i]=WrapPhase(phase);

         double snr=0.0;
         int p=assignPeak[k];
         if(p>=0) snr=peakSNR[p];
         g_bufs[k].snr[i]=snr;
         g_bufs[k].snrDb[i]=10.0*MathLog10(snr+1e-12);
         g_bufs[k].conf[i]=g_confState[k];

         g_bufs[k].cycle[i]=amp*MathCos(phase);
        }

      // ranking for output sum
      int rankIdx[MAX_TRACKS];
      bool usedRank[MAX_TRACKS];
      ArrayInitialize(usedRank,false);
      for(int r=0;r<kActive;r++)
        {
         double bestScore=-1.0;
         int best=-1;
         for(int k=0;k<kActive;k++)
           {
            if(usedRank[k]) continue;
            double score=MathAbs(g_bufs[k].amp[i])*g_confState[k];
            if(score>bestScore)
              { bestScore=score; best=k; }
           }
         rankIdx[r]=best;
         if(best>=0) usedRank[best]=true;
        }

      double sum=0.0;
      if(InpPlotMode==PLOT_SUM_TOP_K)
        {
         for(int r=0;r<kActive;r++)
           {
            int k=rankIdx[r];
            if(k<0) continue;
            sum+=g_bufs[k].cycle[i];
           }
        }
      else if(InpPlotMode==PLOT_SUM_TOP2)
        {
         for(int r=0;r<MinInt(2,kActive);r++)
           {
            int k=rankIdx[r];
            if(k<0) continue;
            sum+=g_bufs[k].cycle[i];
           }
        }
      else if(InpPlotMode==PLOT_SINGLE_INDEX)
        {
         int idxSel=InpCycleIndex-1;
         if(idxSel>=0 && idxSel<kActive)
           {
            int k=rankIdx[idxSel];
            if(k>=0) sum=g_bufs[k].cycle[i];
           }
        }
      else if(InpPlotMode==PLOT_MASK)
        {
         for(int k=0;k<kActive;k++)
           {
            if((InpCycleMask & (1<<k))!=0)
               sum+=g_bufs[k].cycle[i];
           }
        }

      g_outMain[i]=sum;
      g_outRaw[i]=sum;

      // forecast
      double fsum=0.0;
      int h=MaxInt(1,InpForecastH);
      for(int k=0;k<kActive;k++)
        {
         bool include=false;
         if(InpPlotMode==PLOT_SUM_TOP_K)
            include=true;
         else if(InpPlotMode==PLOT_SUM_TOP2)
           {
            for(int r=0;r<MinInt(2,kActive);r++)
               if(rankIdx[r]==k) include=true;
           }
         else if(InpPlotMode==PLOT_SINGLE_INDEX)
           {
            int idxSel=InpCycleIndex-1;
            if(idxSel>=0 && idxSel<kActive && rankIdx[idxSel]==k) include=true;
           }
         else if(InpPlotMode==PLOT_MASK)
           {
            if((InpCycleMask & (1<<k))!=0) include=true;
           }

         if(!include) continue;

         double omega=g_bufs[k].omega[i];
         double amp=g_bufs[k].amp[i];
         double phase=g_bufs[k].phaseUnw[i];
         double decay=MathPow(InpForecastAmpDecay,h);
         fsum+=amp*decay*MathCos(phase+omega*h);
        }
      g_forecast[i]=fsum;
      g_forecastRaw[i]=fsum;


      // direction / flip
      if(i+1<rates_total && g_outMain[i]!=EMPTY_VALUE && g_outMain[i+1]!=EMPTY_VALUE)
        {
         double d=g_outMain[i]-g_outMain[i+1];
         double dir=(d>0.0)?1.0:((d<0.0)?-1.0:0.0);
         g_dir[i]=dir;
         double prevDir=g_dir[i+1];
         if(dir>0 && prevDir<0) g_flip[i]=1.0;
         else if(dir<0 && prevDir>0) g_flip[i]=-1.0;
         else g_flip[i]=0.0;

         if(dir>0) g_colorIdx[i]=0.0; else if(dir<0) g_colorIdx[i]=1.0; else g_colorIdx[i]=2.0;
        }
      else
        {
         g_dir[i]=0.0;
         g_flip[i]=0.0;
         g_colorIdx[i]=2.0;
        }

      if(InpDebugLog && i==end)
         DebugDump(time,close,rates_total,prev_calculated,fullRecalc,newBar,start,end);
     }

   // REPAINT smoothing for omega/amp and dependent outputs
   if(InpEnableSmooth && g_sgSmoothOk && InpSmoothMode==SMOOTH_REPAINT)
     {
      int m=(g_sgSmoothW-1)/2;
      int iStart=fullRecalc ? m : m;
      int iEnd=fullRecalc ? (rates_total-1-m) : m;
      if(iStart<0) iStart=0;
      if(iEnd>rates_total-1-m) iEnd=rates_total-1-m;
      if(iEnd>=iStart)
        for(int i=iStart;i<=iEnd;i++)
        {
         for(int k=0;k<kActive;k++)
           {
            double accO=0.0, accA=0.0;
            for(int n=0;n<g_sgSmoothW;n++)
              {
               int idx=i-m+n;
               if(idx<0 || idx>=rates_total) continue;
               accO+=g_sgSmoothCoeffSym[n]*g_bufs[k].omegaRaw[idx];
               accA+=g_sgSmoothCoeffSym[n]*g_bufs[k].ampRaw[idx];
              }
            g_bufs[k].omega[i]=accO;
            g_bufs[k].period[i]=(MathAbs(accO)>1e-6)?(2.0*PI/accO):0.0;
            g_bufs[k].amp[i]=accA;
            g_bufs[k].cycle[i]=accA*MathCos(g_bufs[k].phaseUnw[i]);
           }

         // recompute output and forecast for this bar
         double sum=0.0;
         int rankIdx[MAX_TRACKS];
         bool usedRank[MAX_TRACKS];
         ArrayInitialize(usedRank,false);
         for(int r=0;r<kActive;r++)
           {
            double bestScore=-1.0;
            int best=-1;
            for(int k=0;k<kActive;k++)
              {
               if(usedRank[k]) continue;
               double score=MathAbs(g_bufs[k].amp[i])*g_bufs[k].conf[i];
               if(score>bestScore)
                 { bestScore=score; best=k; }
              }
            rankIdx[r]=best;
            if(best>=0) usedRank[best]=true;
           }

         if(InpPlotMode==PLOT_SUM_TOP_K)
           {
            for(int r=0;r<kActive;r++)
              {
               int k=rankIdx[r];
               if(k>=0) sum+=g_bufs[k].cycle[i];
              }
           }
         else if(InpPlotMode==PLOT_SUM_TOP2)
           {
            for(int r=0;r<MinInt(2,kActive);r++)
              {
               int k=rankIdx[r];
               if(k>=0) sum+=g_bufs[k].cycle[i];
              }
           }
         else if(InpPlotMode==PLOT_SINGLE_INDEX)
           {
            int idxSel=InpCycleIndex-1;
            if(idxSel>=0 && idxSel<kActive)
              {
               int k=rankIdx[idxSel];
               if(k>=0) sum=g_bufs[k].cycle[i];
              }
           }
         else if(InpPlotMode==PLOT_MASK)
           {
            for(int k=0;k<kActive;k++)
              {
               if((InpCycleMask & (1<<k))!=0)
                  sum+=g_bufs[k].cycle[i];
              }
           }
         g_outMain[i]=sum;
         g_outRaw[i]=sum;

         double fsum=0.0;
         int h=MaxInt(1,InpForecastH);
         for(int k=0;k<kActive;k++)
           {
            bool include=false;
            if(InpPlotMode==PLOT_SUM_TOP_K)
               include=true;
            else if(InpPlotMode==PLOT_SUM_TOP2)
              {
               for(int r=0;r<MinInt(2,kActive);r++)
                  if(rankIdx[r]==k) include=true;
              }
            else if(InpPlotMode==PLOT_SINGLE_INDEX)
              {
               int idxSel=InpCycleIndex-1;
               if(idxSel>=0 && idxSel<kActive && rankIdx[idxSel]==k) include=true;
              }
            else if(InpPlotMode==PLOT_MASK)
              {
               if((InpCycleMask & (1<<k))!=0) include=true;
              }
            if(!include) continue;
            double omega=g_bufs[k].omega[i];
            double amp=g_bufs[k].amp[i];
            double phase=g_bufs[k].phaseUnw[i];
            double decay=MathPow(InpForecastAmpDecay,h);
            fsum+=amp*decay*MathCos(phase+omega*h);
           }
         g_forecast[i]=fsum;
         g_forecastRaw[i]=fsum;


         // update direction/flip for repaint window
         if(i+1<rates_total && g_outMain[i]!=EMPTY_VALUE && g_outMain[i+1]!=EMPTY_VALUE)
           {
            double d=g_outMain[i]-g_outMain[i+1];
            double dir=(d>0.0)?1.0:((d<0.0)?-1.0:0.0);
            g_dir[i]=dir;
            double prevDir=g_dir[i+1];
            if(dir>0 && prevDir<0) g_flip[i]=1.0;
            else if(dir<0 && prevDir>0) g_flip[i]=-1.0;
            else g_flip[i]=0.0;

            if(dir>0) g_colorIdx[i]=0.0; else if(dir<0) g_colorIdx[i]=1.0; else g_colorIdx[i]=2.0;
           }
        }
     }

   // optional output normalization (explicit)
   if(InpNormMode!=NORM_NONE)
     {
      int W=InpNormWindow;
      if(W<5) W=5;
      int maxIdx=MaxInt(start,end);
      int minIdx=MinInt(start,end);
      if(maxIdx>rates_total-1) maxIdx=rates_total-1;
      if(minIdx<0) minIdx=0;
      for(int i=maxIdx;i>=minIdx;i--)
        {
         double baseVal=g_outRaw[i];
         if(baseVal==EMPTY_VALUE)
           {
            g_outMain[i]=EMPTY_VALUE;
            g_forecast[i]=EMPTY_VALUE;
            continue;
           }
         double scale=NormScaleAt(i,W,g_outRaw,rates_total);
         if(scale<=1e-12)
           {
            g_outMain[i]=baseVal;
            g_forecast[i]=g_forecastRaw[i];
            continue;
           }
         double s=InpNormTarget/scale;
         g_outMain[i]=baseVal*s;
         g_forecast[i]=g_forecastRaw[i]*s;
        }
      RecomputeDirFlip(maxIdx,minIdx,rates_total);
     }

   g_lastBarTime=time[0];
   if(!InpCalcOnTick && rates_total>0)
     {
      g_outMain[0]=EMPTY_VALUE;
      g_outRaw[0]=EMPTY_VALUE;
      g_forecast[0]=EMPTY_VALUE;
      g_forecastRaw[0]=EMPTY_VALUE;
      g_dir[0]=0.0;
      g_flip[0]=0.0;
      g_colorIdx[0]=2.0;
      g_trend[0]=EMPTY_VALUE;
      g_resid[0]=EMPTY_VALUE;
      g_hilbertDelayBuf[0]=(double)g_hilbertDelay;
      g_sgDelayBuf[0]=(InpSmoothMode==SMOOTH_REPAINT)?(double)((g_sgSmoothW-1)/2):0.0;
      for(int k=0;k<MAX_TRACKS;k++)
        {
         g_bufs[k].cycle[0]=EMPTY_VALUE;
         g_bufs[k].omega[0]=EMPTY_VALUE;
         g_bufs[k].period[0]=EMPTY_VALUE;
         g_bufs[k].amp[0]=EMPTY_VALUE;
         g_bufs[k].phaseUnw[0]=EMPTY_VALUE;
         g_bufs[k].phaseW[0]=EMPTY_VALUE;
         g_bufs[k].snr[0]=EMPTY_VALUE;
         g_bufs[k].snrDb[0]=EMPTY_VALUE;
         g_bufs[k].conf[0]=EMPTY_VALUE;
        }
     }
   return rates_total;
  }
