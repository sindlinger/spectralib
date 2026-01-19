//+------------------------------------------------------------------+
//| WaveHealthMetrics.mqh                                            |
//| Metrics extractor using spectralib (GPU STFT/FFT + Hilbert)      |
//|                                                                  |
//| This module is standalone and does NOT depend on any indicator.  |
//| You pass a time-series and receive metrics for the panel.        |
//+------------------------------------------------------------------+
#ifndef WAVE_HEALTH_METRICS_MQH
#define WAVE_HEALTH_METRICS_MQH

#include <spectralib/SpectralCommon.mqh>
#include <spectralib/SpectralHilbert.mqh>
#include <spectralib/SpectralOpenCLFFT.mqh>
#include <spectralib/SpectralWindows.mqh>
#include <spectralib/SpectralImpl.mqh>
#include <spectralib/SpectralFIRDesign.mqh>
#include <spectralib/SpectralPyWTConvolution.mqh>
#include <spectralib/SpectralPyWTCommon.mqh>

// ---------------------------
// Metrics output (documented)
// ---------------------------
struct WHM_Metrics
{
   bool   valid;          // overall computation success

   // amp_inst (Hilbert): instantaneous strength of the cycle in time-domain.
   // Use: low amp => phase becomes noisy; filter or down-weight trades.
   double amp_inst;

   // phase_inst (Hilbert): instantaneous phase (radians).
   // Use: phase-based timing / clock.
   double phase_inst;

   // freq_dom: dominant frequency (Hz, or cycles/bar if fs=1).
   // Use: estimated cycle speed.
   double freq_dom;

   // period_bars: dominant period (bars). period_bars = fs / freq_dom.
   double period_bars;

   // mag_peak: peak magnitude in spectrum (dominant energy).
   // mag_med: median magnitude (noise floor proxy).
   // snr: mag_peak / mag_med (confidence of dominant cycle).
   double mag_peak;
   double mag_med;
   double snr;

   // bw_hz: bandwidth around peak at -3dB (half-power).
   // Use: narrower band => more stable cycle.
   double bw_hz;

   // instab: frequency instability (STFT) as std/mean of per-seg peaks.
   // Use: drive dynamic bandwidth (wider when unstable).
   double instab;

   // bw_dyn: dynamic bandwidth (Hz) used for FIR band-pass.
   double bw_dyn;

   // band_rms_in/out: RMS before/after band-pass.
   // band_gain: ratio out/in (gain normalization proxy).
   double band_rms_in;
   double band_rms_out;
   double band_gain;

   // h1/h2 harmonic amplitudes (from FFT) and ratio.
   double h1_amp;
   double h2_amp;
   double h2_ratio;

   // group_delay: estimated group delay (samples/bars) around peak.
   // Use: lag/latency estimate in time domain.
   double group_delay;
};

// ---------------------------
// Helpers
// ---------------------------
inline void WHM_Reset(WHM_Metrics &m)
{
   m.valid = false;
   m.amp_inst = 0.0;
   m.phase_inst = 0.0;
   m.freq_dom = 0.0;
   m.period_bars = 0.0;
   m.mag_peak = 0.0;
   m.mag_med = 0.0;
   m.snr = 0.0;
   m.bw_hz = 0.0;
   m.instab = 0.0;
   m.bw_dyn = 0.0;
   m.band_rms_in = 0.0;
   m.band_rms_out = 0.0;
   m.band_gain = 0.0;
   m.h1_amp = 0.0;
   m.h2_amp = 0.0;
   m.h2_ratio = 0.0;
   m.group_delay = 0.0;
}

// Copy last nfft samples into seg[] as chronological order (oldest->newest).
// is_series = true when input array is series (0 = most recent).
inline bool WHM_ExtractTail(const double &x_in[], const int nfft, const bool is_series, double &seg[])
{
   int N = ArraySize(x_in);
   if(N < nfft || nfft <= 0) return false;
   ArrayResize(seg, nfft);
   ArraySetAsSeries(seg, false); // ensure chronological output
   if(is_series)
   {
      // Use the most recent nfft points (x_in[0..nfft-1]) and reverse to oldest->newest.
      for(int i=0;i<nfft;i++) seg[i] = x_in[nfft-1-i];
   }
   else
   {
      // Input is chronological, take tail.
      int start = N - nfft;
      for(int i=0;i<nfft;i++) seg[i] = x_in[start + i];
   }
   return true;
}

// Median of a copy (ArraySort is fine for small N).
inline double WHM_Median(const double &v_in[])
{
   int n = ArraySize(v_in);
   if(n <= 0) return 0.0;
   double v[]; ArrayCopy(v, v_in);
   ArraySort(v);
   if(n % 2 == 1) return v[n/2];
   return 0.5 * (v[n/2 - 1] + v[n/2]);
}

inline double WHM_RMS(const double &v_in[])
{
   int n = ArraySize(v_in);
   if(n <= 0) return 0.0;
   double acc = 0.0;
   for(int i=0;i<n;i++) acc += v_in[i]*v_in[i];
   return MathSqrt(acc / (double)n);
}

inline double WHM_UnwrapPhase(const double p, const double prev)
{
   double x = p;
   double dp = x - prev;
   while(dp >  PI) { x -= 2.0*PI; dp = x - prev; }
   while(dp < -PI) { x += 2.0*PI; dp = x - prev; }
   return x;
}

// Phase slope stats from analytic signal (last window).
// dphi_mean/std are in radians per sample.
inline void WHM_PhaseStats(const Complex64 &analytic[], const int n, const int win,
                           double &dphi_mean, double &dphi_std)
{
   dphi_mean = 0.0;
   dphi_std = 0.0;
   if(n < 3) return;
   int w = win;
   if(w < 3) w = 3;
   if(w > n - 1) w = n - 1;
   int start = n - w;

   double prev = MathArctan2(analytic[start-1].im, analytic[start-1].re);
   double acc = 0.0;
   double acc2 = 0.0;
   int count = 0;
   for(int i=start; i<n; i++)
   {
      double phi = MathArctan2(analytic[i].im, analytic[i].re);
      double un = WHM_UnwrapPhase(phi, prev);
      double dphi = un - prev;
      prev = un;
      acc += dphi;
      acc2 += dphi*dphi;
      count++;
   }
   if(count <= 0) return;
   dphi_mean = acc / (double)count;
   double var = acc2 / (double)count - dphi_mean*dphi_mean;
   if(var < 0.0) var = 0.0;
   dphi_std = MathSqrt(var);
}

// Estimate instability from STFT peak frequency variance (complex STFT).
// STFT instability using true 2D complex matrix (no flatten).
inline double WHM_Instability_STFT(const double &x[], const double fs, const string win_name,
                                   const int nperseg, const int noverlap, const int nfft)
{
   double freqs[]; double t[]; matrixc Z;
   if(!stft_1d_matrixc(x,fs,win_name,nperseg,noverlap,nfft,0,true,"spectrum",freqs,t,Z))
      return 0.0;
   int nseg=(int)Z.Rows();
   int nfreq=(int)Z.Cols();
   if(nseg<=1 || nfreq<=1) return 0.0;

   double peaks[]; ArrayResize(peaks, nseg);
   for(int s=0;s<nseg;s++)
   {
      int kmax=1; double vmax=0.0;
      for(int k=1;k<nfreq;k++)
      {
         complex z = Z[s][k];
         double mag=MathSqrt(z.real*z.real + z.imag*z.imag);
         if(mag>vmax) { vmax=mag; kmax=k; }
      }
      peaks[s] = (kmax < ArraySize(freqs) ? freqs[kmax] : 0.0);
   }
   double mean_f=0.0;
   for(int i=0;i<nseg;i++) mean_f += peaks[i];
   mean_f /= (double)nseg;
   if(mean_f<=1e-12) return 0.0;
   double var=0.0;
   for(int i=0;i<nseg;i++) { double d=peaks[i]-mean_f; var += d*d; }
   var /= (double)nseg;
   return MathSqrt(var)/mean_f;
}

// Estimate local phase slope around peak for group delay (samples).
inline double WHM_GroupDelay(const Complex64 &spec[], const int k, const double fs, const int nfft)
{
   if(k <= 0 || k >= nfft-1) return 0.0;
   double p1 = MathArctan2(spec[k-1].im, spec[k-1].re);
   double p2 = MathArctan2(spec[k+1].im, spec[k+1].re);
   double dphi = p2 - p1;
   while(dphi >  PI) dphi -= 2.0*PI;
   while(dphi < -PI) dphi += 2.0*PI;
   double df = (2.0 * fs) / (double)nfft;   // freq step between k-1 and k+1
   if(df <= 0.0) return 0.0;
   double domega = 2.0 * PI * df;
   if(MathAbs(domega) < 1e-12) return 0.0;
   return -dphi / domega;
}

// ---------------------------
// Main compute
// ---------------------------
// fs = sampling frequency (1.0 for bars).
// win_name = "hann", "hamming", "blackman", "kaiser", etc.
inline bool WHM_Compute(const double &x_in[], const int nfft, const bool is_series,
                        const double fs, const string win_name, WHM_Metrics &out)
{
   WHM_Reset(out);
   if(nfft <= 8 || fs <= 0.0) return false;

   double seg[];
   if(!WHM_ExtractTail(x_in, nfft, is_series, seg)) return false;

   // Remove DC to stabilize phase estimation.
   double mean = 0.0;
   for(int i=0;i<nfft;i++) mean += seg[i];
   mean /= (double)nfft;
   for(int i=0;i<nfft;i++) seg[i] -= mean;

   out.band_rms_in = WHM_RMS(seg);

   // -------- Raw Hilbert (time-domain amplitude/phase)
   // amp_inst is instantaneous amplitude from analytic signal (time domain).
   Complex64 analytic_raw[];
   if(!hilbert_analytic_gpu(seg, analytic_raw)) return false;

   int last = nfft - 2; // avoid edge artifacts on last sample
   if(last < 0) last = nfft - 1;
   if(last >= ArraySize(analytic_raw)) last = ArraySize(analytic_raw) - 1;
   double are = analytic_raw[last].re;
   double aim = analytic_raw[last].im;
   out.amp_inst = MathSqrt(are*are + aim*aim);
   out.phase_inst = MathArctan2(aim, are);
   out.band_rms_out = out.band_rms_in;
   if(out.band_rms_in > 1e-12) out.band_gain = out.band_rms_out / out.band_rms_in;

   // Phase-based cycle estimate (fallback if FFT fails).
   double dphi_mean = 0.0;
   double dphi_std = 0.0;
   WHM_PhaseStats(analytic_raw, ArraySize(analytic_raw), 24, dphi_mean, dphi_std);
   double dphi_abs = MathAbs(dphi_mean);
   double freq_fallback = (dphi_abs > 1e-6 ? (dphi_abs / (2.0*PI)) * fs : 0.0);
   double period_fallback = (dphi_abs > 1e-6 ? (2.0*PI) / dphi_abs : 0.0);
   double instab_phi = (dphi_abs > 1e-9 ? dphi_std / dphi_abs : dphi_std);

   // -------- FFT (dominant frequency & spectral metrics)
   double win[];
   get_window(win_name, nfft, true, win);
   Complex64 inC[];
   ArrayResize(inC, nfft);
   for(int i=0;i<nfft;i++) inC[i] = Cx(seg[i] * win[i], 0.0);

   static CLFFTPlan plan;
   if(!plan.ready) CLFFTReset(plan);
   bool fft_ok = CLFFTInit(plan, nfft);

   Complex64 spec[];
   if(fft_ok && !CLFFTExecute(plan, inC, spec, false)) fft_ok = false;

   int nfreq = (nfft/2) + 1;
   if(nfreq <= 1) fft_ok = false;

   double mags[];
   if(fft_ok) ArrayResize(mags, nfreq);
   int kmax = 1;
   int kmin = 1;
   int kmax_lim = nfreq - 1;
   // Limit search to reasonable cycle range (avoid 1–2 bar noise peaks).
   double max_period = (double)nfft / 2.0;
   double min_period = 6.0;
   if(max_period < min_period) min_period = max_period;
   if(max_period > 0.0)
     {
      kmin = (int)MathFloor((double)nfft / max_period);
      if(kmin < 1) kmin = 1;
     }
   if(min_period > 0.0)
     {
      kmax_lim = (int)MathCeil((double)nfft / min_period);
      if(kmax_lim > nfreq - 1) kmax_lim = nfreq - 1;
     }
   if(kmax_lim < kmin) { kmin = 1; kmax_lim = nfreq - 1; }
   double maxv = 0.0;
   for(int k=1; fft_ok && k<nfreq; k++)
   {
      double mag = MathSqrt(spec[k].re*spec[k].re + spec[k].im*spec[k].im);
      mags[k] = mag;
      if(k>=kmin && k<=kmax_lim && mag > maxv) { maxv = mag; kmax = k; }
   }

   out.mag_peak = maxv;
   out.mag_med = (fft_ok ? WHM_Median(mags) : 0.0);
   if(fft_ok && out.mag_med <= 1e-12)
     {
      double sum = 0.0;
      int cnt = 0;
      for(int k=1;k<nfreq;k++) { sum += mags[k]; cnt++; }
      if(cnt > 0) out.mag_med = sum / (double)cnt;
     }
   if(!fft_ok && out.band_rms_in > 1e-12)
     {
      // Fallback proxy: use time-domain RMS as a magnitude estimate.
      out.mag_peak = out.band_rms_in * 1.41421356237;
      out.mag_med = out.band_rms_in;
     }
   if(out.mag_med > 1e-12) out.snr = out.mag_peak / out.mag_med;

   if(fft_ok)
     {
      out.freq_dom = (double)kmax * (fs / (double)nfft);
      out.period_bars = (out.freq_dom > 1e-12 ? (fs / out.freq_dom) : 0.0);
     }
   else
     {
      out.freq_dom = freq_fallback;
      out.period_bars = period_fallback;
     }

   // bandwidth at -3dB (0.707)
   double thr = maxv * 0.70710678;
   int kL = kmax;
   int kR = kmax;
   while(fft_ok && kL > 1 && mags[kL] >= thr) kL--;
   while(fft_ok && kR < nfreq-1 && mags[kR] >= thr) kR++;
   int bwBins = (kR - kL);
   out.bw_hz = (fft_ok ? (double)bwBins * (fs / (double)nfft) : 0.0);

   // group delay estimate around peak (samples/bars)
   out.group_delay = (fft_ok ? WHM_GroupDelay(spec, kmax, fs, nfft) : 0.0);

   // Harmonics (from FFT)
   out.h1_amp = (fft_ok ? mags[kmax] : 0.0);
   int k2 = kmax*2;
   if(fft_ok && k2 < nfreq) out.h2_amp = mags[k2];
   if(out.h1_amp > 1e-12) out.h2_ratio = out.h2_amp / out.h1_amp;

   // -------- STFT instability (path C)
   int nperseg = MathMin(128, nfft);
   if(nperseg < 16) nperseg = nfft;
   int noverlap = nperseg/2;
   int nfft_stft = nperseg;
   double instab_stft = WHM_Instability_STFT(seg, fs, win_name, nperseg, noverlap, nfft_stft);
   out.instab = (instab_stft > 0.0 ? instab_stft : instab_phi);
   if(out.band_rms_in <= 1e-9 || out.amp_inst <= 1e-9)
      out.instab = MathMax(out.instab, 1.0);

   // -------- FIR band-pass (path B)
   double f0 = out.freq_dom;
   if(f0 <= 0.0) f0 = 0.0;
   double bw_base = (out.bw_hz > 1e-12 ? out.bw_hz : f0*0.25);
   if(bw_base <= 0.0) bw_base = fs/(double)nfft;
   double k_instab = 1.5;
   out.bw_dyn = bw_base * (1.0 + k_instab * out.instab);

   double f1 = f0 - out.bw_dyn*0.5;
   double f2 = f0 + out.bw_dyn*0.5;
   double fnyq = 0.5*fs;
   if(f1 < fs/(double)nfft) f1 = fs/(double)nfft;
   if(f2 > fnyq*0.99) f2 = fnyq*0.99;

   double filt_sig[];
   bool have_filt = false;
   if(fft_ok && f0 > 0.0 && f2 > f1)
   {
      int numtaps = (int)MathRound(out.period_bars*2.0 + 1.0);
      if(numtaps < 31) numtaps = 31;
      if(numtaps > 129) numtaps = 129;
      if((numtaps % 2) == 0) numtaps++;

      double cutoff[2]; cutoff[0]=f1; cutoff[1]=f2;
      double win_params[]; ArrayResize(win_params,0);
      double h[];
      if(firwin(numtaps, cutoff, 0.0, "hann", win_params, "bandpass", true, fs, h))
      {
         double hmin[];
         if(!minimum_phase(h, "homomorphic", 0, hmin))
            ArrayCopy(hmin, h);

         double yfull[];
         if(PyWT_DownsamplingConvolution(seg, hmin, 1, PYWT_MODE_REFLECT, yfull))
         {
            int N = ArraySize(seg);
            ArrayResize(filt_sig, N);
            for(int i=0;i<N;i++) filt_sig[i] = (i < ArraySize(yfull) ? yfull[i] : 0.0);
            have_filt = true;
         }
      }
   }

   // -------- Hilbert (instantaneous phase/amplitude) on filtered signal
   if(have_filt)
     {
      Complex64 analytic_filt[];
      if(hilbert_analytic_gpu(filt_sig, analytic_filt))
        {
         int lastf = nfft - 2;
         if(lastf < 0) lastf = nfft - 1;
         if(lastf >= ArraySize(analytic_filt)) lastf = ArraySize(analytic_filt) - 1;
         double aref = analytic_filt[lastf].re;
         double aimf = analytic_filt[lastf].im;
         // amp_inst is time-domain amplitude (not spectral magnitude).
         out.amp_inst = MathSqrt(aref*aref + aimf*aimf);
         out.phase_inst = MathArctan2(aimf, aref);
         out.band_rms_out = WHM_RMS(filt_sig);
         if(out.band_rms_in > 1e-12) out.band_gain = out.band_rms_out / out.band_rms_in;
        }
     }

   out.valid = true;
   return true;
}

#endif // WAVE_HEALTH_METRICS_MQH
