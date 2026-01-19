//+------------------------------------------------------------------+
//| WaveHealthPanel.mqh                                              |
//| Overlay health panel using CCanvas                               |
//+------------------------------------------------------------------+
#ifndef WAVE_HEALTH_PANEL_MQH
#define WAVE_HEALTH_PANEL_MQH

#include <Canvas\Canvas.mqh>

// ---------------- inputs ----------------
input group "WHP PANEL"
input bool   WHP_Enable      = true;
input int    WHP_PanelX      = 10;   // left (px)
input int    WHP_PanelY      = 30;   // top (px)
input bool   WHP_Background  = false; // default: no panel background
input color  WHP_BackColor   = clrBlack;

input group "WHP STYLE"
input int    WHP_ClockSize      = 80;   // radius in px (master size)
input int    WHP_FontSize       = 26;   // label/value size
input int    WHP_RingThickness  = 4;    // ring thickness (px)
input int    WHP_HandThickness  = 3;    // pointer thickness (px)
input int    WHP_FrameStyle     = 2;    // 0=round, 1=square, 2=both
input bool   WHP_ShowLabels     = true;
input int    WHP_LabelOffset    = 6;    // pixels from ring
input bool   WHP_NumInside      = false; // numbers inside ring (false = outside)
input color  WHP_RingColor      = clrSilver;
input color  WHP_NumColor       = clrWhite;
input color  WHP_HandColor      = clrRed;
input color  WHP_CenterColor    = clrWhite;
input color  WHP_TextColor      = clrWhite;
input color  WHP_PosColor       = clrLimeGreen;
input color  WHP_NegColor       = clrRed;

input group "WHP LAYOUT"
input int    WHP_ClockPerRow    = 0;    // 0=vertical stack, >=2 grid (side-by-side)
input int    WHP_ClockGap       = 18;   // extra spacing between clocks (px)

input group "WHP SPARK"
input bool   WHP_ShowSparks     = true;
input int    WHP_SparkWidth     = 140;  // px
input int    WHP_SparkHeight    = 42;   // px
input int    WHP_SparkAlpha     = 60;   // fill alpha
input int    WHP_SparkLineAlpha = 220;  // line alpha
input bool   WHP_ShowSparkLabels = true;
input int    WHP_SparkLabelSize  = 12;

input group "WHP RANGES"
input bool   WHP_UseFixedRanges = true;
input double WHP_AmpMin         = 0.0;
input double WHP_AmpMax         = 2.0;
input double WHP_MagMin         = 0.0;
input double WHP_MagMax         = 5.0;
input double WHP_SnrMin         = 0.0;
input double WHP_SnrMax         = 10.0;
input double WHP_BwMin          = 0.0;   // percent
input double WHP_BwMax          = 100.0; // percent
input double WHP_CycMin         = 5.0;   // bars
input double WHP_CycMax         = 80.0;  // bars
input double WHP_H2Min          = 0.0;
input double WHP_H2Max          = 1.0;
input double WHP_EqMin          = 0.0;
input double WHP_EqMax          = 1.0;
input double WHP_InstabRef      = 0.35; // lower = more stable

input group "WHP DEBUG"
input bool   WHP_DebugListObjects = false;
input int    WHP_DebugEveryNCalls = 100;

// ---------------- internal state ----------------
static string whp_prefix = "WHP_";
static bool   whp_inited = false;
static bool   whp_has_prev = false;
static double whp_prev_phase = 0.0;
static int    whp_dbg_count = 0;

static double whp_hist_amp[];
static double whp_hist_mag[];
static double whp_hist_snr[];
static double whp_hist_bw[];
static double whp_hist_cycle[];
static double whp_hist_h2[];
static double whp_hist_eq[];

static CCanvas whp_canvas;
static string  whp_canvas_name = "";
static int     whp_canvas_x = 0;
static int     whp_canvas_y = 0;
static int     whp_canvas_w = 0;
static int     whp_canvas_h = 0;
static int     whp_subwin = 0;
static bool    whp_visible = true;

#define WHP_HIST_LEN 32
#define WHP_PI 3.14159265358979323846

// ---------------- helpers ----------------
uint WHP_RGBA(const color c, const int a)
{
   int aa = a; if(aa < 0) aa = 0; if(aa > 255) aa = 255;
   uint rgb = COLOR2RGB(c);
   return TRGB(aa, rgb);
}

uint WHP_UColor(const color c)
{
   return COLOR2RGB(c);
}

int WHP_ClocksCount()
{
   return 8; // phase + 7 metrics
}

void WHP_SparkDims(int &spark_w, int &spark_h)
{
   int radius = WHP_ClockSize;
   if(!WHP_ShowSparks)
   {
      spark_w = 0;
      spark_h = 0;
      return;
   }
   spark_w = (WHP_SparkWidth > 0 ? WHP_SparkWidth : (2*radius));
   spark_h = (WHP_SparkHeight > 0 ? WHP_SparkHeight : radius);
   if(spark_h > 2*radius) spark_h = 2*radius;
}

void WHP_ComputeCanvasSize(int &w, int &h)
{
   int radius = WHP_ClockSize;
   int margin = MathMax(12, (WHP_FontSize/2) + 8);
   int gap = MathMax(0, WHP_ClockGap);
   int clocks = WHP_ClocksCount();
   int spark_w, spark_h;
   WHP_SparkDims(spark_w, spark_h);

   int per_row = WHP_ClockPerRow;
   if(per_row <= 1) per_row = 1;
   if(per_row > clocks) per_row = clocks;

   if(per_row == 1)
     {
      w = (2*radius) + margin + spark_w + margin;
      int step = (2*radius) + margin + gap;
      h = clocks * step + margin;
     }
   else
     {
      int rows = (clocks + per_row - 1) / per_row;
      int cell_w = MathMax(2*radius, spark_w);
      int cell_h = (2*radius) + margin + spark_h;
      w = per_row * cell_w + (per_row + 1) * margin + (per_row - 1) * gap;
      h = rows * cell_h + (rows + 1) * margin + (rows - 1) * gap;
     }
}

void WHP_LayoutPos(const int idx, int &cx, int &cy, int &sx, int &sy)
{
   int radius = WHP_ClockSize;
   int margin = MathMax(12, (WHP_FontSize/2) + 8);
   int gap = MathMax(0, WHP_ClockGap);
   int spark_w, spark_h;
   WHP_SparkDims(spark_w, spark_h);

   int clocks = WHP_ClocksCount();
   int per_row = WHP_ClockPerRow;
   if(per_row <= 1) per_row = 1;
   if(per_row > clocks) per_row = clocks;

   if(per_row == 1)
     {
      int step = (2*radius) + margin + gap;
      cx = WHP_PanelX + radius + margin;
      cy = WHP_PanelY + radius + margin + idx * step;
      sx = cx + radius + margin;
      sy = cy - (spark_h/2);
      return;
     }

   int row = idx / per_row;
   int col = idx % per_row;
   int cell_w = MathMax(2*radius, spark_w);
   int cell_h = (2*radius) + margin + spark_h;
   int left = WHP_PanelX + margin + col*(cell_w + gap);
   int top = WHP_PanelY + margin + row*(cell_h + gap);
   cx = left + cell_w/2;
   cy = top + radius;
   sx = left + (cell_w - spark_w)/2;
   sy = cy + radius + (margin/2);
}

void WHP_PushHist(double &arr[], double v)
{
   int n = ArraySize(arr);
   if(n != WHP_HIST_LEN) { ArrayResize(arr, WHP_HIST_LEN); ArrayInitialize(arr, 0.0); n = WHP_HIST_LEN; }
   for(int i=n-1; i>0; --i) arr[i] = arr[i-1];
   arr[0] = v;
}

void WHP_HistMinMax(const double &arr[], double &mn, double &mx)
{
   int n = ArraySize(arr);
   if(n <= 0) { mn=0.0; mx=0.0; return; }
   mn = arr[0]; mx = arr[0];
   for(int i=1;i<n;i++) { if(arr[i]<mn) mn=arr[i]; if(arr[i]>mx) mx=arr[i]; }
}

double WHP_Clamp(const double v, const double mn, const double mx)
{
   if(v < mn) return mn;
   if(v > mx) return mx;
   return v;
}

double WHP_WrapPhase(const double p)
{
   double x = p;
   while(x < 0.0) x += 2.0*WHP_PI;
   while(x >= 2.0*WHP_PI) x -= 2.0*WHP_PI;
   return x;
}

double WHP_UnwrapPhase(const double p, const double prev)
{
   double x = p;
   double dp = x - prev;
   while(dp >  WHP_PI) { x -= 2.0*WHP_PI; dp = x - prev; }
   while(dp < -WHP_PI) { x += 2.0*WHP_PI; dp = x - prev; }
   return x;
}

string WHP_FormatValue(const string label, const double value)
{
   if(label == "PHA") return StringFormat("%.1f°", value);
   if(label == "AMP") return StringFormat("%.2f", value);
   if(label == "MAG") return StringFormat("%.2f", value);
   if(label == "SNR") return StringFormat("%.2f", value);
   if(label == "BW%") return StringFormat("%.1f", value);
   if(label == "CYC") return StringFormat("%.1f", value);
   if(label == "H2")  return StringFormat("%.2f", value);
   if(label == "EQ")  return StringFormat("%.2f", value);
   return StringFormat("%.2f", value);
}

void WHP_DrawClockNumbers(const int cx, const int cy, const int r)
{
   int tx = WHP_FontSize / 2;
   int ty = WHP_FontSize / 2;
   int num_r = (WHP_NumInside ? (r - (WHP_FontSize + 6)) : (r + 12));
   for(int i=0;i<12;i++)
   {
      int num = (i==0 ? 12 : i);
      double a = -WHP_PI/2.0 + (2.0*WHP_PI)*(double)i/12.0;
      int x = cx + (int)MathRound(num_r*MathCos(a)) - tx;
      int y = cy + (int)MathRound(num_r*MathSin(a)) - ty;
      whp_canvas.TextOut(x, y, IntegerToString(num), WHP_UColor(WHP_NumColor));
   }
}

// Remove any leftover WHP_* objects (legacy object-based UI).
void WHP_ClearLegacyObjects()
{
   int total = ObjectsTotal(0, -1, -1);
   for(int i = total - 1; i >= 0; --i)
   {
      string name = ObjectName(0, i);
      if(StringFind(name, whp_prefix) == 0)
         ObjectDelete(0, name);
   }
}

void WHP_SetSubWindow(const int subwin)
{
   if(whp_subwin == subwin) return;
   whp_subwin = subwin;
   if(ObjectFind(0, whp_canvas_name) >= 0)
      ObjectDelete(0, whp_canvas_name);
   whp_canvas.Destroy();
   whp_canvas_name = "";
}

void WHP_SetVisible(const bool on)
{
   if(whp_visible == on) return;
   whp_visible = on;
   if(!on)
   {
      if(ObjectFind(0, whp_canvas_name) >= 0)
         ObjectDelete(0, whp_canvas_name);
      whp_canvas.Destroy();
   }
   else
   {
      whp_canvas_name = "";
   }
}

bool WHP_CreateCanvas()
{
   if(whp_canvas_name == "")
      whp_canvas_name = whp_prefix + "CANVAS";

   int w, h;
   WHP_ComputeCanvasSize(w, h);

   int chart_w = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS, whp_subwin);
   int chart_h = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS, whp_subwin);
   if(chart_w > 0 && w > chart_w-10) w = chart_w-10;
   if(chart_h > 0 && h > chart_h-10) h = chart_h-10;

   int x = WHP_PanelX;
   int y = WHP_PanelY;

   whp_canvas_x = x;
   whp_canvas_y = y;
   whp_canvas_w = w;
   whp_canvas_h = h;

   // create or attach bitmap label
   if(ObjectFind(0, whp_canvas_name) < 0)
   {
      if(!whp_canvas.CreateBitmapLabel(0, whp_subwin, whp_canvas_name, x, y, w, h, COLOR_FORMAT_ARGB_NORMALIZE))
         return false;
   }
   else
   {
      if(StringLen(whp_canvas.ResourceName()) == 0)
      {
         if(!whp_canvas.Attach(0, whp_canvas_name, COLOR_FORMAT_ARGB_NORMALIZE))
            return false;
      }
      ObjectSetInteger(0, whp_canvas_name, OBJPROP_XDISTANCE, x);
      ObjectSetInteger(0, whp_canvas_name, OBJPROP_YDISTANCE, y);
   }

   return true;
}

void WHP_BeginDraw()
{
   if(!WHP_Background)
      whp_canvas.Erase(0x00000000);
   else
      whp_canvas.Erase(WHP_RGBA(WHP_BackColor, 200));
   // no border/frame (keep transparent overlay)
   whp_canvas.FontSet("Arial", WHP_FontSize, FW_NORMAL);
}

void WHP_Text(const int x, const int y, const string text, const color col)
{
   whp_canvas.TextOut(x, y, text, WHP_UColor(col));
}

int WHP_TextWidth(const string text)
{
   return (int)MathRound((double)StringLen(text) * (double)WHP_FontSize * 0.6);
}

void WHP_TextCenter(const int cx, const int y, const string text, const color col)
{
   int w = WHP_TextWidth(text);
   WHP_Text(cx - (w/2), y, text, col);
}

void WHP_DrawLineThick(const int x1, const int y1, const int x2, const int y2, const int t, const color col)
{
   if(t <= 1) { whp_canvas.Line(x1, y1, x2, y2, WHP_UColor(col)); return; }
   double dx = (double)(x2 - x1);
   double dy = (double)(y2 - y1);
   double len = MathSqrt(dx*dx + dy*dy);
   if(len <= 0.0) { whp_canvas.PixelSet(x1, y1, WHP_UColor(col)); return; }
   double ux = -dy / len;
   double uy =  dx / len;
   int half = t/2;
   for(int i=-half; i<=half; i++)
   {
      int ox = (int)MathRound(ux * i);
      int oy = (int)MathRound(uy * i);
      whp_canvas.Line(x1+ox, y1+oy, x2+ox, y2+oy, WHP_UColor(col));
   }
}

void WHP_DrawFrame(const int cx, const int cy, const int r, const color col)
{
   if(WHP_FrameStyle == 0 || WHP_FrameStyle == 2)
      whp_canvas.Circle(cx, cy, r, WHP_UColor(col));
   if(WHP_FrameStyle == 1 || WHP_FrameStyle == 2)
      whp_canvas.Rectangle(cx - r, cy - r, cx + r, cy + r, WHP_UColor(col));
}

void WHP_DrawClockAt(const int cx, const int cy, const int r, const double phase,
                     const string label, const string value)
{
   int thick = WHP_RingThickness; if(thick < 1) thick = 1;

   for(int i=0;i<thick;i++)
      WHP_DrawFrame(cx, cy, r - i, WHP_RingColor);

   // numbers
   WHP_DrawClockNumbers(cx, cy, r);

   // hand (0 deg at top)
   double ang = -WHP_PI/2.0 + phase;
   if(ang < 0.0) ang += 2.0*WHP_PI;
   int hx = cx + (int)MathRound((r-2)*MathCos(ang));
   int hy = cy + (int)MathRound((r-2)*MathSin(ang));
   WHP_DrawLineThick(cx, cy, hx, hy, WHP_HandThickness, WHP_HandColor);

   // center
   whp_canvas.PixelSet(cx, cy, WHP_UColor(WHP_CenterColor));

   // centered label/value
   if(WHP_ShowLabels)
   {
      WHP_TextCenter(cx, cy - (WHP_FontSize/2) - 2, label, WHP_TextColor);
      WHP_TextCenter(cx, cy + 2, value, WHP_NumColor);
   }
}

void WHP_DrawGaugeAt(const int cx, const int cy, const int r,
                     const string label, const double value, const double ideal, const double &hist[],
                     const double range_min, const double range_max, const bool higher_is_better)
{
   int thick = WHP_RingThickness; if(thick < 1) thick = 1;
   for(int i=0;i<thick;i++)
      WHP_DrawFrame(cx, cy, r - i, WHP_RingColor);
   WHP_DrawClockNumbers(cx, cy, r);

   // pointer based on deviation
   double mn, mx;
   if(WHP_UseFixedRanges)
   {
      mn = range_min;
      mx = range_max;
   }
   else
   {
      WHP_HistMinMax(hist, mn, mx);
      if(mx - mn < 1e-6)
      {
         double pad = (ideal != 0.0 ? MathAbs(ideal) * 0.25 : 1.0);
         mn = ideal - pad;
         mx = ideal + pad;
      }
   }
   double t = (mx - mn > 0.0 ? (value - mn) / (mx - mn) : 0.5);
   if(t < 0.0) t = 0.0; if(t > 1.0) t = 1.0;
   double ang = -WHP_PI/2.0 + t * (2.0*WHP_PI); // full circle, 12 o'clock at t=0

   int hx = cx + (int)MathRound((r-2)*MathCos(ang));
   int hy = cy + (int)MathRound((r-2)*MathSin(ang));
   color pcol = (higher_is_better ? (value >= ideal ? WHP_PosColor : WHP_NegColor)
                                  : (value <= ideal ? WHP_PosColor : WHP_NegColor));
   WHP_DrawLineThick(cx, cy, hx, hy, WHP_HandThickness, pcol);
   whp_canvas.PixelSet(cx, cy, WHP_UColor(WHP_RingColor));

   // centered label/value
   if(WHP_ShowLabels)
   {
      string vtxt = WHP_FormatValue(label, value);
      WHP_TextCenter(cx, cy - (WHP_FontSize/2) - 2, label, WHP_TextColor);
      WHP_TextCenter(cx, cy + 2, vtxt, WHP_NumColor);
   }

   // optional sparkline
}

void WHP_DrawSparkline(const int x, const int y, const int w, const int h,
                       const double &hist[], const double range_min, const double range_max,
                       const string label, const color line_col)
{
   if(!WHP_ShowSparks) return;
   int n = ArraySize(hist);
   if(n < 2 || w < 4 || h < 4) return;

   double mn, mx;
   if(WHP_UseFixedRanges && range_max > range_min)
   {
      mn = range_min;
      mx = range_max;
   }
   else
   {
      WHP_HistMinMax(hist, mn, mx);
      if(mx - mn < 1e-9) { mx = mn + 1.0; }
   }

   int base_y = y + h - 1;
   double ema = hist[n-1];
   const double alpha = 0.35;

   if(WHP_ShowSparkLabels && label != "")
   {
      int fs = (WHP_SparkLabelSize > 0 ? WHP_SparkLabelSize : MathMax(10, WHP_FontSize/2));
      whp_canvas.FontSet("Arial", fs, FW_NORMAL);
      WHP_Text(x, y - fs - 2, label, line_col);
      whp_canvas.FontSet("Arial", WHP_FontSize, FW_NORMAL);
   }

   int px_prev = x;
   int py_prev = base_y;
   for(int i=0;i<n;i++)
   {
      int idx = n - 1 - i; // oldest -> newest
      double v = hist[idx];
      if(i == 0) ema = v;
      else ema = ema + alpha * (v - ema);

      double t = (ema - mn) / (mx - mn);
      t = WHP_Clamp(t, 0.0, 1.0);

      int px = x + (int)MathRound((double)i * (double)(w - 1) / (double)(n - 1));
      int py = base_y - (int)MathRound(t * (double)(h - 1));

      whp_canvas.Line(px, py, px, base_y, WHP_RGBA(line_col, WHP_SparkAlpha));
      if(i > 0)
         whp_canvas.Line(px_prev, py_prev, px, py, WHP_RGBA(line_col, WHP_SparkLineAlpha));

      px_prev = px;
      py_prev = py;
   }

   whp_canvas.Line(x, base_y, x + w - 1, base_y, WHP_RGBA(WHP_RingColor, 80));
}

// ---------------- public API ----------------
void WHP_Init(const string indicator_name)
{
   whp_prefix = "WHP_" + indicator_name + "_";
   whp_canvas_name = "";
   whp_inited = true;
   whp_has_prev = false;
   whp_visible = true;
   WHP_ClearLegacyObjects();
}

void WHP_Deinit()
{
   if(ObjectFind(0, whp_canvas_name) >= 0)
      ObjectDelete(0, whp_canvas_name);
   whp_canvas.Destroy();
   whp_inited = false;
   whp_has_prev = false;
}

void WHP_Update(const double phase, const double amp, const double mag_ratio, const double snr,
                const double bw_pct, const double cycle_bars, const double h2_ratio,
                const double instab)
{
   if(!whp_inited) return;
   if(!WHP_Enable) return;
   if(!whp_visible) return;

   if(!WHP_CreateCanvas()) return;

   // update stats
   bool phase_ok = (MathIsValidNumber(phase) && phase != EMPTY_VALUE);
   double p = (phase_ok ? phase : 0.0);
   if(phase_ok && whp_has_prev)
      p = WHP_UnwrapPhase(p, whp_prev_phase);
   double pwrap = WHP_WrapPhase(p);

   double amp_raw = (MathIsValidNumber(amp) ? amp : 0.0);
   double mag_raw = (MathIsValidNumber(mag_ratio) ? mag_ratio : 0.0);
   double snr_raw = (MathIsValidNumber(snr) ? snr : 0.0);
   double bw_raw  = (MathIsValidNumber(bw_pct) ? bw_pct : 0.0);
   double cyc_raw = (MathIsValidNumber(cycle_bars) ? cycle_bars : 0.0);
   double h2_raw  = (MathIsValidNumber(h2_ratio) ? h2_ratio : 0.0);
   double instab_val = (MathIsValidNumber(instab) ? MathMax(instab, 0.0) : 0.0);

   double amp_val  = WHP_Clamp(amp_raw, WHP_AmpMin, WHP_AmpMax);
   double mag_val  = WHP_Clamp(mag_raw, WHP_MagMin, WHP_MagMax);
   double snr_val  = WHP_Clamp(snr_raw, WHP_SnrMin, WHP_SnrMax);
   double bw_val   = WHP_Clamp(bw_raw, WHP_BwMin, WHP_BwMax);
   double cyc_val  = WHP_Clamp(cyc_raw, WHP_CycMin, WHP_CycMax);
   double h2_val   = WHP_Clamp(h2_raw, WHP_H2Min, WHP_H2Max);

   double instab_ref = (WHP_InstabRef > 1e-6 ? WHP_InstabRef : 0.35);
   double instab_norm = WHP_Clamp(instab_val / instab_ref, 0.0, 1.0);
   double amp_factor = WHP_Clamp(amp_raw, 0.0, 1.0);
   double eq_raw = (1.0 - instab_norm) * amp_factor;
   double eq_val = WHP_Clamp(eq_raw, WHP_EqMin, WHP_EqMax);

   // history
   WHP_PushHist(whp_hist_amp, amp_raw);
   WHP_PushHist(whp_hist_mag, mag_raw);
   WHP_PushHist(whp_hist_snr, snr_raw);
   WHP_PushHist(whp_hist_bw, bw_raw);
   WHP_PushHist(whp_hist_cycle, cyc_raw);
   WHP_PushHist(whp_hist_h2, h2_raw);
   WHP_PushHist(whp_hist_eq, eq_raw);

   // draw
   WHP_BeginDraw();

   // layout: vertical stack at top-left
   int spark_w, spark_h;
   WHP_SparkDims(spark_w, spark_h);
   int clocks = WHP_ClocksCount();
   color col_amp = clrLimeGreen;
   color col_mag = clrDeepSkyBlue;
   color col_snr = clrGold;
   color col_bw  = clrOrange;
   color col_cyc = clrDodgerBlue;
   color col_h2  = clrMediumSeaGreen;
   color col_eq  = clrAqua;

   for(int i=0;i<clocks;i++)
   {
      int cx, cy, sx, sy;
      WHP_LayoutPos(i, cx, cy, sx, sy);
      if(i == 0)
        {
         string pha_txt = WHP_FormatValue("PHA", pwrap * 180.0 / WHP_PI);
         WHP_DrawClockAt(cx, cy, WHP_ClockSize, pwrap, "PHA", pha_txt);
        }
      else if(i == 1)
        {
         WHP_DrawGaugeAt(cx, cy, WHP_ClockSize, "AMP", amp_val, 1.0, whp_hist_amp, WHP_AmpMin, WHP_AmpMax, true);
         WHP_DrawSparkline(sx, sy, spark_w, spark_h, whp_hist_amp, WHP_AmpMin, WHP_AmpMax, "AMP", col_amp);
        }
      else if(i == 2)
        {
         WHP_DrawGaugeAt(cx, cy, WHP_ClockSize, "MAG", mag_val, 1.5, whp_hist_mag, WHP_MagMin, WHP_MagMax, true);
         WHP_DrawSparkline(sx, sy, spark_w, spark_h, whp_hist_mag, WHP_MagMin, WHP_MagMax, "MAG", col_mag);
        }
      else if(i == 3)
        {
         WHP_DrawGaugeAt(cx, cy, WHP_ClockSize, "SNR", snr_val, 2.0, whp_hist_snr, WHP_SnrMin, WHP_SnrMax, true);
         WHP_DrawSparkline(sx, sy, spark_w, spark_h, whp_hist_snr, WHP_SnrMin, WHP_SnrMax, "SNR", col_snr);
        }
      else if(i == 4)
        {
         double bw_ideal = WHP_BwMin + (WHP_BwMax - WHP_BwMin) * 0.35;
         WHP_DrawGaugeAt(cx, cy, WHP_ClockSize, "BW%", bw_val, bw_ideal, whp_hist_bw, WHP_BwMin, WHP_BwMax, false);
         WHP_DrawSparkline(sx, sy, spark_w, spark_h, whp_hist_bw, WHP_BwMin, WHP_BwMax, "BW%", col_bw);
        }
      else if(i == 5)
        {
         double cyc_ideal = (WHP_CycMin + WHP_CycMax) * 0.5;
         WHP_DrawGaugeAt(cx, cy, WHP_ClockSize, "CYC", cyc_val, cyc_ideal, whp_hist_cycle, WHP_CycMin, WHP_CycMax, true);
         WHP_DrawSparkline(sx, sy, spark_w, spark_h, whp_hist_cycle, WHP_CycMin, WHP_CycMax, "CYC", col_cyc);
        }
      else if(i == 6)
        {
         WHP_DrawGaugeAt(cx, cy, WHP_ClockSize, "H2", h2_val, 0.2, whp_hist_h2, WHP_H2Min, WHP_H2Max, false);
         WHP_DrawSparkline(sx, sy, spark_w, spark_h, whp_hist_h2, WHP_H2Min, WHP_H2Max, "H2", col_h2);
        }
      else if(i == 7)
        {
         WHP_DrawGaugeAt(cx, cy, WHP_ClockSize, "EQ", eq_val, 0.7, whp_hist_eq, WHP_EqMin, WHP_EqMax, true);
         WHP_DrawSparkline(sx, sy, spark_w, spark_h, whp_hist_eq, WHP_EqMin, WHP_EqMax, "EQ", col_eq);
        }
   }

   whp_canvas.Update();

   if(phase_ok)
     {
      whp_prev_phase = p;
      whp_has_prev = true;
     }

   // debug dump (throttled)
   if(WHP_DebugListObjects)
   {
      whp_dbg_count++;
      int every = WHP_DebugEveryNCalls; if(every < 1) every = 1;
      if(whp_dbg_count >= every)
      {
         whp_dbg_count = 0;
         int total = ObjectsTotal(0, -1, -1);
         Print("WHP Debug: total=", total, " prefix=", whp_prefix);
         for(int i=total-1;i>=0;--i)
         {
            string name = ObjectName(0, i);
            if(StringFind(name, whp_prefix) != 0) continue;
            int x = (int)ObjectGetInteger(0, name, OBJPROP_XDISTANCE);
            int y = (int)ObjectGetInteger(0, name, OBJPROP_YDISTANCE);
            Print("WHP_OBJ name=", name, " x=", x, " y=", y);
         }
      }
   }
}

#endif // WAVE_HEALTH_PANEL_MQH
