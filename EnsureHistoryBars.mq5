#property script_show_inputs

#include <spectralib/v7_port/HistoryEnsure.mqh>

input string          InpSymbol     = "";              // vazio = simbolo atual
input ENUM_TIMEFRAMES InpTimeframe  = PERIOD_CURRENT;  // timeframe
input int             InpBarsNeeded = 10000;           // barras necessarias
input int             InpTimeoutMs  = 30000;           // timeout (ms)
input int             InpPollMs     = 50;              // polling (ms)

void OnStart()
  {
   string sym = (InpSymbol=="" ? _Symbol : InpSymbol);
   ENUM_TIMEFRAMES tf = (InpTimeframe==PERIOD_CURRENT ? (ENUM_TIMEFRAMES)_Period : InpTimeframe);

   bool ok = EnsureHistoryBars(sym, tf, InpBarsNeeded, InpTimeoutMs, InpPollMs);
   long bars = Bars(sym, tf);
   bool sync = (bool)SeriesInfoInteger(sym, tf, SERIES_SYNCHRONIZED);

   if(ok)
      PrintFormat("[EnsureHistoryBars] OK: %s %s bars=%d sync=%d", sym, EnumToString(tf), bars, (int)sync);
   else
      PrintFormat("[EnsureHistoryBars] FAIL: %s %s bars=%d sync=%d (timeout)", sym, EnumToString(tf), bars, (int)sync);
  }
