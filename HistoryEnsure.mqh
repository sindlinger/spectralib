#ifndef __SPECTRAL_HISTORY_ENSURE_MQH__
#define __SPECTRAL_HISTORY_ENSURE_MQH__

// Ensure enough history bars are available for a symbol/timeframe.
// This triggers terminal history download via CopyRates if needed.
// Returns true when bars are available (or if bars_needed <= 0).
inline bool EnsureHistoryBars(const string symbol,const ENUM_TIMEFRAMES tf,
                              const int bars_needed,const int timeout_ms=15000,
                              const int poll_ms=50)
  {
   if(bars_needed<=0) return true;
   SymbolSelect(symbol,true);

   ulong start=GetTickCount64();
   MqlRates rates[];

   while((GetTickCount64()-start) < (ulong)timeout_ms)
     {
      int copied=CopyRates(symbol,tf,0,bars_needed,rates);
      long bars=Bars(symbol,tf);
      bool sync=(bool)SeriesInfoInteger(symbol,tf,SERIES_SYNCHRONIZED);

      if(copied>=bars_needed && bars>=bars_needed && sync)
         return true;

      Sleep(poll_ms);
     }
   return false;
  }

// Force download by time range (chunked). Useful when bars are available
// but not yet loaded into the terminal cache.
inline bool EnsureHistoryRange(const string symbol,const ENUM_TIMEFRAMES tf,
                               datetime from_time,datetime to_time,
                               const int chunk_bars=5000,
                               const int timeout_ms=30000,
                               const int poll_ms=50)
  {
   if(from_time<=0 || to_time<=0 || from_time>=to_time) return false;
   SymbolSelect(symbol,true);
   int period_sec=PeriodSeconds(tf);
   if(period_sec<=0) period_sec=60;
   int chunk=chunk_bars;
   if(chunk<100) chunk=100;

   ulong start=GetTickCount64();
   MqlRates rates[];
   datetime cur=from_time;
   while(cur<to_time && (GetTickCount64()-start) < (ulong)timeout_ms)
     {
      datetime end=cur + (datetime)((long)chunk*period_sec);
      if(end>to_time) end=to_time;
      int copied=CopyRates(symbol,tf,cur,end,rates);
      if(copied>0)
        {
         cur = rates[copied-1].time + period_sec;
        }
      else
        {
         Sleep(poll_ms);
         cur += period_sec;
        }
     }
   return true;
  }

// Force download by progressively requesting more bars (like DownloadHistory.mq5 script).
// Useful in live terminal when history is available but not yet cached.
inline bool EnsureHistoryBarsChunked(const string symbol,const ENUM_TIMEFRAMES tf,
                                     const int bars_needed,const int chunk_bars=5000,
                                     const int timeout_ms=120000,const int wait_ms=200,
                                     const bool verbose=true)
  {
   if(bars_needed<=0) return true;
   SymbolSelect(symbol,true);

   int chunk=chunk_bars;
   if(chunk<100) chunk=100;

   // In tester/optimization, history is controlled by the tester; avoid long loops.
   if(MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_OPTIMIZATION))
     {
      MqlRates rates[];
      int request=MathMin(bars_needed, chunk);
      CopyRates(symbol, tf, 0, request, rates);
      return (Bars(symbol, tf) >= bars_needed);
     }

   ulong start_ms=GetTickCount64();
   int bars=Bars(symbol,tf);
   if(verbose)
      PrintFormat("[HistoryEnsure] Start chunked: %s %s bars=%d target=%d",
                  symbol, EnumToString(tf), bars, bars_needed);

   MqlRates rates[];
   int attempts=0;
   int last_bars=bars;
   int last_copied=-1;
   int stall=0;
   int max_attempts=(int)MathMax(5.0, (double)timeout_ms/MathMax(1.0,(double)wait_ms)+2.0);
   int max_stall=(int)MathMax(10.0, (double)max_attempts/5.0);
   while(bars < bars_needed)
     {
      int request=MathMin(bars_needed, bars + chunk);
      ResetLastError();
      int copied=CopyRates(symbol, tf, 0, request, rates);
      int err=GetLastError();
      bool sync=(bool)SeriesInfoInteger(symbol, tf, SERIES_SYNCHRONIZED);
      bool log_now=(verbose && (attempts==0 || copied!=last_copied || bars!=last_bars || (attempts%50)==0));
      if(log_now)
         PrintFormat("[HistoryEnsure] chunk try=%d request=%d copied=%d bars=%d sync=%d err=%d",
                     attempts, request, copied, bars, (int)sync, err);

      Sleep(wait_ms);
      int bars_now=Bars(symbol,tf);
      if(bars_now==last_bars && copied==last_copied)
         stall++;
      else
         stall=0;
      last_bars=bars_now;
      last_copied=copied;
      bars=bars_now;
      attempts++;
      if(stall >= max_stall)
        {
         if(verbose)
            PrintFormat("[HistoryEnsure] chunk stalled (bars=%d copied=%d) -> stop", bars, last_copied);
         break;
        }
      if(attempts >= max_attempts)
         break;
      if((GetTickCount64()-start_ms) > (ulong)timeout_ms)
         break;
     }

   if(verbose)
      PrintFormat("[HistoryEnsure] Done chunked: bars=%d target=%d elapsed=%dms",
                  bars, bars_needed, (int)(GetTickCount64()-start_ms));
   if(bars < bars_needed && verbose)
      Print("[HistoryEnsure] NOTE: server/history limit reached or tester mode.");

   return (bars >= bars_needed);
  }

#endif
