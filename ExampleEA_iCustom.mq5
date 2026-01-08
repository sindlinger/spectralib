//+------------------------------------------------------------------+
//| ExampleEA_iCustom.mq5                                            |
//| Minimal EA reading DominantCycles_STFT_Kalman_Hilbert_SG          |
//+------------------------------------------------------------------+
#property strict

#include <Trade/Trade.mqh>

input string InpIndicatorName="DominantCycles_STFT_Kalman_Hilbert_SG";
input int    InpKCycles=5;        // must match indicator K
input double InpSNRGate=2.0;      // average SNR gate
input double InpLots=0.10;
input long   InpMagic=20250101;

CTrade g_trade;
int g_handle=-1;
datetime g_lastBarTime=0;

int OnInit()
  {
   g_handle=iCustom(_Symbol,_Period,InpIndicatorName);
   if(g_handle==INVALID_HANDLE)
     {
      Print("Failed to create indicator handle: ",GetLastError());
      return INIT_FAILED;
     }
   g_trade.SetExpertMagicNumber(InpMagic);
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   if(g_handle!=INVALID_HANDLE)
      IndicatorRelease(g_handle);
  }

bool ReadBuffer(int bufferIndex,int shift,double &value)
  {
   double tmp[];
   if(CopyBuffer(g_handle,bufferIndex,shift,1,tmp)!=1)
      return false;
   value=tmp[0];
   return true;
  }

void ClosePositionsByType(const string symbol,ENUM_POSITION_TYPE type)
  {
   for(int i=PositionsTotal()-1;i>=0;i--)
     {
      if(!PositionGetTicket(i)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=symbol) continue;
      if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE)!=type) continue;
      ulong ticket=PositionGetInteger(POSITION_TICKET);
      g_trade.PositionClose(ticket);
     }
  }

void OnTick()
  {
   datetime t=iTime(_Symbol,_Period,0);
   if(t==g_lastBarTime) return; // only on new bar
   g_lastBarTime=t;

   double outMain=0.0, dir=0.0, flip=0.0;
   if(!ReadBuffer(0,1,outMain)) return; // OutputMain
   if(!ReadBuffer(3,1,dir)) return;     // Dir
   if(!ReadBuffer(4,1,flip)) return;    // Flip

   // average SNR of first K cycles
   int kActive=MathMax(3,MathMin(7,InpKCycles));
   double snrSum=0.0;
   int snrCount=0;
   for(int k=0;k<kActive;k++)
     {
      int base=5+k*9;
      double snr=0.0;
      if(ReadBuffer(base+6,1,snr))
        {
         snrSum+=snr;
         snrCount++;
        }
     }
   double snrAvg=(snrCount>0)?(snrSum/snrCount):0.0;
   if(snrAvg<InpSNRGate) return;

   // example: read period of cycle #1
   double period1=0.0;
   ReadBuffer(5+2,1,period1);
   if(period1<=0.0) return;

   // trade on flips
   if(flip>0.0 && dir>0.0)
     {
      ClosePositionsByType(_Symbol,POSITION_TYPE_SELL);
      if(!PositionSelect(_Symbol))
         g_trade.Buy(InpLots,_Symbol);
     }
   else if(flip<0.0 && dir<0.0)
     {
      ClosePositionsByType(_Symbol,POSITION_TYPE_BUY);
      if(!PositionSelect(_Symbol))
         g_trade.Sell(InpLots,_Symbol);
     }
  }
