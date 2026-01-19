//+------------------------------------------------------------------+
//| SpectralServiceBridge.mqh                                       |
//| Bridge para SpectralLibService (named pipe)                     |
//+------------------------------------------------------------------+
#ifndef __SPECTRAL_SERVICE_BRIDGE_MQH__
#define __SPECTRAL_SERVICE_BRIDGE_MQH__

#define SPECTRAL_PIPE_NAME "\\\\.\\pipe\\SpectralLibServiceAsync"
#define SPECTRAL_CMD_SWT    1
#define SPECTRAL_CMD_STFT   2
#define SPECTRAL_CMD_CWT    3
#define SPECTRAL_CMD_INIT   10
#define SPECTRAL_CMD_PUSH   11
#define SPECTRAL_CMD_GET    12
#define SPECTRAL_CMD_INIT_BEGIN 20
#define SPECTRAL_CMD_INIT_CHUNK 21
#define SPECTRAL_CMD_INIT_END   22
#define SPECTRAL_TAG        "SPS1"

#define GENERIC_READ 0x80000000
#define GENERIC_WRITE 0x40000000
#define OPEN_EXISTING 3
#define FILE_ATTRIBUTE_NORMAL 0x00000080
#define INVALID_HANDLE_VALUE -1

#import "kernel32.dll"
int WaitNamedPipeW(string name, uint timeout);
int CreateFileW(string name, uint access, uint share, int security, uint creation, uint flags, int templateFile);
int ReadFile(int handle, uchar &buffer[], uint bytesToRead, uint &bytesRead, int overlapped);
int WriteFile(int handle, const uchar &buffer[], uint bytesToWrite, uint &bytesWritten, int overlapped);
int CloseHandle(int handle);
#import

inline bool SpectralPipeSend(const string &payload, string &response, const int timeout_ms=5)
{
   if(!WaitNamedPipeW(SPECTRAL_PIPE_NAME, timeout_ms)) return false;
   int h = CreateFileW(SPECTRAL_PIPE_NAME, GENERIC_READ|GENERIC_WRITE, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
   if(h == INVALID_HANDLE_VALUE) return false;

   uchar outbuf[];
   int outlen = StringToCharArray(payload, outbuf, 0, StringLen(payload), CP_UTF8);
   uint written = 0;
   if(outlen <= 0 || WriteFile(h, outbuf, (uint)outlen, written, 0) == 0)
     { CloseHandle(h); return false; }

   uchar inbuf[]; ArrayResize(inbuf, 262144);
   uint read = 0;
   if(ReadFile(h, inbuf, (uint)ArraySize(inbuf), read, 0) == 0 || read == 0)
     { CloseHandle(h); return false; }
   CloseHandle(h);

   response = CharArrayToString(inbuf, 0, (int)read, CP_UTF8);
   return (StringLen(response) > 0);
}

inline bool SpectralPipeSendNoWait(const string &payload, const int timeout_ms=0)
{
   if(!WaitNamedPipeW(SPECTRAL_PIPE_NAME, timeout_ms)) return false;
   int h = CreateFileW(SPECTRAL_PIPE_NAME, GENERIC_READ|GENERIC_WRITE, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
   if(h == INVALID_HANDLE_VALUE) return false;

   uchar outbuf[];
   int outlen = StringToCharArray(payload, outbuf, 0, StringLen(payload), CP_UTF8);
   uint written = 0;
   if(outlen <= 0 || WriteFile(h, outbuf, (uint)outlen, written, 0) == 0)
     { CloseHandle(h); return false; }
   CloseHandle(h);
   return true;
}

inline bool SpectralPipeCallSwtArray(const double &data[], const int level, double &out0, double &out1, const int timeout_ms=5)
{
   int len = ArraySize(data);
   if(len < 8) return false;
   static long s_seq = 0;
   s_seq++;

   string req = SPECTRAL_TAG + "|" + IntegerToString(SPECTRAL_CMD_SWT) + "|" + (string)s_seq + "|" +
                (string)len + "|" + (string)level + "|0|0|";
   for(int i=0;i<len;i++)
   {
      if(i>0) req += ",";
      req += DoubleToString(data[i], 10);
   }

   string resp="";
   if(!SpectralPipeSend(req, resp, timeout_ms)) return false;

   string parts[];
   int n = StringSplit(resp, '|', parts);
   if(n < 5) return false;
   if(parts[0] != SPECTRAL_TAG) return false;
   if((int)StringToInteger(parts[1]) != SPECTRAL_CMD_SWT) return false;

   out0 = StringToDouble(parts[3]);
   out1 = StringToDouble(parts[4]);
   return true;
}

inline bool SpectralPipeCallSwtSeries(const double &series[], const int len, const int level, double &out0, double &out1, const int timeout_ms=5)
{
   if(len < 8) return false;
   static long s_seq = 0;
   s_seq++;

   string req = SPECTRAL_TAG + "|" + IntegerToString(SPECTRAL_CMD_SWT) + "|" + (string)s_seq + "|" +
                (string)len + "|" + (string)level + "|0|0|";
   for(int i=len-1; i>=0; --i)
   {
      if(i!=len-1) req += ",";
      req += DoubleToString(series[i], 10);
   }

   string resp="";
   if(!SpectralPipeSend(req, resp, timeout_ms)) return false;

   string parts[];
   int n = StringSplit(resp, '|', parts);
   if(n < 5) return false;
   if(parts[0] != SPECTRAL_TAG) return false;
   if((int)StringToInteger(parts[1]) != SPECTRAL_CMD_SWT) return false;

   out0 = StringToDouble(parts[3]);
   out1 = StringToDouble(parts[4]);
   return true;
}

inline bool SpectralPipeCallStftLast(const double &series[], const int len, const int nperseg, const int noverlap, const int nfft,
                                     double &peakFreq, double &peakMag, const int timeout_ms=5)
{
   if(len < 8) return false;
   static long s_seq = 0;
   s_seq++;

   string req = SPECTRAL_TAG + "|" + IntegerToString(SPECTRAL_CMD_STFT) + "|" + (string)s_seq + "|" +
                (string)len + "|" + (string)nperseg + "|" + (string)noverlap + "|" + (string)nfft + "|";
   for(int i=len-1; i>=0; --i)
   {
      if(i!=len-1) req += ",";
      req += DoubleToString(series[i], 10);
   }

   string resp="";
   if(!SpectralPipeSend(req, resp, timeout_ms)) return false;

   string parts[];
   int n = StringSplit(resp, '|', parts);
   if(n < 5) return false;
   if(parts[0] != SPECTRAL_TAG) return false;
   if((int)StringToInteger(parts[1]) != SPECTRAL_CMD_STFT) return false;

   peakFreq = StringToDouble(parts[3]);
   peakMag = StringToDouble(parts[4]);
   return true;
}

inline bool SpectralPipeCallCwtLast(const double &series[], const int len, const double scale, const int wavelet_code, const int precision,
                                    double &outRe, double &outIm, const int timeout_ms=5)
{
   if(len < 8) return false;
   static long s_seq = 0;
   s_seq++;

   string req = SPECTRAL_TAG + "|" + IntegerToString(SPECTRAL_CMD_CWT) + "|" + (string)s_seq + "|" +
                (string)len + "|" + DoubleToString(scale,6) + "|" + (string)wavelet_code + "|" + (string)precision + "|";
   for(int i=len-1; i>=0; --i)
   {
      if(i!=len-1) req += ",";
      req += DoubleToString(series[i], 10);
   }

   string resp="";
   if(!SpectralPipeSend(req, resp, timeout_ms)) return false;

   string parts[];
   int n = StringSplit(resp, '|', parts);
   if(n < 5) return false;
   if(parts[0] != SPECTRAL_TAG) return false;
   if((int)StringToInteger(parts[1]) != SPECTRAL_CMD_CWT) return false;

   outRe = StringToDouble(parts[3]);
   outIm = StringToDouble(parts[4]);
   return true;
}

// --- Chunked init (async-friendly) ---
inline bool SpectralPipeInitBegin(const int total_len, const int level, const int timeout_ms=2)
{
   static long s_seq = 0;
   s_seq++;
   if(total_len < 8) return false;
   string req = SPECTRAL_TAG + "|" + IntegerToString(SPECTRAL_CMD_INIT_BEGIN) + "|" + (string)s_seq + "|" +
                "0|" + (string)level + "|" + (string)total_len + "|0|";
   string resp="";
   return SpectralPipeSend(req, resp, timeout_ms);
}

inline bool SpectralPipeInitChunk(const double &chunk[], const int count, const int timeout_ms=2)
{
   if(count <= 0) return false;
   static long s_seq = 0;
   s_seq++;
   string req = SPECTRAL_TAG + "|" + IntegerToString(SPECTRAL_CMD_INIT_CHUNK) + "|" + (string)s_seq + "|" +
                (string)count + "|0|0|0|";
   for(int i=0;i<count;i++)
   {
      if(i>0) req += ",";
      req += DoubleToString(chunk[i], 10);
   }
   string resp="";
   return SpectralPipeSend(req, resp, timeout_ms);
}

inline bool SpectralPipeInitEnd(double &out0, double &out1, const int timeout_ms=5)
{
   static long s_seq = 0;
   s_seq++;
   string req = SPECTRAL_TAG + "|" + IntegerToString(SPECTRAL_CMD_INIT_END) + "|" + (string)s_seq + "|0|0|0|0|";
   string resp="";
   if(!SpectralPipeSend(req, resp, timeout_ms)) return false;
   string parts[];
   int n = StringSplit(resp, '|', parts);
   if(n < 5) return false;
   if(parts[0] != SPECTRAL_TAG) return false;
   if((int)StringToInteger(parts[1]) != SPECTRAL_CMD_INIT_END) return false;
   out0 = StringToDouble(parts[3]);
   out1 = StringToDouble(parts[4]);
   return true;
}

inline bool SpectralPipeInitSwtStream(const double &series[], const int len, const int level, double &out0, double &out1, const int timeout_ms=5)
{
   if(len < 8) return false;
   static long s_seq = 0;
   s_seq++;

   string req = SPECTRAL_TAG + "|" + IntegerToString(SPECTRAL_CMD_INIT) + "|" + (string)s_seq + "|" +
                (string)len + "|" + (string)level + "|0|0|";
   for(int i=len-1; i>=0; --i)
   {
      if(i!=len-1) req += ",";
      req += DoubleToString(series[i], 10);
   }

   string resp="";
   if(!SpectralPipeSend(req, resp, timeout_ms)) return false;

   string parts[];
   int n = StringSplit(resp, '|', parts);
   if(n < 5) return false;
   if(parts[0] != SPECTRAL_TAG) return false;
   if((int)StringToInteger(parts[1]) != SPECTRAL_CMD_INIT) return false;

   out0 = StringToDouble(parts[3]);
   out1 = StringToDouble(parts[4]);
   return true;
}

inline bool SpectralPipePushSample(const double sample, const int level, const int is_new_bar, const int timeout_ms=0)
{
   static long s_seq = 0;
   s_seq++;
   string req = SPECTRAL_TAG + "|" + IntegerToString(SPECTRAL_CMD_PUSH) + "|" + (string)s_seq + "|" +
                "1|" + (string)level + "|" + (string)is_new_bar + "|0|" + DoubleToString(sample, 10);
   return SpectralPipeSendNoWait(req, timeout_ms);
}

inline bool SpectralPipeGetLast(double &out0, double &out1, const int timeout_ms=0)
{
   static long s_seq = 0;
   s_seq++;
   string req = SPECTRAL_TAG + "|" + IntegerToString(SPECTRAL_CMD_GET) + "|" + (string)s_seq + "|0|0|0|0|";

   string resp="";
   if(!SpectralPipeSend(req, resp, timeout_ms)) return false;

   string parts[];
   int n = StringSplit(resp, '|', parts);
   if(n < 5) return false;
   if(parts[0] != SPECTRAL_TAG) return false;
   if((int)StringToInteger(parts[1]) != SPECTRAL_CMD_GET) return false;

   out0 = StringToDouble(parts[3]);
   out1 = StringToDouble(parts[4]);
   return true;
}

#endif // __SPECTRAL_SERVICE_BRIDGE_MQH__
