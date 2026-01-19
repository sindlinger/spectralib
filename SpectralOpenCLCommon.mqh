#ifndef __SPECTRAL_OPENCL_COMMON_MQH__
#define __SPECTRAL_OPENCL_COMMON_MQH__

// Helper: create GPU context with float64 requirement, but avoid CL_USE_GPU_DOUBLE_ONLY bug.
inline bool _spectral_is_cpu_name(const string n)
  {
   string u=n; StringToUpper(u);
   if(StringFind(u,"CPU")>=0) return true;
   if(StringFind(u,"INTEL(R) CORE")>=0) return true;
   if(StringFind(u,"RYZEN")>=0) return true;
   if(StringFind(u,"THREADRIPPER")>=0) return true;
   return false;
  }

inline bool _spectral_is_gpu_name(const string n)
  {
   string u=n; StringToUpper(u);
   if(StringFind(u,"NVIDIA")>=0) return true;
   if(StringFind(u,"QUADRO")>=0) return true;
   if(StringFind(u,"RTX")>=0) return true;
   if(StringFind(u,"GTX")>=0) return true;
   if(StringFind(u,"RADEON")>=0) return true;
   if(StringFind(u,"VEGA")>=0) return true;
   if(StringFind(u,"GPU")>=0) return true;
   if(StringFind(u,"AMD RADEON")>=0) return true;
   return false;
  }
inline int CLCreateContextGPUFloat64(const string tag="")
  {
   static bool log_once=false;
   static bool probe_once=false;
   int support = (int)TerminalInfoInteger(TERMINAL_OPENCL_SUPPORT);
   int ctx=CLContextCreate(CL_USE_GPU_DOUBLE_ONLY);
   if(ctx!=INVALID_HANDLE) return ctx;
   int err1=GetLastError();

   int ctx2=CLContextCreate(CL_USE_GPU_ONLY);
   if(ctx2==INVALID_HANDLE)
     {
      int err2=GetLastError();
      if(!log_once)
        {
         PrintFormat("%s: CLContextCreate failed (double_only err=%d, gpu err=%d, opencl_support=%d)",
                     (tag=="" ? "SpectralOpenCL" : tag), err1, err2, support);
         log_once=true;
        }

      // Try CL_USE_ANY as last resort (but reject clear CPU devices)
      int ctx_any=CLContextCreate(CL_USE_ANY);
      if(ctx_any==INVALID_HANDLE)
        {
         if(!probe_once)
           {
            PrintFormat("%s: CL_USE_ANY also failed (err=%d, opencl_support=%d).",
                        (tag=="" ? "SpectralOpenCL" : tag), GetLastError(), support);
            probe_once=true;
           }
         return INVALID_HANDLE;
        }

      string dev_name="";
      bool name_ok = CLGetInfoString(ctx_any, CL_DEVICE_NAME, dev_name);
      if(name_ok && _spectral_is_cpu_name(dev_name) && !_spectral_is_gpu_name(dev_name))
        {
         if(!probe_once)
           {
            PrintFormat("%s: CL_USE_ANY picked CPU device '%s' - refusing.",
                        (tag=="" ? "SpectralOpenCL" : tag), dev_name);
            probe_once=true;
           }
         CLContextFree(ctx_any);
         return INVALID_HANDLE;
        }

      string exts_any="";
      bool exts_any_ok = CLGetInfoString(ctx_any,CL_DEVICE_EXTENSIONS,exts_any);
      if(exts_any_ok && StringFind(exts_any,"cl_khr_fp64")<0)
        {
         if(!probe_once)
           {
            PrintFormat("%s: CL_USE_ANY device lacks cl_khr_fp64 ('%s') - refusing.",
                        (tag=="" ? "SpectralOpenCL" : tag), (name_ok?dev_name:""));
            probe_once=true;
           }
         CLContextFree(ctx_any);
         return INVALID_HANDLE;
        }

      if(!probe_once)
        {
         PrintFormat("%s: CL_USE_ANY selected '%s' (opencl_support=%d).",
                     (tag=="" ? "SpectralOpenCL" : tag), (name_ok?dev_name:"?"), support);
         probe_once=true;
        }
      return ctx_any;
     }

   string exts="";
   bool exts_ok = CLGetInfoString(ctx2,CL_DEVICE_EXTENSIONS,exts);
   if(!exts_ok)
     {
      if(!log_once)
        {
         PrintFormat("%s: CLGetInfoString failed after CL_USE_GPU_ONLY (err=%d) - proceeding without fp64 check",
                     (tag=="" ? "SpectralOpenCL" : tag), GetLastError());
         log_once=true;
        }
     }
   if(exts_ok && StringFind(exts,"cl_khr_fp64")<0)
     {
      if(!log_once)
        {
         PrintFormat("%s: GPU context lacks cl_khr_fp64, refusing.",
                     (tag=="" ? "SpectralOpenCL" : tag));
         log_once=true;
        }
      CLContextFree(ctx2);
      return INVALID_HANDLE;
     }

   if(!log_once)
     {
      PrintFormat("%s: CL_USE_GPU_DOUBLE_ONLY failed (err=%d), using CL_USE_GPU_ONLY with fp64.",
                  (tag=="" ? "SpectralOpenCL" : tag), err1);
      log_once=true;
     }
   return ctx2;
  }

#endif
