#define _AZ_MAIN_
#include "AzUtil.hpp"
#include "AzpMain_reNet.hpp"

extern AzPdevice dev; 

/*-----------------------------------------------------------------*/
void help()
{
#ifdef __AZ_GPU__  
  cout << "Arguments:  gpu#[:mem]  action  parameters" <<endl; 
#else
  cout << "Arguments:  _  action  parameters" <<endl; 
#endif 
  cout << "   action: train | predict | write_word_mapping | write_embedded"<<endl; 
}

/*******************************************************************/
/*     main                                                        */
/*******************************************************************/
int main(int argc, const char *argv[]) 
{
  AzException *stat = NULL; 
  
  if (argc < 3) {
    help(); 
    return -1; 
  }
  const char *gpu_param = argv[1]; 
  int ix; 
  for (ix = 2; ix < argc; ++ix) {
    argv[ix-1] = argv[ix]; 
  }
  --argc; 

  int gpu_dev = dev.setDevice(gpu_param); 
  if (gpu_dev < 0) {
    AzPrint::writeln(log_out, "Using CPU ... "); 
  }
  else {
    AzPrint::writeln(log_out, "Using GPU#", gpu_dev); 
  }  

  const char *action = argv[1]; 
  AzBytArr s_action(action); 
  int ret = 0; 
  try {
    Az_check_system2_(); 

    AzpMain_reNet driver; 
    if      (s_action.equals("train"))              driver.renet(argc-2, argv+2, s_action); 
    else if (s_action.equals("predict"))            driver.predict(argc-2, argv+2, s_action);     
    else if (s_action.equals("write_word_mapping")) driver.write_word_mapping(argc-2, argv+2, s_action);      
    else if (s_action.equals("write_embedded"))     driver.write_embedded(argc-2, argv+2, s_action);      
    else {
      help(); 
      ret = -1; 
      goto labend; 
    }
  } 
  catch (AzException *e) {
    stat = e; 
  }
  if (stat != NULL) {
    cout << stat->getMessage() << endl; 
    ret = -1; 
    goto labend;  
  }

labend: 
  dev.closeDevice();   
  return ret; 
}
