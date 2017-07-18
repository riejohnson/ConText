/* * * * *
 *  AzpMain_reNet.hpp
 *  Copyright (C) 2017 Rie Johnson
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * * * * */

#ifndef _AZP_MAIN_RENET_HPP_
#define _AZP_MAIN_RENET_HPP_

#include <time.h>
#include "AzUtil.hpp"
#include "AzPrint.hpp"
#include "AzpDataSetDflt.hpp"
#include "AzParam.hpp"
#include "AzpCompoSetDflt.hpp"

#include "AzpReNet.hpp"

/*------------------------------------------------------------*/ 
class AzpMain_reNet_Param_ {
public: 
  bool doLog, doDump; 
  
  AzpMain_reNet_Param_() : doLog(true), doDump(false) {}
  virtual void resetParam(const AzOut &out, AzParam &azp) = 0; 
  virtual void printHelp(AzHelp &h) const {
    h.writeln("No parameter!"); 
  }
  virtual void reset(AzParam &azp, const AzOut &out, const AzBytArr &s_action) {
    if (azp.needs_help()) {
      printHelp(out); 
      AzX::throw_if(true, AzNormal, "", "");       
    }      
    print_hline(out);     
    AzPrint::writeln(out, s_action, " ", azp.c_str()); 
    print_hline(out);  
    resetParam(out, azp); 
  }
  /*------------------------------------------------------------*/   
  void print_hline(const AzOut &out) const {
    if (out.isNull()) return; 
    AzPrint::writeln(out, "--------------------");  
  }  
protected:   
  void printHelp(const AzOut &out) const { AzHelp h(out); printHelp(h); h.end(); }
  void _resetParam(AzPrint &o, AzParam &azp); 
  static void _printHelp(AzHelp &h); 
  void setupLogDmp(AzPrint &o, AzParam &azp); 
}; 

/*------------------------------------------------------------*/ 
class AzpMain_reNet {
protected:
  AzpCompoSetDflt cs; 
  virtual AzpReNet *alloc_renet_for_test(AzObjPtrArr<AzpReNet> &opa, AzParam &azp) {  
    return alloc_renet(opa, azp, true); 
  }
  virtual AzpReNet *alloc_renet(AzObjPtrArr<AzpReNet> &opa, AzParam &azp, bool for_test=false) {
    opa.alloc(1); 
    cs.reset(azp, for_test, log_out); 
    opa.set(0, new AzpReNet(&cs)); 
    return opa(0); 
  }
  
public:
  AzpMain_reNet() {} 

  void renet(int argc, const char *argv[], const AzBytArr &s_action); 
  void predict(int argc, const char *argv[], const AzBytArr &s_action); 
  void write_word_mapping(int argc, const char *argv[], const AzBytArr &s_action); 
  void write_embedded(int argc, const char *argv[], const AzBytArr &s_action); 
  
protected:
  void _write_embedded(AzpReNet *net, AzParam &azp, const AzpData_ *tst, int mb, int feat_top_num, const char *fn); 
  AzOut *reset_eval(const AzBytArr &s_eval_fn, AzOfs &ofs, AzOut &eval_out);                                  
}; 
#endif
