/* * * * *
 *  AzpTimer_CNN.hpp
 *  Copyright (C) 2014,2015,2017 Rie Johnson
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

#ifndef _AZP_TIMER_CNN_HPP_
#define _AZP_TIMER_CNN_HPP_

#include <time.h>
#include "AzUtil.hpp"
#include "AzPmat.hpp"
#include "AzPrint.hpp"

namespace AzpTimer_CNN_type {
  enum t_type { /* 25 */
    t_Upward=0, t_Downward=1, 
    t_Flush=2, t_Test=3, t_DataX=4, t_DataY=5, 
    t_Merge=6, t_ThreadCreate=7, t_ThreadJoin=8, 
  }; 
  enum l_type {
    l_Upward=0, l_Downward=1, 
    l_UpdateDelta=2, l_Flush=3, l_Apply=4
  }; 
  enum ld_type {
    ld_UPatch=5, ld_UPool=6, ld_ULmod=7, 
    ld_DPatch=8, ld_DPool=9, ld_DLmod=10, 
    ld_UDropout=11, ld_DDropout=12, ld_UActiv=13, ld_DActiv=14, 
    ld_UResnorm=15, ld_DResnorm=16, 
  }; 
}

/*------------------------------------------------------------*/
class AzpTimer_CNN {
protected: 
  bool doing_thread; 
  #define AzpTimer_CNN_ClockMax 10
  double curr[AzpTimer_CNN_ClockMax];
  void reset() {
    for (int ix = 0; ix < AzpTimer_CNN_ClockMax; ++ix) curr[ix] = 0; 
  }  
  void reset_clock() {
    double clk = AzPdevice::sync_clock(); 
    for (int ix = 0; ix < AzpTimer_CNN_ClockMax; ++ix) curr[ix] = clk; 
  }
  void stamp(int clock_no=0, double *period=NULL) {
    AzX::throw_if((clock_no < 0 || clock_no >= AzpTimer_CNN_ClockMax), "AzpTimer_CNN::stamp", "invalid clock#"); 
    double prev = curr[clock_no]; curr[clock_no] = AzPdevice::sync_clock(); 
    if (period != NULL) *period += (curr[clock_no]-prev); 
  }  

  /*---  ---*/
  int layer_num; 
  #define _maxtyp_ 25
  double total[_maxtyp_]; 
  #define _maxlay_ 100
  double lay[_maxtyp_][_maxlay_]; 
  
public:
  AzpTimer_CNN() : layer_num(0), doing_thread(false) { reset(); }
  
  void reset(int hid_num) { 
    reset_clock(); 
    AzX::throw_if((hid_num+1 > _maxlay_-1),  /* +1 for the top layer; -1 for sum */
                  "AzpTimer_CNet::reset", "Too many layers for the timer"); 
    layer_num = hid_num+1;    
    set_zero(); 
  }
  void reset_doing_thread(bool flag) { doing_thread = flag; }
  
  void reset_Total() { stamp(0); }
  void reset_Layer() { stamp(1); }
  void reset_LayerDetail() { stamp(2); }
  void reset_Thread() { stamp(3); }
  void stamp_Total(AzpTimer_CNN_type::t_type typ) {
    stamp(0, &total[typ]); 
  }
  void stamp_Layer(AzpTimer_CNN_type::l_type typ, int lx) {
    stamp(1, &lay[typ][lx]); 
  }
  void stamp_LayerDetail(AzpTimer_CNN_type::ld_type typ, int lx) {
    stamp(2, &lay[typ][lx]); 
  }
  void stamp_Thread(AzpTimer_CNN_type::t_type typ) {
    stamp(3, &total[typ]);   
  }
    
  void show(const AzOut &out, const char *msg="") {
    using namespace AzpTimer_CNN_type; 
    AzPrint::writeln(out, msg); 
    AzBytArr s; 
    double trn = 0; 
    trn += format(s, "dataX", total[t_DataX]); 
    trn += format(s, "dataY", total[t_DataY]);     
    trn += format(s, "up", total[t_Upward]); 
    trn += format(s, "down", total[t_Downward]);
    trn += format(s, "flush", total[t_Flush]);
    if (doing_thread) trn += format(s, "merge", total[t_Merge]);    
    format(s, "test", total[t_Test]); /* test */
    format(s, "data-up-dn-fl-total", trn); 
    if (doing_thread) {
      format(s, "threadC", total[t_ThreadCreate]); 
      format(s, "threadJ", total[t_ThreadJoin]);       
    }
    s.nl(); 

    if (!doing_thread) {
      int sx = layer_num; 
      for (int ix = 0; ix < _maxtyp_; ++ix) {
        lay[ix][sx] = 0; 
        for (int lx = 0; lx < layer_num; ++lx) lay[ix][sx] += lay[ix][lx]; 
      }

      for (int lx = 0; lx <= layer_num; ++lx) {
        if (lx == layer_num) s.c("all-layers: "); 
        else                 s << "layer#" << lx << ": "; 
        double trn = 0; 
        trn += format(s, "upward", lay[l_Upward][lx]); 
        trn += format(s, "downward", lay[l_Downward][lx]); 
        trn += format(s, "updateDelta", lay[l_UpdateDelta][lx]); 
        trn += format(s, "flush", lay[l_Flush][lx]); 
        format(s, "apply", lay[l_Apply][lx]); 
        format(s, "tr_total", trn); 
        format(s, "total", trn + lay[l_Apply][lx]); 
        s.nl();       
      }

      s.c("---  layer detail   ---"); s.nl(); 
      for (int lx = 0; lx <= layer_num; ++lx) {
        if (lx == layer_num) s.c("all-layers: "); 
        else                 s << "layer#" << lx << ": "; 
        double up = 0, dn=0; 
        up += format(s, "up_dropout", lay[ld_UDropout][lx]);      
        up += format(s, "up patch", lay[ld_UPatch][lx]); 
        up += format(s, "up_lmod", lay[ld_ULmod][lx]); 
        up += format(s, "up_act", lay[ld_UActiv][lx]);  
        up += format(s, "up_pooling", lay[ld_UPool][lx]);       
        up += format(s, "up_resnorm", lay[ld_UResnorm][lx]);         
        dn += format(s, "down_dropout", lay[ld_DDropout][lx]);            
        dn += format(s, "down_patch", lay[ld_DPatch][lx]); 
        dn += format(s, "down_lmod", lay[ld_DLmod][lx]);       
        dn += format(s, "down_act", lay[ld_DActiv][lx]);       
        dn += format(s, "down_pooling", lay[ld_DPool][lx]);       
        dn += format(s, "down_resnorm", lay[ld_DResnorm][lx]);        
        format(s, "up_total", up); 
        format(s, "down_total", dn); 
        format(s, "lmod_total(up,down,updateDelta)", lay[ld_ULmod][lx]+lay[ld_DLmod][lx]+lay[l_UpdateDelta][lx]); 
        format(s, "up_down_total", up+dn); 
        s.nl();       
      } 
    } /* !doing_thread */
    
    AzPrint::writeln(out, s);
    AzPrint::writeln(out, ""); 
  }
  
  void set_zero() {
    for (int ix1 = 0; ix1 < _maxtyp_; ++ix1) {
      total[ix1] = 0; 
      for (int ix2 = 0; ix2 < _maxlay_; ++ix2) lay[ix1][ix2] = 0; 
    }
  }  
  
protected:   
  double format(AzBytArr &s, const char *nm, double ticks) const {
    s << nm << "=" << (double)ticks/(double)CLOCKS_PER_SEC << " "; 
    return ticks; 
  }    
}; 
#endif 