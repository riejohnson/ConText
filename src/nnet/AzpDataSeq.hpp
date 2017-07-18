/* * * * *
 *  AzpDataSeq.hpp
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

#ifndef _AZP_DATA_SEQ_HPP_
#define _AZP_DATA_SEQ_HPP_

#include "AzUtil.hpp"
#include "AzTools.hpp"
#include "AzDmat.hpp"

class AzpDataSeq {
protected:
  AzIntArr ia_seq;  
  bool dont_shuffle, do_adjust, do_force_fast_shuffle; 
  AzDataArr<AzIntArr> aia_cl_dxs; 
  
public: 
  AzpDataSeq() : dont_shuffle(false), do_adjust(false), do_force_fast_shuffle(false) {}

  /*------------------------------------------------------------*/   
  #define kw_dont_shuffle "PreserveDataOrder"  
  #define kw_do_adjust "AdjustDataOrder"
  #define kw_do_force_fast_shuffle "FastShuffle"
  void resetParam(AzParam &azp) {
    azp.swOn(&dont_shuffle, kw_dont_shuffle); 
    azp.swOn(&do_adjust, kw_do_adjust); 
    if (!dont_shuffle && !do_adjust) azp.swOn(&do_force_fast_shuffle, kw_do_force_fast_shuffle); 
  }
  void printParam(const AzOut &out) const {
    AzPrint o(out); 
    o.printSw(kw_dont_shuffle, dont_shuffle); 
    o.printSw(kw_do_adjust, do_adjust); 
    o.printSw(kw_do_force_fast_shuffle, do_force_fast_shuffle); 
  }  
  
  /*------------------------------------------------------------*/   
  template <class T>
  void init(const T *data, const AzOut &out) {
    aia_cl_dxs.reset(); 
    if (!do_adjust) return; 

    data->gen_membership(aia_cl_dxs); 
    AzBytArr s("AzpDataSeq::init: class distribution -- "); 
    for (int cx = 0; cx < aia_cl_dxs.size(); ++cx) s << aia_cl_dxs[cx]->size() << " "; 
    AzPrint::writeln(out, s); 
  }

  /*------------------------------------------------------------*/ 
  template <class T>
  const int *gen(const T *data, int data_size, bool dont_shuffle_override=false) {
    const char *eyec = "AzDataSeq::gen"; 
    ia_seq.reset(); 
    if (do_adjust && data != NULL) {
      if (data->batchNum() > 1) {
        data->gen_membership(aia_cl_dxs);  /* since it's likely to be a new batch ...  */
      }
      int class_num = aia_cl_dxs.size(); 
      AzX::throw_if((class_num <= 0), eyec, "call init first"); 
      if (!dont_shuffle && !dont_shuffle_override) {
        for (int cx = 0; cx < class_num; ++cx) AzTools::shuffle2(*aia_cl_dxs(cx)); 
      }
      ia_seq.prepare(data_size); 
      for (int ix = 0; ; ++ix) {
        for (int cx = 0; cx < class_num; ++cx) {        
          const AzIntArr *ia_dxs = aia_cl_dxs.point(cx); 
          int jx = ix % ia_dxs->size(); 
          ia_seq.put(ia_dxs->get(jx)); 
        }
        if (ia_seq.size() >= data_size) break; 
      }
    }
    else {
      ia_seq.range(0, data_size);       
      if (!dont_shuffle) {      
        /* "shuffle2" is faster but, keep "shuffle" for compatibility with the published results */
        if (do_force_fast_shuffle || data_size >= 26000000) AzTools::shuffle2(ia_seq); 
        else                                                AzTools::shuffle(-1, &ia_seq);                  
      }
    }
    return ia_seq.point(); 
  }
}; 

#endif 