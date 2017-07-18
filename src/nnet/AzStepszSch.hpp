/* * * * *
 *  AzStepszSch.hpp
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
 
#ifndef _AZ_STEPSZ_SCH_HPP_
#define _AZ_STEPSZ_SCH_HPP_

#include "AzUtil.hpp"
#include "AzHelp.hpp"
#include "AzTools.hpp"

/*---  change the learning rate after every iteration  ---*/
#define AzStepszSch_Type_Tinv 'T'
#define AzStepszSch_Type_Expo 'E'
#define AzStepszSch_Type_expo 'e'
#define AzStepszSch_Type_Linear 'L'
#define AzStepszSch_Type_Few 'F'
#define AzStepszSch_Type_None 'N'

class AzStepszSch {
protected:
  AzBytArr s_typ; 
  AzByte typ; 

  /*---  Few (decay at a few points)  ---*/
  AzBytArr s_decay_at; 
  AzIntArr ia_decay_at;
  double decay_factor; 
  int decay_index; 
  
  /*---  Tinv (t inverse): eta (1 + x t)^{-y}  ---*/
  /*---  Tinv (t inverse): eta (1 + (b^{-1/c}-1)/a t)^{-c} ... reaches eta*b after a epochs ---*/
  /*---  Expo: eta pow(b^{1/a}, t)         ... reaches eta*b after a epochs and keeps shrinking ---*/
  /*---  expo: eta MAX(b, pow(b^{1/a}, t)) ... reaches eta*b after a epochs and stops shrinking ---*/  
  /*---  Linear: eta MAX(b, 1 - (1-b)/a t) ... reaches eta*b after a epochs  ---*/
  double aa, bb, cc; 

  double bval; /* used for Exponential and Tinv */
  
public: 
  AzStepszSch() : typ(AzStepszSch_Type_None), decay_index(0), decay_factor(-1), aa(-1), bb(-1), cc(-1), bval(-1) {}

  #define kw_s_typ_old "step_size_scheduler="
  #define kw_decay_factor_old "step_size_decay="
  #define kw_decay_at_old "step_size_decay_at="
  #define kw_aa "step_size_a="
  #define kw_bb "step_size_b="
  #define kw_cc "step_size_c="
  #define kw_s_typ "ss_scheduler="
  #define kw_decay_factor "ss_decay="
  #define kw_decay_at "ss_decay_at="
  
  void resetParam(AzParam &azp, const char *pfx="") {
    const char *eyec = "AzpStepszSch::resetParam";
    azp.reset_prefix(pfx);     
    s_decay_at.reset(); 
    ia_decay_at.reset();   
    typ = AzStepszSch_Type_None; 

    azp.vStr(kw_s_typ, &s_typ, kw_s_typ_old);
    if (s_typ.length() > 0) {
      AzStrPool sp_supported_typ(10,10); 
      sp_supported_typ.put("Tinv", "Expo", "Linear", "Few", "None"); sp_supported_typ.put("expo"); 
      sp_supported_typ.commit();
      AzX::throw_if((sp_supported_typ.find(s_typ.c_str()) < 0), AzInputError, eyec, "unknown scheduler type: ", s_typ.c_str()); 
      typ = *s_typ.point(); 
    }
    if (typ == AzStepszSch_Type_None) {}
    else if (typ == AzStepszSch_Type_Few) {
      azp.vStr(kw_decay_at, &s_decay_at, kw_decay_at_old, kw_aa);
      azp.vFloat(kw_decay_factor, &decay_factor, kw_decay_factor_old, kw_bb);
 
      AzTools::getInts(s_decay_at.c_str(), '_', &ia_decay_at); 
      AzXi::throw_if_nonpositive(decay_factor, eyec, kw_decay_factor);    
    }
    else if (typ == AzStepszSch_Type_Tinv) {
      /*---  Tinv (t inverse): eta (1 + (b^{-1/c}-1)/a t)^{-c} ... reaches eta*b after a epochs ---*/
      cc = 1; /* default */      
      azp.vFloat(kw_aa, &aa);
      azp.vFloat(kw_bb, &bb);
      azp.vFloat(kw_cc, &cc); 
      AzXi::throw_if_nonpositive(aa, eyec, kw_aa); 
      AzXi::throw_if_nonpositive(bb, eyec, kw_bb); 
      AzXi::throw_if_nonpositive(cc, eyec, kw_cc); 
      bval = (pow(bb, -1/cc) - 1)/aa; 
    }
    else if (typ == AzStepszSch_Type_Expo || typ == AzStepszSch_Type_expo || typ == AzStepszSch_Type_Linear) {
      /*---  Expo: eta pow(b^{1/a},t), expo: eta MAX(b,pow(b^{1/a},t)) ... reaches eta*b after a epochs ---*/
      /*---  Linear: eta MAX(b, 1 - (1-b)/a t) ... reaches eta*b after a epochs  ---*/
      azp.vFloat(kw_aa, &aa);
      azp.vFloat(kw_bb, &bb);
      AzXi::throw_if_nonpositive(aa, eyec, kw_aa);       
      AzXi::throw_if_nonpositive(bb, eyec, kw_bb);
      AzX::throw_if((bb >= 1), AzInputError, eyec, kw_bb, "must be smaller than 1"); 
      if (typ == AzStepszSch_Type_Expo || typ == AzStepszSch_Type_expo) {
        bval = pow(bb, 1/aa); 
      }
    }
    azp.reset_prefix(); 
  }
  void printParam(const AzOut &out, const char *pfx="") const {
    AzPrint o(out, pfx); 
    o.printV_if_not_empty(kw_s_typ, s_typ); 
    if (typ == AzStepszSch_Type_None) {}
    else if (typ == AzStepszSch_Type_Few) {
      o.printV(kw_decay_factor, decay_factor); 
      o.printV(kw_decay_at, s_decay_at); 
    }
    else {    
      o.printV(kw_aa, aa);
      o.printV(kw_bb, bb);
      o.printV(kw_cc, cc); 
    }
    o.printEnd(); 
  }
  void printHelp(AzHelp &h) const {
/*    h.item(kw_typ, "Step-size scheduler type.  \"Few\" (reduce it a few times) | \"Linear\" | \"Expo\" (exponential)"); */
    h.item(kw_s_typ, "Step-size scheduler type.  \"Few\" (reduce it a few times). ", "no scheduling");
    h.item(kw_decay_factor, "Use this with \"Few\".  Step-size is reduced by multiplying this value when it is reduced."); 
    h.item(kw_decay_at, "Use this with \"Few\".  Reduce step-size after this many epochs.  To reduce it more than once, use \"_\" as a delimiter, e.g., \"80_90\""); 
/*    h.item(kw_aa, "A: Use this with \"Linear\", \"Expo\", or \"Tinv\"."); */
/*    h.item(kw_bb, "B: Use this with \"Linear\", \"Expo\", or \"Tinv\".  Step-size is reduced after every epoch at the rate that it becomes the initial value times B after A epochs, either linearly or exponentially."); */
/*    h.item(kw_cc, "C: Use this with \"Tinv\".  (1+(B^{-1/C}-1)/A t)^{-C}"); */
  }
  
  void init() {
    decay_index = 0; 
  }
    
  /*------------------------------------------------------------*/ 
  double get_stepsize_coeff(int tt) {
    double coeff = -1; 
    if (typ == AzStepszSch_Type_Few) {
      if (decay_index < ia_decay_at.size()) {
        if (tt >= ia_decay_at.get(decay_index)) {  
          coeff = pow(decay_factor, decay_index+1); 
          ++decay_index; 
        }
      }
    }
    else if (typ == AzStepszSch_Type_Tinv) {   
      /*---  Tinv (t inverse): eta (1 + (b^{-1/c}-1)/a t)^{-c} ... reaches eta*b after a epochs ---*/
      coeff = pow(1 + bval * (double)tt, -cc); 
    }
    else if (typ == AzStepszSch_Type_Expo) {
      /*---  Exponential: eta pow(b^{1/a}, t) ... reaches eta*b after a epochs and keeps shrinking ---*/  
      coeff = pow(bval, (double)(tt-1));  /* 09/18/2013 */
    }
    else if (typ == AzStepszSch_Type_expo) {
      /*---  Exponential: eta pow(b^{1/a}, t) ... reaches eta*b after a epochs and stops shrinking ---*/  
      coeff = MAX(bb, pow(bval, (double)(tt-1)));  /* 11/27/2016 */
    }    
    else if (typ == AzStepszSch_Type_Linear) {
      /*---  Linear: eta (1 - (1-b)/a t) ... reaches eta*b after a epochs  ---*/
      coeff = 1 - (1-bb)/aa * (double)(tt-1); /* 09/18/2013 */
    }
    return coeff; 
  }
};   
 
#endif 
  