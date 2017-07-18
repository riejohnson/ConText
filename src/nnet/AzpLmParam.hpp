/* * * * *
 *  AzpLmParam.hpp
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

#ifndef _AZP_LM_PARAM_HPP_
#define _AZP_LM_PARAM_HPP_

#include "AzUtil.hpp"
#include "AzParam.hpp"
#include "AzPmat.hpp"
#include "AzHelp.hpp"

#define AzpLmParam_reg_infinity 9999
#define AzpLmParam_iw_auto 9999
/* parameters for linear model training.  default is sgd.  */
class AzpLmParam {
public:
  double reg_L2, reg_L1L2, reg_L1L2_delta, reg_L2const, reg_L2init; 
  bool do_l2const_each; 
  double initw_max, initint, initw_rownorm; 
  int initw_nzmax; 
  bool do_initw_nonega; 
  bool do_no_intercept, do_reg_intercept; 
  bool do_count_regions; 
  bool do_iw_uniform; 
  bool do_fixed; 
  bool do_nodiv; 
  double grad_clip, weight_clip; 
  bool do_fixw, do_fixi; /* analysis purposes only */
  bool do_showwi; 
  
  bool no_regadd() const {
    return (reg_L2 <= 0 && reg_L1L2 <= 0 && reg_L2init <= 0);  
  }
  
  AzpLmParam() : initw_max(0.01), initint(0), initw_nzmax(-1), reg_L2(-1), reg_L1L2(-1), reg_L1L2_delta(-1),
                 reg_L2const(-1), reg_L2init(-1), do_count_regions(false), do_fixed(false), do_nodiv(false), 
                 do_no_intercept(false), do_reg_intercept(false), do_initw_nonega(false), do_iw_uniform(false), 
                 grad_clip(-1), weight_clip(-1), do_fixw(false), do_fixi(false), initw_rownorm(-1), do_showwi(false),
                 do_l2const_each(false) {}
         
  /*------------------------------------------------------------*/ 
  #define kw_do_iw_uniform "InitWeightUniform"
  #define kw_initw_max   "init_weight="
  #define kw_initint     "init_intercept="
  #define kw_initw_nzmax "init_weight_nzmax="
  #define kw_initw_rownorm "init_weight_row_norm="
  #define kw_do_initw_nonega "NoNegativeInitW"
  #define kw_reg_L2      "reg_L2="  
  #define kw_reg_L2const "reg_L2const="
  #define kw_do_l2const_each "L2constEach"
  #define kw_reg_L2init  "reg_L2init="  
  #define kw_reg_L1L2    "reg_L1L2="
  #define kw_reg_L1L2_delta "reg_L1L2_delta="
  #define kw_do_no_intercept "NoIntercept"
  #define kw_do_reg_intercept "RegularizeIntercept"
  #define kw_do_count_regions "CountRegions"
  #define kw_do_nodiv "NoDiv"
  #define kw_do_fixed "Fixed"
  #define kw_grad_clip "grad_clip="
  #define kw_weight_clip "weight_clip="
  #define kw_do_fixw "FixW"
  #define kw_do_fixi "FixI"
  #define kw_do_showwi "ShowWI"
  virtual void resetParam(AzParam &azp, const char *pfx, bool is_warmstart=false) {
    azp.reset_prefix(pfx); 
    azp.swOn(&do_fixed, kw_do_fixed); 
    azp.swOn(&do_no_intercept, kw_do_no_intercept, false); 
    if (!is_warmstart) {
      azp.vFloat(kw_initw_max, &initw_max); 
      azp.vInt(kw_initw_nzmax, &initw_nzmax); 
      azp.vFloat(kw_initint, &initint); 
      azp.swOn(&do_initw_nonega, kw_do_initw_nonega, false); 
      azp.swOn(&do_iw_uniform, kw_do_iw_uniform); 
      azp.vFloat(kw_initw_rownorm, &initw_rownorm); 
    }
    else {
      initw_max = initint = initw_nzmax = -1; 
    }
    if (!do_fixed) {
      azp.swOn(&do_reg_intercept, kw_do_reg_intercept);     
      azp.vFloat(kw_reg_L2, &reg_L2); 
      azp.vFloat(kw_reg_L2const, &reg_L2const); 
      azp.swOn(&do_l2const_each, kw_do_l2const_each); 
      azp.vFloat(kw_reg_L2init, &reg_L2init);     
      if (reg_L2init > 0) {
        reg_L2 = MAX(0, reg_L2); 
      }
      azp.vFloat(kw_reg_L1L2, &reg_L1L2); 
      azp.vFloat(kw_reg_L1L2_delta, &reg_L1L2_delta); 

      azp.swOn(&do_count_regions, kw_do_count_regions); 
      azp.swOn(&do_nodiv, kw_do_nodiv, false); 
      azp.vFloat(kw_grad_clip, &grad_clip); 
      azp.vFloat(kw_weight_clip, &weight_clip);       
      azp.swOn(&do_fixw, kw_do_fixw); 
      azp.swOn(&do_fixi, kw_do_fixi); 
    }
    azp.swOn(&do_showwi, kw_do_showwi); 

    azp.reset_prefix(); 
  }  

  virtual void checkParam(const char *pfx) {  
    const char *eyec = "AzpLmParam::checkParam";   
    if (dont_update()) { 
      reg_L2const = reg_L2 = reg_L1L2 = -1;  /* no update -> no regularization */
    }
    if (reg_L2init >= 0 && (reg_L2const > 0 || reg_L1L2 > 0)) {
      AzBytArr s(kw_reg_L2init); s << " is exclusive with: " << kw_reg_L2const << " and " << kw_reg_L1L2; 
      AzX::throw_if(true, AzInputError, eyec, s.c_str()); 
    }
    if (reg_L2 >= 0 && reg_L2const > 0 || reg_L2 >= 0 && reg_L1L2 > 0 || reg_L2const > 0 && reg_L1L2 > 0) {
      AzBytArr s(kw_reg_L2); s << " " << kw_reg_L2const << " " << kw_reg_L1L2; 
      AzX::throw_if(true, AzInputError, "mutually exclusive parameters: ", s.c_str()); 
    }
    AzX::throw_if((reg_L1L2 > 0 && reg_L1L2_delta <= 0), AzInputError, kw_reg_L1L2_delta, "must be positive"); 
    AzXi::throw_if_both((do_no_intercept && do_reg_intercept), eyec, kw_do_no_intercept, kw_do_reg_intercept); 
  }
  
  virtual void printParam(const AzOut &out, const char *pfx) const {
    if (out.isNull()) return; 
    AzPrint o(out, pfx); 
    o.printV(kw_initw_max, initw_max); 
    o.printV(kw_initw_nzmax, initw_nzmax); 
    o.printV(kw_initw_rownorm, initw_rownorm); 
    o.printV(kw_initint, initint); 
    o.printSw(kw_do_initw_nonega, do_initw_nonega); 
    o.printSw(kw_do_iw_uniform, do_iw_uniform); 
    o.printSw(kw_do_fixed, do_fixed); 
    o.printV(kw_reg_L2, reg_L2);     
    o.printV(kw_reg_L2const, reg_L2const);  
    o.printSw(kw_do_l2const_each, do_l2const_each); 
    o.printV(kw_reg_L2init, reg_L2init); 
    o.printV(kw_reg_L1L2, reg_L1L2); 
    o.printV(kw_reg_L1L2_delta, reg_L1L2_delta);
    o.printSw(kw_do_no_intercept, do_no_intercept); 
    o.printSw(kw_do_reg_intercept, do_reg_intercept); 

    o.printSw(kw_do_count_regions, do_count_regions); 
    o.printSw(kw_do_nodiv, do_nodiv); 
    
    o.printV(kw_grad_clip, grad_clip);
    o.printV(kw_weight_clip, weight_clip);    
    o.printSw(kw_do_fixw, do_fixw); 
    o.printSw(kw_do_fixi, do_fixi); 
    o.printSw(kw_do_showwi, do_showwi); 
    o.printEnd(); 
  } 
  virtual void printHelp(AzHelp &h) const {
    AzBytArr s("x: scale of weight initialization.  Weights will be initialized by Gaussian distribution with zero mean with standard deviation x.  If "); s << kw_do_iw_uniform << " is on, initial weights will be in [-x,x]."; 
    h.item(kw_initw_max, s.c_str(), 0.01); 
    h.item(kw_do_iw_uniform, "Initialize weights with uniform distribution.", "Gaussian distribution");    
    h.item(kw_initint, "Initial values of intercepts.", 0); 
    h.item(kw_do_fixed, "Do not update weights or intercepts.  Fix them after initialization."); 
    h.item(kw_reg_L2, "L2 regularization parameter."); 
    h.item(kw_do_no_intercept, "Do not use intercepts.");     
    h.item(kw_do_reg_intercept, "Regularize intercepts."); 
    /* kw_initw_nzmax kw_do_initw_nonega kw_reg_L2const kw_reg_L2init */
    /* kw_reg_L1L2 kw_reg_L1L2_delta kw_do_count_regions kw_grad_clip */
  }
  virtual void force_no_reg() { reg_L2=reg_L1L2=reg_L2const=reg_L2init=-1; }  
  inline virtual bool dont_update() const { 
    return (do_fixed || reg_L2init >= AzpLmParam_reg_infinity); 
  }
}; 
#endif 